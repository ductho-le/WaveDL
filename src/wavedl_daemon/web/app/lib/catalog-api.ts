import { ConnectionSession, ConnectionSessionError } from './connection-session';
import { DaemonApiError } from './daemon-api';
import { hostProfileSchema, type HostProfile } from '../../shared/schemas/host-profile';
import {
  browseLocationSchema,
  browseSelectionSchema,
  browseRootsSchema,
  browsePageSchema,
  resolvedDatasetSchema,
  datasetHostPathSchema,
  relativePathSchema,
  modelCatalogSchema,
  customModelSelectionSchema,
} from '../../shared/schemas/catalog';
import type { Json } from './daemon-api';

export interface BrowseEntry {
  readonly name: string;
  readonly kind: 'file' | 'directory';
}

export interface BrowsePage {
  readonly root: string;
  readonly path: string;
  readonly entries: readonly BrowseEntry[];
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
}

export interface DataSelection {
  readonly path: string;
  readonly root: string;
  readonly relativePath: string;
  readonly daemonId: string;
  readonly kind: 'file' | 'directory';
}

export interface ModelDescription {
  readonly id: string;
  readonly displayName: string;
  readonly architectureFamily: string;
  readonly authorInstitution: string;
  readonly yearPublished: number;
  readonly paramsM: number;
  readonly sizeMb: number;
  readonly shortDescription: string;
  readonly description: string;
  readonly bestFor: string;
  readonly pros: readonly string[];
  readonly cons: readonly string[];
  readonly computeTier: string;
  readonly tags: readonly string[];
  readonly supportedDims: readonly number[];
  readonly category: string;
  readonly pretrained: boolean;
}

export interface ModelSelection {
  readonly model: string;
  readonly daemonId: string;
  readonly source?: 'custom';
  readonly runtimeVerified?: false;
}

type Selection = DataSelection | ModelSelection;

const scopes = new WeakMap<Selection, { session: ConnectionSession; revision: number }>();

const catalogues = new WeakMap<
  readonly ModelDescription[],
  { session: ConnectionSession; revision: number }
>();

const fail = (): never => {
  throw DaemonApiError.fromKey('response', 'invalidDaemonCatalogResponse');
};

export function relativePath(value: Json | undefined): value is string {
  return relativePathSchema.safeParse(value).success;
}

/** Catalogues describe installed metadata; they never certify runtime capability. */
export class CatalogApi {
  constructor(readonly session: ConnectionSession) {}
  assertSelection(selection: Selection): void {
    const scope = scopes.get(selection);

    if (
      !scope ||
      scope.session !== this.session ||
      scope.revision !== this.session.snapshot().revision ||
      !this.session.api ||
      selection.daemonId !== this.session.api.daemonId
    )
      throw ConnectionSessionError.fromKey('stale', 'catalogSelectionStale');
  }
  #selection<T extends Selection>(selection: T): T {
    scopes.set(selection, { session: this.session, revision: this.session.snapshot().revision });
    Object.freeze(selection);

    return selection;
  }
  /**
   * Advisory: an older daemon, a slow probe or a malformed reply yields null and
   * never retires the connection, so request failures stay inside the work.
   */
  async host(signal?: AbortSignal): Promise<HostProfile | null> {
    const body = await this.session.perform(
      async (api, requestSignal) => {
        try {
          return await api.read(
            '/api/host',
            AbortSignal.any([requestSignal, AbortSignal.timeout(45_000)]),
          );
        } catch {
          return null;
        }
      },
      { signal, timeoutMs: 60_000 },
    );

    const parsed = hostProfileSchema.safeParse(body);

    return parsed.success ? Object.freeze(parsed.data) : null;
  }
  roots(signal?: AbortSignal): Promise<readonly string[]> {
    return this.session.perform(
      async (api, requestSignal) => {
        const body = await api.read('/api/browse', requestSignal);

        const parsed = browseRootsSchema.safeParse(body);

        if (!parsed.success) return fail();
        const roots = parsed.data.roots.map((item) => item.id);

        return Object.freeze(roots);
      },
      { signal },
    );
  }
  browse(root: string, path = '', offset = 0, signal?: AbortSignal): Promise<BrowsePage> {
    if (!browseLocationSchema.safeParse({ root, path, offset }).success)
      return Promise.reject(ConnectionSessionError.fromKey('request', 'invalidBrowseLocation'));
    const limit = 100;

    return this.session.perform(
      async (api, requestSignal) => {
        const query = new URLSearchParams({
          root,
          path,
          offset: String(offset),
          limit: String(limit),
        });

        const body = await api.read(`/api/browse?${query}`, requestSignal);

        const parsed = browsePageSchema.safeParse(body);

        if (
          !parsed.success ||
          parsed.data.root !== root ||
          parsed.data.path !== path ||
          parsed.data.offset !== offset
        )
          return fail();
        const entries = parsed.data.entries.map((item) => Object.freeze(item));

        return Object.freeze({
          root,
          path,
          offset,
          limit,
          total: parsed.data.total,
          entries: Object.freeze(entries),
        });
      },
      { signal },
    );
  }
  resolve(root: string, path: string, signal?: AbortSignal): Promise<DataSelection> {
    if (!browseSelectionSchema.safeParse({ root, path }).success)
      return Promise.reject(ConnectionSessionError.fromKey('request', 'invalidBrowseSelection'));

    return this.session.perform(
      async (api, requestSignal) => {
        const body = await api.mutate(
          '/api/browse/resolve',
          { root, path },
          { signal: requestSignal },
        );

        const parsed = resolvedDatasetSchema.safeParse(body);

        if (
          !parsed.success ||
          parsed.data.root !== root ||
          parsed.data.relative_path !== path ||
          !api.daemonId
        )
          return fail();
        const resolved = parsed.data;

        return this.#selection({
          path: resolved.path,
          root,
          relativePath: path,
          kind: resolved.kind,
          daemonId: api.daemonId,
        });
      },
      { signal },
    );
  }
  /**
   * Maps an absolute host path (from an imported config) onto a configured browse
   * root. Null when no root contains it; the daemon still verifies the final path.
   */
  async locate(path: string, signal?: AbortSignal): Promise<DataSelection | null> {
    const parsed = datasetHostPathSchema.safeParse(path);

    if (!parsed.success) return null;
    const target = parsed.data.replaceAll('\\', '/');

    for (const root of await this.roots(signal)) {
      let base: DataSelection;

      try {
        base = await this.resolve(root, '', signal);
      } catch (error) {
        // One unreadable root must not hide a match under another.
        if (signal?.aborted) throw error;
        continue;
      }

      const prefix = base.path.replaceAll('\\', '/').replace(/\/?$/, '/');
      const relative = target.startsWith(prefix) ? target.slice(prefix.length) : '';

      const parsedRelative = relativePathSchema.safeParse(relative);

      if (!relative || !parsedRelative.success) continue;

      try {
        return await this.resolve(root, parsedRelative.data, signal);
      } catch (error) {
        if (signal?.aborted) throw error;

        return null;
      }
    }

    return null;
  }
  models(signal?: AbortSignal): Promise<readonly ModelDescription[]> {
    return this.session.perform(
      async (api, requestSignal) => {
        const body = await api.read('/api/models', requestSignal);

        if (new TextEncoder().encode(JSON.stringify(body)).length > 256 * 1024) return fail();
        const parsed = modelCatalogSchema.safeParse(body);

        if (!parsed.success) return fail();
        const categories = new Map<string, string>();

        for (const [name, category] of parsed.data.categories) {
          for (const id of category.models) {
            if (categories.has(id)) return fail();
            categories.set(id, name);
          }
        }

        const models = parsed.data.models.map((item) => {
          if (!categories.has(item.id)) return fail();

          return Object.freeze({
            id: item.id,
            displayName: item.display_name,
            architectureFamily: item.architecture_family,
            authorInstitution: item.author_institution,
            yearPublished: item.year_published,
            paramsM: item.params_m,
            sizeMb: item.size_mb,
            shortDescription: item.short_description,
            description: item.description,
            bestFor: item.best_for,
            pros: Object.freeze([...item.pros]),
            cons: Object.freeze([...item.cons]),
            computeTier: item.compute_tier,
            tags: Object.freeze([...item.tags]),
            supportedDims: Object.freeze([...item.supported_dims]),
            category: categories.get(item.id) ?? '',
            pretrained: item.is_pretrained,
          });
        });

        if (new Set(models.map((item) => item.id)).size !== models.length) return fail();
        const catalogue = Object.freeze(models);
        catalogues.set(catalogue, {
          session: this.session,
          revision: this.session.snapshot().revision,
        });

        return catalogue;
      },
      { signal },
    );
  }
  selectModel(model: ModelDescription, catalogue: readonly ModelDescription[]): ModelSelection {
    const daemonId = this.session.api?.daemonId;
    const scope = catalogues.get(catalogue);

    if (
      !daemonId ||
      !scope ||
      scope.session !== this.session ||
      scope.revision !== this.session.snapshot().revision ||
      !catalogue.includes(model)
    )
      throw ConnectionSessionError.fromKey('stale', 'modelCatalogReloadRequired');

    return this.#selection({ model: model.id, daemonId });
  }
  /** Explicit registry identifier only: required imports belong to the run request. */
  selectCustomModel(modelId: string, signal?: AbortSignal): Promise<ModelSelection> {
    if (!customModelSelectionSchema.safeParse({ model: modelId }).success)
      return Promise.reject(ConnectionSessionError.fromKey('request', 'modelIdInvalid'));

    return this.session.perform(
      async (api, requestSignal) => {
        const health = await api.health(requestSignal);

        return this.#selection({
          model: modelId,
          daemonId: health.daemon_id,
          source: 'custom',
          runtimeVerified: false,
        });
      },
      { signal },
    );
  }
}
