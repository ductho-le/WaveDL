import { clientError } from '../i18n/client-messages';
import { ConnectionSession, ConnectionSessionError } from './connection-session';
import { toRaw } from 'vue';
import { DaemonApiError, isUuid, parseJson, type Json } from './daemon-api';

import { artifactPathSchema, inventorySchema, selectionMetadataSchema } from '../../shared/schemas/artifacts';
import type { z } from 'zod';

export interface Artifact {
  readonly runId: string;
  readonly generation: string;
  readonly attempt: number;
  readonly step: number;
  readonly path: string;
  readonly size: number;
  readonly sha256: string;
}

export const BLOB_BUDGET = 64 * 1024 * 1024;

export async function pngPreviewBlob(blob: Blob): Promise<Blob> {
  const header = new Uint8Array(await blob.slice(0, 33).arrayBuffer());

  if (
    header.length !== 33 ||
    header.slice(0, 8).join(',') !== '137,80,78,71,13,10,26,10' ||
    header.slice(12, 16).join(',') !== '73,72,68,82'
  )
    throw clientError('notARasterPng', {});
  const view = new DataView(header.buffer);
  const width = view.getUint32(16);
  const height = view.getUint32(20);

  if (
    view.getUint32(8) !== 13 ||
    width < 1 ||
    height < 1 ||
    width > 8192 ||
    height > 8192 ||
    width * height > 16 * 1024 * 1024
  )
    throw clientError('pngPixelBudget', {});

  return new Blob([blob], { type: 'image/png' });
}

const scopes = new WeakMap<Artifact, { session: ConnectionSession; revision: number }>();

const fail = (): never => {
  throw DaemonApiError.fromKey('response', 'invalidArtifactInventory');
};

export function artifactPath(value: unknown): value is string {
  return artifactPathSchema.safeParse(value).success;
}

export function parseInventory(value: Json, runId: string): Artifact[] {
  const parsed = inventorySchema.safeParse(value);

  if (!isUuid(runId) || !parsed.success) return fail();
  const result: Artifact[] = [];
  const seen = new Set<string>();

  for (const item of parsed.data.generations) {
    if (seen.has(item.generation)) return fail();
    seen.add(item.generation);

    for (const [path, record] of Object.entries(item.files)) {
      if (result.length >= 16384) return fail();
      result.push(Object.freeze({
        runId,
        generation: item.generation,
        attempt: item.attempt,
        step: item.step,
        path,
        ...record,
      }));
    }
  }

  return result;
}

/** Inventory grants are memory-only and belong to one verified connection. */
export class ArtifactClient {
  readonly session: ConnectionSession;
  constructor(session: ConnectionSession) {
    this.session = toRaw(session);
  }
  async list(runId: string, signal?: AbortSignal): Promise<readonly Artifact[]> {
    if (!isUuid(runId)) throw clientError('invalidRunIdentity', {});
    const revision = this.session.snapshot().revision;

    const records = await this.session.perform(
      async (api, scoped) =>
        parseInventory(await api.read(`/api/runs/${runId}/artifacts`, scoped), runId),
      { signal },
    );

    if (revision !== this.session.snapshot().revision || !this.session.api)
      throw ConnectionSessionError.fromKey('stale', 'artifactRefreshRequired');

    for (const record of records) scopes.set(record, { session: this.session, revision });

    return Object.freeze(records);
  }
  assertSelection(artifact: Artifact): void {
    const scope = scopes.get(artifact);

    if (
      !scope ||
      scope.session !== this.session ||
      scope.revision !== this.session.snapshot().revision ||
      !this.session.api
    )
      throw ConnectionSessionError.fromKey('stale', 'artifactRefreshRequired');
  }
  async verifiedBlob(artifact: Artifact, signal?: AbortSignal): Promise<Blob> {
    this.assertSelection(artifact);

    if (artifact.size > BLOB_BUDGET) throw clientError('artifactMemoryBudget', {});

    return this.session.perform(
      async (api, scoped) => {
        this.assertSelection(artifact);
        const path = artifact.path.split('/').map(encodeURIComponent).join('/');

        const response = await api.downloadResponse(
          `/api/runs/${artifact.runId}/artifacts/${artifact.generation}/${path}`,
          scoped,
        );

        if (
          !response.body ||
          response.headers.get('content-type') !== 'application/octet-stream' ||
          response.headers.get('content-length') !== String(artifact.size)
        ) {
          await response.body?.cancel();
          throw DaemonApiError.fromKey('response', 'artifactInventoryMismatch');
        }

        const reader = response.body.getReader();
        const bytes = new Uint8Array(artifact.size);
        let offset = 0;

        const abort = () => {
          void reader.cancel();
        };

        scoped.addEventListener('abort', abort, { once: true });

        try {
          for (;;) {
            scoped.throwIfAborted();
            api.assertResponseCurrent(response);
            const { done, value } = await reader.read();

            if (done) break;

            if (offset + value.length > artifact.size)
              throw DaemonApiError.fromKey('response', 'artifactExceedsDeclaredSize');
            bytes.set(value, offset);
            offset += value.length;
          }

          if (offset !== artifact.size)
            throw DaemonApiError.fromKey('response', 'artifactIsTruncated');
          const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
          const hash = Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
          scoped.throwIfAborted();
          api.assertResponseCurrent(response);
          this.assertSelection(artifact);

          if (hash !== artifact.sha256)
            throw DaemonApiError.fromKey('response', 'artifactChecksumChanged');

          return new Blob([bytes], { type: 'application/octet-stream' });
        } finally {
          scoped.removeEventListener('abort', abort);
          await reader.cancel().catch(() => undefined);
          reader.releaseLock();
        }
      },
      { signal, timeoutMs: 60000 },
    );
  }
  async selectionMetadata(artifact: Artifact, signal?: AbortSignal): Promise<SelectionMetadata> {
    if (artifact.path !== 'checkpoint_selection.json' || artifact.size > 4096)
      throw clientError('checkpointMetadataRequired', {});
    const blob = await this.verifiedBlob(artifact, signal);

    return parseSelectionMetadata(parseJson(await blob.text()));
  }
}

export type SelectionMetadata = z.infer<typeof selectionMetadataSchema>;

export function parseSelectionMetadata(value: Json): SelectionMetadata {
  const parsed = selectionMetadataSchema.safeParse(value);

  if (!parsed.success) return fail();
  const result: SelectionMetadata = { schema_version: 1, kind: parsed.data.kind };

  if (parsed.data.fold !== undefined) result.fold = parsed.data.fold;

  if (parsed.data.source_seq !== undefined) result.source_seq = parsed.data.source_seq;

  return Object.freeze(result);
}
