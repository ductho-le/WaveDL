import { clientError } from '../i18n/client-messages';
import { ConnectionSession, ConnectionSessionError } from './connection-session';
import { toRaw } from 'vue';
import { DaemonApiError, isUuid, type Json } from './daemon-api';

import { inventorySchema } from '../../shared/schemas/artifacts';

export interface Artifact {
  readonly runId: string;
  readonly path: string;
  readonly size: number;
  readonly modified: string;
}

export interface Checkpoint {
  readonly runId: string;
  readonly name: string;
}

export interface Inventory {
  readonly files: readonly Artifact[];
  readonly checkpoints: readonly Checkpoint[];
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

const scopes = new WeakMap<
  Artifact | Checkpoint,
  { session: ConnectionSession; revision: number }
>();

const fail = (): never => {
  throw DaemonApiError.fromKey('response', 'invalidArtifactInventory');
};

export function parseInventory(value: Json, runId: string): Inventory {
  const parsed = inventorySchema.safeParse(value);

  if (!isUuid(runId) || !parsed.success) return fail();
  const { files, checkpoints } = parsed.data;

  if (
    new Set(files.map((file) => file.path)).size !== files.length ||
    new Set(checkpoints).size !== checkpoints.length
  )
    return fail();

  return Object.freeze({
    files: Object.freeze(files.map((file) => Object.freeze({ runId, ...file }))),
    checkpoints: Object.freeze(checkpoints.map((name) => Object.freeze({ runId, name }))),
  });
}

/** Inventory grants are memory-only and belong to one verified connection. */
export class ArtifactClient {
  readonly session: ConnectionSession;
  constructor(session: ConnectionSession) {
    this.session = toRaw(session);
  }
  async list(runId: string, signal?: AbortSignal): Promise<Inventory> {
    if (!isUuid(runId)) throw clientError('invalidRunIdentity', {});
    const revision = this.session.snapshot().revision;

    const inventory = await this.session.perform(
      async (api, scoped) =>
        parseInventory(await api.read(`/api/runs/${runId}/artifacts`, scoped), runId),
      { signal },
    );

    if (revision !== this.session.snapshot().revision || !this.session.api)
      throw ConnectionSessionError.fromKey('stale', 'artifactRefreshRequired');

    for (const record of [...inventory.files, ...inventory.checkpoints])
      scopes.set(record, { session: this.session, revision });

    return inventory;
  }
  assertSelection(record: Artifact | Checkpoint): void {
    const scope = scopes.get(record);

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

    const blob = await this.session.perform(
      async (api, scoped) => {
        this.assertSelection(artifact);
        const path = artifact.path.split('/').map(encodeURIComponent).join('/');

        const response = await api.downloadResponse(
          `/api/runs/${artifact.runId}/artifacts/${path}`,
          scoped,
        );

        // A live run rewrites its files, so the listed size can be stale.
        const length = response.headers.get('content-length') ?? '';
        const size = /^\d+$/.test(length) ? Number(length) : Number.NaN;

        if (
          !response.body ||
          response.headers.get('content-type') !== 'application/octet-stream' ||
          !Number.isSafeInteger(size)
        ) {
          await response.body?.cancel();
          throw DaemonApiError.fromKey('response', 'artifactInventoryMismatch');
        }

        // Returned rather than thrown: a failure inside perform ends the session.
        if (size > BLOB_BUDGET) {
          await response.body.cancel();

          return null;
        }

        const reader = response.body.getReader();
        const bytes = new Uint8Array(size);
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

            if (offset + value.length > size)
              throw DaemonApiError.fromKey('response', 'artifactExceedsDeclaredSize');
            bytes.set(value, offset);
            offset += value.length;
          }

          if (offset !== size) throw DaemonApiError.fromKey('response', 'artifactIsTruncated');
          scoped.throwIfAborted();
          api.assertResponseCurrent(response);
          this.assertSelection(artifact);

          return new Blob([bytes], { type: 'application/octet-stream' });
        } finally {
          scoped.removeEventListener('abort', abort);
          await reader.cancel().catch(() => undefined);
          reader.releaseLock();
        }
      },
      { signal, timeoutMs: 60000 },
    );

    if (!blob) throw clientError('artifactMemoryBudget', {});

    return blob;
  }
}
