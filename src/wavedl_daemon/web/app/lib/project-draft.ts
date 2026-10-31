import { clientError } from '../i18n/client-messages';
import {
  draftConfigFields,
  draftConfigSchema,
  projectDraftSchema,
  type DraftConfig,
  type ProjectBlock,
  type ProjectDraft,
} from '../../shared/schemas/project-draft';
import { storageMappingSchema, type StorageMapping } from '../../shared/schemas/storage-mapping';
import type { TrainingRequest } from '../../shared/schemas/training-request';
import type { RuntimeSetup } from '../../shared/schemas/host-profile';
import type { DataSelection, ModelSelection } from './catalog-api';
import { parseJson } from './daemon-api';
import type { TrainingImportDocument } from '../../shared/schemas/training-import';
import { boundedText, browserStorage, type PreferenceStorage } from './preferences';

function draftKey(daemonId: string): string {
  if (!daemonId || daemonId.length > 256 || [...daemonId].some((char) => char.charCodeAt(0) < 32))
    throw clientError('invalidDaemonIdentity', {});

  return `wavedl.training-draft.v1.${encodeURIComponent(daemonId)}`;
}

/** Keeps only individually valid values, so one half-typed field never discards the rest. */
export function validDraftConfig(config: TrainingImportDocument): DraftConfig {
  return draftConfigFields.reduce<DraftConfig>((valid, name) => {
    if (config[name] === undefined) return valid;
    const parsed = draftConfigSchema.safeParse({ [name]: config[name] });

    return parsed.success ? { ...valid, ...parsed.data } : valid;
  }, {});
}

function validTopology(topology: TrainingRequest['topology']): ProjectDraft['topology'] {
  const topologySchema = projectDraftSchema.shape.topology;

  return [
    { num_machines: topology.num_machines },
    { processes_per_machine: topology.processes_per_machine },
    { machine_rank: topology.machine_rank },
  ].reduce<ProjectDraft['topology']>((valid, part) => {
    const parsed = topologySchema.safeParse(part);

    return parsed.success ? { ...valid, ...parsed.data } : valid;
  }, {});
}

function validMappings(mappings: TrainingRequest['storage_map']): StorageMapping[] {
  return (mappings ?? []).flatMap((mapping) => {
    const parsed = storageMappingSchema.safeParse(mapping);

    return parsed.success ? [parsed.data] : [];
  });
}

export interface DraftSnapshot {
  readonly block: ProjectBlock;
  readonly data: DataSelection | null;
  readonly model: ModelSelection | null;
  readonly request: TrainingRequest;
  readonly setup: RuntimeSetup;
}

export function projectDraft(daemonId: string, snapshot: DraftSnapshot): ProjectDraft {
  const { request } = snapshot;

  return projectDraftSchema.parse({
    version: 1,
    daemonId,
    savedAt: Date.now(),
    block: snapshot.block,
    dataset: snapshot.data
      ? { root: snapshot.data.root, relativePath: snapshot.data.relativePath }
      : null,
    model: snapshot.model
      ? { id: snapshot.model.model, custom: snapshot.model.source === 'custom' }
      : null,
    backend: request.backend,
    setup: snapshot.setup,
    config: validDraftConfig(request.config),
    runtime: {
      python: request.runtime.python.slice(0, 8192),
      mode: request.runtime.mode,
      device: request.runtime.device,
    },
    topology: validTopology(request.topology),
    storage_map: validMappings(request.storage_map),
  });
}

export function saveProjectDraft(
  daemonId: string,
  snapshot: DraftSnapshot,
  storage: PreferenceStorage | null = browserStorage(),
): boolean {
  try {
    if (!storage) return false;
    storage.setItem(
      draftKey(daemonId),
      boundedText(JSON.stringify(projectDraft(daemonId, snapshot))),
    );

    return true;
  } catch {
    return false;
  }
}

export function loadProjectDraft(
  daemonId: string,
  storage: PreferenceStorage | null = browserStorage(),
): ProjectDraft | null {
  try {
    const raw = storage?.getItem(draftKey(daemonId));

    if (!raw) return null;
    const draft = projectDraftSchema.parse(parseJson(boundedText(raw)));

    return draft.daemonId === daemonId ? draft : null;
  } catch {
    return null;
  }
}

export function clearProjectDraft(
  daemonId: string,
  storage: PreferenceStorage | null = browserStorage(),
): void {
  try {
    storage?.removeItem(draftKey(daemonId));
  } catch {
    // Storage may be unavailable (private mode); an unsaved draft is not an error.
  }
}

/** Older drafts predate automatic setup: an interpreter they name was chosen by hand. */
export function draftSetup(draft: ProjectDraft): RuntimeSetup {
  return draft.setup ?? (draft.runtime.python.trim() ? 'manual' : 'automatic');
}

/** Draft values over a fresh request; environment values never round-trip. */
export function draftRequest(draft: ProjectDraft, fresh: TrainingRequest): TrainingRequest {
  return {
    ...fresh,
    backend: draft.backend,
    config: { ...fresh.config, ...draft.config },
    runtime: { ...fresh.runtime, ...draft.runtime },
    topology: { ...fresh.topology, ...draft.topology },
    storage_map: draft.storage_map.length ? draft.storage_map : undefined,
  };
}
