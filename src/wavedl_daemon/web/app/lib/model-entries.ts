import type { ModelDescription } from './catalog-api';
import { parseJson } from './daemon-api';
import { recentModelsSchema } from '../../shared/schemas/model-recents';

/**
 * One architecture as the user thinks of it. The catalogue lists `x` and
 * `x_pretrained` separately; weights only load when the run also sets `pretrained`.
 */
export interface ModelEntry {
  readonly key: string;
  /** Trains from scratch; null when the architecture only ships a weights-capable id. */
  readonly scratch: ModelDescription | null;
  /** Accepts `pretrained: true`; null when no ImageNet weights exist. */
  readonly weighted: ModelDescription | null;
  /** The description shown for the entry (the from-scratch variant when both exist). */
  readonly primary: ModelDescription;
}

const suffix = '_pretrained';

export function modelEntryKey(id: string): string {
  return id.endsWith(suffix) ? id.slice(0, -suffix.length) : id;
}

export function modelEntries(models: readonly ModelDescription[]): ModelEntry[] {
  const groups = new Map<string, ModelDescription[]>();

  for (const model of models) {
    const key = modelEntryKey(model.id);
    groups.set(key, [...(groups.get(key) ?? []), model]);
  }

  return [...groups.entries()].flatMap(([key, variants]) => {
    const scratch = variants.find((model) => !model.pretrained) ?? null;

    const weighted =
      variants.find((model) => model.pretrained && model.id.endsWith(suffix)) ??
      variants.find((model) => model.pretrained) ??
      null;

    const primary = scratch ?? weighted;

    return primary ? [{ key, scratch, weighted, primary }] : [];
  });
}

/** The catalogue id a run uses for this entry, given the weights choice. */
export function entryModel(entry: ModelEntry, pretrained: boolean): ModelDescription {
  return (pretrained ? entry.weighted : entry.scratch) ?? entry.primary;
}

const recentKey = 'wavedl.recent-models.v1';
const recentLimit = 4;

export function loadRecentModels(storage: Pick<Storage, 'getItem'> | null): string[] {
  try {
    const raw = storage?.getItem(recentKey);
    const value = raw ? parseJson(raw) : [];

    const parsed = recentModelsSchema.safeParse(value);

    return parsed.success ? parsed.data : [];
  } catch {
    return [];
  }
}

export function rememberRecentModel(
  storage: Pick<Storage, 'getItem' | 'setItem'> | null,
  key: string,
): string[] {
  const next = [key, ...loadRecentModels(storage).filter((item) => item !== key)].slice(
    0,
    recentLimit,
  );

  try {
    storage?.setItem(recentKey, JSON.stringify(next));
  } catch {
    // Recents are a convenience; private browsing may refuse storage.
  }

  return next;
}
