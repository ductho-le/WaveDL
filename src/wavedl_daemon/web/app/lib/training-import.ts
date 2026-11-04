import { parse } from 'yaml';
import { clientError } from '../i18n/client-messages';
import type { DraftConfig } from '../../shared/schemas/project-draft';
import { validDraftConfig } from './project-draft';
import {
  trainingImportSchema,
  importedSelectionsSchema,
  type TrainingImportDocument,
} from '../../shared/schemas/training-import';

export const CONFIG_IMPORT_LIMIT = 64 * 1024;

export interface ImportedTrainingConfig {
  readonly config: DraftConfig;
  readonly dataPath: string | null;
  readonly model: string | null;
  /** Keys present in the file that were not applied (unknown, daemon-owned or invalid). */
  readonly skipped: readonly string[];
}

// The CLI writes these as YAML lists; the draft keeps comma-separated text.
const commaSeparated = new Set(['loss_weights', 'betas', 'milestones']);
const listFields = new Set(['import_modules', 'constraint', 'constraint_weight']);

function normalizedTrainingConfig(document: TrainingImportDocument): TrainingImportDocument {
  return Object.fromEntries(
    Object.entries(document).map(([name, value]): [string, TrainingImportDocument[string]] => {
      if (commaSeparated.has(name) && Array.isArray(value)) return [name, value.join(',')];

      if (listFields.has(name) && value !== null && !Array.isArray(value)) return [name, [value]];

      return [name, value];
    }),
  );
}

/**
 * Reads a WaveDL training YAML (exported here, or a CLI experiment/effective config).
 * JSON is valid YAML, so JSON presets import through the same path.
 */
export function parseTrainingConfig(source: string): ImportedTrainingConfig {
  if (new TextEncoder().encode(source).byteLength > CONFIG_IMPORT_LIMIT)
    throw clientError('configImportTooLarge', {});
  let input: unknown;

  try {
    input = parse(source, { maxAliasCount: 0, uniqueKeys: true });
  } catch {
    throw clientError('configImportUnreadable', {});
  }

  const parsed = trainingImportSchema.safeParse(input);

  if (!parsed.success) throw clientError('configImportUnreadable', {});
  const document = parsed.data;
  const values = normalizedTrainingConfig(document);

  // `mixed_precision` is the CLI alias of `precision`.
  if (values.precision === undefined && values.mixed_precision !== undefined)
    values.precision = values.mixed_precision;
  const config = validDraftConfig(values);
  const applied = new Set<string>(Object.keys(config));

  if (applied.has('precision') && document.mixed_precision !== undefined)
    applied.add('mixed_precision');
  const selections = importedSelectionsSchema.parse(document);
  const dataPath = selections.data_path;
  const model = selections.model;

  if (dataPath) applied.add('data_path');

  if (model) applied.add('model');

  return {
    config,
    dataPath,
    model,
    skipped: Object.keys(document).filter((name) => !applied.has(name)),
  };
}
