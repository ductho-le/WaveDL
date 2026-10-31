import { z } from 'zod';
import { jsonContainerSchema } from './json';

export const trainingImportSchema = jsonContainerSchema;

export type TrainingImportDocument = z.output<typeof trainingImportSchema>;

const selection = z.string().max(8192).refine((value) => value.trim().length > 0).nullable().catch(null);

export const importedSelectionsSchema = z.object({
  data_path: selection,
  model: selection,
});
