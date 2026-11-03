import { z } from 'zod';
import { jsonObjectSchema } from './json';
import { trainingConfigSchema } from './training-request';

export const approvedTrainingManifestSchema = jsonObjectSchema.and(z.object({
  run_id: z.string(),
  operation: z.literal('train'),
  resolved_config: jsonObjectSchema,
  runtime: jsonObjectSchema.and(z.object({ container: z.null() })),
}));

export const trainingPlanPreviewSchema = z.object({
  topology: jsonObjectSchema,
  target: z.string(),
});

export const approvedTrainingConfigSchema = jsonObjectSchema.refine((value) =>
  Object.keys(value).every((key) => key === 'output_dir' || Object.hasOwn(trainingConfigSchema.shape, key)),
).transform((value) => Object.fromEntries(
  Object.entries(value).filter(([key]) => key !== 'output_dir'),
));

export const sourcePlotManifestSchema = jsonObjectSchema.and(z.object({ resolved_config: jsonObjectSchema }));

export const approvedPlotManifestSchema = sourcePlotManifestSchema.and(z.object({
  operation: z.literal('plot'),
  run_id: z.string(),
  checkpoint_source: jsonObjectSchema.and(z.object({
    run_id: z.string(),
    generation: z.string(),
    manifest_sha256: z.string(),
    complete_sha256: z.string().regex(/^[0-9a-f]{64}$/),
  })),
}));
