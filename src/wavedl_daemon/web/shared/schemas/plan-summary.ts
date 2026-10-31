import { z } from 'zod';
import { jsonObjectSchema } from './json';

const planReviewSchema = z.object({
  operation: z.string().optional().catch(undefined),
  resolved_config: jsonObjectSchema.catch({}).default({}),
  runtime: z.object({ device: z.string().optional().catch(undefined) }).catch({}),
});

export const planManifestSchema = jsonObjectSchema.transform((manifest) => ({
  manifest,
  review: planReviewSchema.parse(manifest),
}));

export const planPreviewContextSchema = z.object({
  target: z.string().optional().catch(undefined),
  topology: jsonObjectSchema.catch({}).default({}),
});
