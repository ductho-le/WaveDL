import { z } from 'zod';
import { jsonSchema, jsonObjectSchema } from '../../shared/schemas/json';
import { trainingRequestSchema } from '../../shared/schemas/training-request';
import { resultRequestSchema } from '../../shared/schemas/result-request';

// Deliberately permits omitted members for malformed-response fixtures.
export const fixtureResponseSchema = z.union([
  jsonSchema,
  z.record(z.string(), jsonSchema.optional()),
]);

export type FixtureResponse = z.input<typeof fixtureResponseSchema>;

export const fixtureEventOverridesSchema = z.record(z.string(), jsonSchema);

export type FixtureEventOverrides = z.input<typeof fixtureEventOverridesSchema>;

export const fixturePlanRequestSchema = z.union([trainingRequestSchema, resultRequestSchema]);

// Fixtures deliberately mutate these fields into unsupported manifest values.
export const fixtureTrainingManifestSchema = jsonObjectSchema.and(
  z.object({
    resolved_config: jsonObjectSchema,
    runtime: jsonObjectSchema,
  }),
);
