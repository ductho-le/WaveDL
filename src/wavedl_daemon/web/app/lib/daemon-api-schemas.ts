import { z } from 'zod';
import { jsonObjectSchema, jsonSchema } from '../../shared/schemas/json';

// Run only after boundedJson/parseJson validate the JSON domain. All response
// objects historically permit additive fields; DTO schemas strip only at return.
export { jsonObjectSchema } from '../../shared/schemas/json';

export const RUN_STATES = [
  'planned',
  'pending',
  'submission_unknown',
  'submitted',
  'queued',
  'held',
  'running',
  'stopping',
  'completed',
  'failed',
  'cancelled',
  'terminal_unknown',
  'observation_unavailable',
  'supervision_lost',
] as const;

const uuid = z.string().regex(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);

export function isUuid(value: unknown): value is string {
  return uuid.safeParse(value).success;
}

const integer = z.number().refine(Number.isSafeInteger);
const positive = integer.refine((value) => value > 0);
const printable = z.string().regex(/^[\x21-\x7e]{1,256}$/);

export const planPreviewSchema = jsonObjectSchema.and(z.object({
  spec: jsonObjectSchema.optional(),
}));

export const runViewSchema = z.object({
  plan_id: uuid,
  run_id: uuid,
  request_id: uuid,
  backend: z.string().min(1),
  operation: z.enum(['train', 'test', 'plot']),
  attempt: positive.nullable(),
  state: z.enum(RUN_STATES),
  uncertainty: z.boolean(),
  submission_requested: z.boolean(),
  cancellation_requested: z.boolean(),
  evidence: jsonObjectSchema.nullable(),
  // Older daemons omit these; a run is still valid without a human identity.
  created_at: z.iso.datetime({ offset: true }).nullable().optional(),
  summary: z
    .object({
      name: z.string().nullable(),
      model: z.string().nullable(),
      data_path: z.string().nullable(),
      epochs: positive.nullable(),
      cv: positive.nullable(),
      source_run_id: uuid.nullable(),
    })
    .nullable()
    .optional(),
});

export const healthSchema = z.object({
  daemon_id: printable,
  build_id: printable,
  protocol_min: integer,
  protocol_max: integer,
});

export const sessionGrantSchema = z.object({
  access_token: z.string(),
  token_type: z.literal('Bearer'),
  expires_at: positive,
  daemon_id: printable,
});

export const planSchema = z.object({
  plan_id: uuid,
  run_id: uuid,
  request_id: uuid,
  preview: planPreviewSchema,
});

export const runListSchema = z.object({ runs: z.array(jsonSchema) });
