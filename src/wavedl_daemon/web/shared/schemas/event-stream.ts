import { z } from 'zod';
import { jsonObjectSchema } from './json';

const integer = z.number().refine(Number.isSafeInteger);

export const cursorSchema = z.object({ attempt: integer.min(0), seq: integer.min(-1) })
  .refine((value) => value.attempt !== 0 || value.seq === -1);

export const runEventSchema = z.object({
  protocol: z.literal('wavedl-run'),
  version: z.literal(2),
  run_id: z.string(),
  attempt: integer.min(1),
  seq: integer.min(0),
  type: z.enum([
    'attempt',
    'submitting',
    'probed',
    'submitted',
    'observed',
    'observation_lost',
    'cancel_requested',
    'cancel_delivered',
    'exited',
    'metric',
    'warning',
  ]),
  ts: z.string().regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$/).refine((value) => Number.isFinite(Date.parse(value))),
  payload: jsonObjectSchema,
});
