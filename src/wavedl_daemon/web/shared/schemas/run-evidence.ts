import { z } from 'zod';

const integer = z.number().refine(Number.isSafeInteger);

export const runEvidenceSchema = z.object({
  reason: z.string(),
  phase: z.enum(['queued', 'held', 'running']).nullable(),
  native_state: z.string().nullable(),
  outcome: z.enum(['completed', 'failed', 'cancelled', 'requeued', 'unknown', 'lost']).nullable(),
  exit_code: integer.nullable(),
  signal: integer.refine((value) => value > 0).nullable(),
  attempt_exit: z.object({ batch: integer, training: integer, stage_out: integer }).nullable(),
  restart_count: integer.refine((value) => value >= 0).nullable(),
});
