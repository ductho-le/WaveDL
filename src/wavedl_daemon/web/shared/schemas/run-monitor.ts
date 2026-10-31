import { z } from 'zod';

// Monitor ownership intentionally has no int64 ceiling on the scheduler id;
// the API evidence decoder enforces its stricter wire contract separately.
export const monitorNativeIdentitySchema = z.object({
  run_id: z.string(),
  request_id: z.string(),
  cluster: z.string().regex(/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/),
  owner: z.string().regex(/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/),
  job_id: z.string().regex(/^[1-9][0-9]{0,18}$/),
  restart_count: z.number().refine(Number.isSafeInteger).min(0),
});

export const diagnosticPayloadSchema = z.object({
  message: z.string().optional().catch(undefined),
  text: z.string().optional().catch(undefined),
});
