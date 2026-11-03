import { z } from 'zod';

export const diagnosticPayloadSchema = z.object({
  message: z.string().optional().catch(undefined),
  text: z.string().optional().catch(undefined),
});
