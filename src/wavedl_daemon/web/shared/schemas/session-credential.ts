import { z } from 'zod';

export const credentialSchema = z.strictObject({
  version: z.literal(1),
  daemonId: z.string().regex(/^[\x21-\x7e]{1,256}$/),
  origin: z.string().max(512),
  token: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
  expiresAt: z.number().safe().int().positive(),
});

export type SessionCredential = z.infer<typeof credentialSchema>;
