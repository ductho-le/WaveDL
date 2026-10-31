import { z } from 'zod';
import { validationMessage } from '../../app/i18n/client-messages';

/** Only bearer syntax is shared; credentials never become persisted drafts. */
export const daemonCredentialSchema = z.object({
  token: z
    .string()
    .min(1, { error: () => validationMessage('credentialRequired') })
    .max(4096, { error: () => validationMessage('theCredentialIsTooLong') })
    .regex(/^[A-Za-z0-9._~+/=-]+$/, {
      error: () => validationMessage('enterAValidBearerCredential'),
    }),
});

export type DaemonCredentialValues = z.infer<typeof daemonCredentialSchema>;
