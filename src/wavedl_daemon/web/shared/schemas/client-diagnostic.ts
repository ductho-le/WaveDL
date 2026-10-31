import { z } from 'zod';
import englishClientErrors from '../../app/i18n/locales/en/client-errors.json';

export const clientErrorKeySchema = z.custom<keyof typeof englishClientErrors>(
  (value) => Object.keys(englishClientErrors).some((key) => key === value),
);

export const clientErrorParamsSchema = z.record(
  z.string(), z.union([z.string(), z.number()]).optional().catch(undefined),
).transform((params) => Object.fromEntries(
  Object.entries(params).flatMap(([key, value]) => value === undefined ? [] : [[key, value] as const]),
));

export const clientDiagnosticSchema = z.object({
  clientMessageKey: clientErrorKeySchema,
  clientMessageParams: z.record(z.string(), z.union([z.string(), z.number()])).optional(),
});

export const customIssueParamsSchema = z.object({
  clientMessageKey: clientErrorKeySchema,
}).and(clientErrorParamsSchema);
