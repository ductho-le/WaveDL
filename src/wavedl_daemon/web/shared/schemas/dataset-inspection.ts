import { z } from 'zod';

const integer = z
  .number()
  .int()
  .nonnegative()
  .max(8 * 1024 ** 3);

const text = z
  .string()
  .max(256)
  .refine((value) => !value.includes('\0') && new TextEncoder().encode(value).length <= 256);

const variable = z.strictObject({
  name: text,
  dtype: text,
  'shape': z.array(integer).max(16),
  sample_count: integer.nullable(),
  declared_bytes: integer,
});

export const datasetInspectionSchema = z.strictObject({
  schema_version: z.literal(1),
  format: z.enum(['npy', 'npz', 'mat', 'h5', 'hdf5']),
  variables: z.array(variable).max(128),
});
