import { z } from 'zod';

export const artifactPathSchema = z.string().max(4096).refine((value) =>
  !/[\\:%]/.test(value) &&
  !Array.from(value).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) &&
  value.split('/').every((part) => part !== '' && part !== '.' && part !== '..'),
);

export const checkpointNameSchema = z.string().regex(/^(?:best|interrupted|epoch_[0-9]+)_checkpoint$/);

export const checkpointSourceSchema = z.strictObject({
  run_id: z.string().uuid(),
  checkpoint: checkpointNameSchema,
});

export const inventorySchema = z.strictObject({
  files: z.array(z.strictObject({
    path: artifactPathSchema,
    size: z.number().refine(Number.isSafeInteger).refine((value) => value >= 0),
    modified: z.iso.datetime({ offset: true }),
  })).max(16384),
  checkpoints: z.array(checkpointNameSchema).max(4096),
});
