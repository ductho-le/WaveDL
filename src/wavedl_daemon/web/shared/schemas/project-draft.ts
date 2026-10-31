import { z } from 'zod';
import { trainingConfigBaseSchema, trainingRequestSchema } from './training-request';
import { storageMappingSchema } from './storage-mapping';
import { runtimeSetups } from './host-profile';

export const projectBlocks = ['dataset', 'model', 'training', 'runtime', 'plan'] as const;
export type ProjectBlock = (typeof projectBlocks)[number];

/** Field-wise valid training values; selections are stored separately and re-resolved. */
export const draftConfigSchema = trainingConfigBaseSchema
  .omit({ data_path: true, model: true })
  .partial();
export type DraftConfig = z.infer<typeof draftConfigSchema>;
export const draftConfigFields = draftConfigSchema.keyof().options;

const runtimeFields = trainingRequestSchema.shape.runtime.shape;
const topologyFields = trainingRequestSchema.shape.topology.shape;

/**
 * A browser-local editing aid, never a submission. Environment values are excluded
 * because they may carry credentials; everything else is re-validated on restore.
 */
export const projectDraftSchema = z.strictObject({
  version: z.literal(1),
  daemonId: z.string().min(1).max(256),
  savedAt: z.number().int().nonnegative(),
  block: z.enum(projectBlocks),
  dataset: z
    .strictObject({
      root: z.string().regex(/^[A-Za-z0-9_-]{1,64}$/),
      relativePath: z.string().max(4096),
    })
    .nullable(),
  model: z
    .strictObject({
      id: z.string().regex(/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/),
      custom: z.boolean(),
    })
    .nullable(),
  backend: trainingRequestSchema.shape.backend,
  // Absent in drafts saved before automatic runtime setup existed.
  setup: z.enum(runtimeSetups).optional(),
  config: draftConfigSchema,
  runtime: z.strictObject({
    python: z.string().max(8192),
    mode: runtimeFields.mode,
    device: runtimeFields.device,
  }),
  topology: z
    .strictObject({
      num_machines: topologyFields.num_machines,
      processes_per_machine: topologyFields.processes_per_machine,
      machine_rank: topologyFields.machine_rank,
    })
    .partial(),
  storage_map: z.array(storageMappingSchema).max(32),
});
export type ProjectDraft = z.infer<typeof projectDraftSchema>;
