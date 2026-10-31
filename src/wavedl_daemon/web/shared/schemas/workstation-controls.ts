import { z } from 'zod';
import { trainingRequestSchema } from './training-request';
import { inferenceConfigSchema } from './result-request';
import { WORKSTATION_SECTIONS } from '../../app/lib/shell-test-ids';

export const runtimeDeviceSchema = trainingRequestSchema.shape.runtime.shape.device;
export const resultExportSchema = inferenceConfigSchema.shape.export;

export function segmentedSelectionSchema(choices: readonly string[]) {
  return z.enum(choices);
}

export const locateRequestSchema = z.object({ root: z.string(), path: z.string() });

export const storedDockSlotsSchema = z.array(
  z.enum(WORKSTATION_SECTIONS).nullable().catch(null),
).length(5);

export type StoredDockSlots = z.output<typeof storedDockSlotsSchema>;
