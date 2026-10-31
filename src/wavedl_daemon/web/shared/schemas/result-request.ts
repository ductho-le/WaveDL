import { z } from 'zod';
import { trainingRequestSchema } from './training-request';
const text = z
  .string()
  .min(1)
  .max(8192)
  .refine((value) => !value.includes('\0'));
export const inferenceConfigSchema = z.strictObject({
  data_path: text,
  model: text.nullable(),
  import_modules: trainingRequestSchema.shape.config.shape.import_modules
    .refine((modules) => modules.every((module) => !module.startsWith('-')))
    .default([]),
  format: z.enum(['auto', 'npz', 'mat', 'hdf5']),
  input_key: text.nullable(),
  output_key: text.nullable(),
  param_names: z.array(text).min(1).max(256).nullable(),
  input_channels: z.number().int().positive().nullable(),
  batch_size: z.number().int().positive(),
  workers: z.number().int().min(0),
  save_predictions: z.boolean(),
  plot: z.boolean(),
  plot_format: z
    .array(z.enum(['png', 'pdf', 'svg', 'eps', 'tiff']))
    .min(1)
    .max(5),
  verbose: z.boolean(),
  export: z.literal('onnx').nullable(),
  export_opset: z.number().int().min(11).max(17),
  no_denorm: z.boolean(),
  out_size: z.number().int().positive().nullable(),
});
export const inferenceDefaults: z.infer<typeof inferenceConfigSchema> = {
  data_path: '',
  model: null,
  import_modules: [],
  format: 'auto',
  input_key: null,
  output_key: null,
  param_names: null,
  input_channels: null,
  batch_size: 128,
  workers: 0,
  save_predictions: false,
  plot: false,
  plot_format: ['png'],
  verbose: false,
  export: null,
  export_opset: 17,
  no_denorm: false,
  out_size: null,
};
export const resultRequestSchema = z.strictObject({
  backend: trainingRequestSchema.shape.backend,
  storage_map: trainingRequestSchema.shape.storage_map,
  runtime: trainingRequestSchema.shape.runtime,
  topology: trainingRequestSchema.shape.topology,
  operation: z.enum(['test', 'plot']),
  checkpoint_source: z.strictObject({
    run_id: z.string().uuid(),
    generation: z.string().regex(/^gen-\d{6,}$/),
  }),
  config: inferenceConfigSchema,
});
export type ResultRequest = z.infer<typeof resultRequestSchema>;
