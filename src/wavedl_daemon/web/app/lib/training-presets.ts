import {
  trainingConfigBaseSchema,
  type TrainingConfig,
} from '../../shared/schemas/training-request';

/** Presets cannot retarget datasets, models, executable code or checkpoints. */
export const trainingPresetSchema = trainingConfigBaseSchema
  .omit({
    data_path: true,
    model: true,
    import_modules: true,
    constraint_file: true,
    resume: true,
  })
  .partial();

export const trainingPresets = [
  {
    id: 'fast',
    label: 'Fast iteration',
    description: 'Shorter exploratory training, not a duration guarantee.',
    config: trainingPresetSchema.parse({
      epochs: 100,
      lr: 0.003,
      batch_size: 256,
      patience: 10,
      weight_decay: 0.0001,
      scheduler: 'cosine',
      optimizer: 'adamw',
      loss: 'mse',
      precision: 'bf16',
    }),
  },
  {
    id: 'balanced',
    label: 'Balanced',
    description: 'Moderate epoch budget and early stopping.',
    config: trainingPresetSchema.parse({
      epochs: 500,
      lr: 0.001,
      batch_size: 128,
      patience: 20,
      weight_decay: 0.0001,
      scheduler: 'plateau',
      optimizer: 'adamw',
      loss: 'mse',
      precision: 'bf16',
    }),
  },
  {
    id: 'high_accuracy',
    label: 'Longer training',
    description: 'More epochs and patience; improved accuracy is not guaranteed.',
    config: trainingPresetSchema.parse({
      epochs: 2000,
      lr: 0.0005,
      batch_size: 64,
      patience: 50,
      weight_decay: 0.00005,
      scheduler: 'cosine_restarts',
      optimizer: 'adamw',
      loss: 'mse',
      precision: 'bf16',
    }),
  },
];

export function applyTrainingPreset(current: TrainingConfig, preset: Partial<TrainingConfig>): TrainingConfig {
  // Presets edit a draft; the full request schema validates CV compatibility
  // before preview, without silently discarding incompatible user choices.
  return trainingConfigBaseSchema.parse({ ...current, ...trainingPresetSchema.parse(preset) });
}
