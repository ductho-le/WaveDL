import { validationMessage } from '../../app/i18n/client-messages';
import { z } from 'zod';
import options from '../training-options.json';
import { storageMapSchema } from './storage-mapping';

const finite = z.number().finite().min(-1e12).max(1e12);
const integer = finite.int();
const text = z
  .string()
  .max(8192)
  .refine((value) => !value.includes('\0'), { params: { clientMessageKey: 'nulNotAllowed' } });
const nullableText = text.nullable();
const csvNumbers = text.refine(
  (value) => value.split(',').every((part) => part.trim() !== '' && Number.isFinite(Number(part))),
  { params: { clientMessageKey: 'enterCommaSeparatedFiniteNumbers' } },
);
/** Field types for editable drafts and preferences, before submission compatibility. */
export const trainingConfigBaseSchema = z.strictObject({
  model: text.min(1),
  import_modules: z.array(text.min(1)).max(128),
  pretrained: z.boolean(),
  batch_size: integer.min(1),
  lr: finite.positive(),
  epochs: integer.min(1),
  patience: integer.min(0),
  weight_decay: finite.min(0),
  grad_clip: finite,
  grad_accum_steps: integer.min(1),
  loss: z.enum(['mse', 'mae', 'huber', 'smooth_l1', 'log_cosh', 'weighted_mse']),
  huber_delta: finite.positive(),
  loss_weights: csvNumbers.nullable(),
  optimizer: z.enum(['adamw', 'adam', 'sgd', 'nadam', 'radam', 'rmsprop']),
  momentum: finite.min(0),
  nesterov: z.boolean(),
  betas: csvNumbers.refine(
    (value) =>
      value.split(',').length === 2 &&
      value.split(',').every((part) => Number(part) >= 0 && Number(part) < 1),
    { params: { clientMessageKey: 'enterTwoBetaValuesIn01' } },
  ),
  scheduler: z.enum([
    'plateau',
    'cosine',
    'cosine_restarts',
    'onecycle',
    'step',
    'multistep',
    'exponential',
    'linear_warmup',
  ]),
  scheduler_patience: integer.min(0),
  min_lr: finite.min(0),
  scheduler_factor: finite,
  warmup_epochs: integer.min(0),
  step_size: integer.min(1),
  milestones: text
    .refine((value) => value.split(',').every((part) => /^\d+$/.test(part.trim())), {
      params: { clientMessageKey: 'enterCommaSeparatedEpochs' },
    })
    .nullable(),
  data_path: text.min(1),
  workers: integer.min(-1),
  seed: integer,
  deterministic: z.boolean(),
  cache_validate: z.enum(['sha256', 'fast', 'size']),
  single_channel: z.boolean(),
  cv: integer.min(0),
  cv_stratify: z.boolean(),
  cv_bins: integer.min(1),
  resume: nullableText,
  save_every: integer.min(0),
  fresh: z.boolean(),
  compile: z.boolean(),
  precision: z.enum(['bf16', 'fp16', 'no']),
  constraint: z.array(text.min(1)).max(128),
  constraint_file: nullableText,
  constraint_weight: z.array(finite).min(1).max(128),
  constraint_reduction: z.enum(['mse', 'mae']),
  wandb: z.boolean(),
  wandb_watch: z.boolean(),
  project_name: text,
  run_name: nullableText,
});
const configDefaults = trainingConfigBaseSchema.parse(
  Object.fromEntries(options.map((option) => [option.name, option.default])),
);
// Keep this list aligned with wavedl.core.training_cli.CV_UNSUPPORTED_OPTIONS.
const cvUnsupportedOptions: (keyof z.infer<typeof trainingConfigBaseSchema>)[] = [
  'resume',
  'save_every',
  'fresh',
  'compile',
  'wandb',
  'wandb_watch',
  'project_name',
  'run_name',
];
export const trainingConfigSchema = trainingConfigBaseSchema.superRefine((config, context) => {
  if (config.cv <= 0) return;
  if (config.cv < 2) {
    context.addIssue({
      code: 'custom',
      path: ['cv'],
      params: { clientMessageKey: 'cvFoldCount' },
    });
    return;
  }
  if (config.nesterov) {
    if (config.optimizer !== 'sgd')
      context.addIssue({
        code: 'custom',
        path: ['optimizer'],
        params: { clientMessageKey: 'trainingCvNesterovRequiresSgd' },
      });
    if (config.momentum <= 0)
      context.addIssue({
        code: 'custom',
        path: ['momentum'],
        params: { clientMessageKey: 'cvNesterovMomentum' },
      });
  }
  if (
    config.scheduler === 'plateau' &&
    (config.scheduler_factor < 0 || config.scheduler_factor >= 1)
  )
    context.addIssue({
      code: 'custom',
      path: ['scheduler_factor'],
      params: { clientMessageKey: 'cvPlateauFactor' },
    });
  for (const name of cvUnsupportedOptions) {
    if (config[name] !== configDefaults[name])
      context.addIssue({
        code: 'custom',
        path: [name],
        params: { clientMessageKey: 'trainingUnsupportedCvOption', option: name },
      });
  }
  // cache_validate is retained but inapplicable: CV reads the source directly.
  // The default save_every placeholder is accepted; CV publishes completed folds.
});
export type TrainingConfig = z.infer<typeof trainingConfigSchema>;
export const trainingDefaults: TrainingConfig = configDefaults;
const absolutePath = text
  .min(1)
  .refine(
    (value) => value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('\\\\'),
    { params: { clientMessageKey: 'chooseAnAbsoluteDaemonHostPath' } },
  );
export const trainingRequestSchema = z
  .strictObject({
    backend: z.enum(['local', 'slurm']),
    operation: z.literal('train'),
    storage_map: storageMapSchema.optional(),
    config: trainingConfigSchema,
    runtime: z.strictObject({
      python: absolutePath,
      mode: z.enum(['existing', 'selected', 'bundled', 'container']),
      device: z.enum(['cpu', 'mps', 'cuda', 'auto']),
      environment: z.record(
        z
          .string()
          .regex(/^[A-Za-z_][A-Za-z0-9_]*$/)
          .refine((name) => !reservedRuntimeVariable(name), {
            params: { clientMessageKey: 'reservedIdentityVariable' },
          }),
        text,
      ),
    }),
    topology: z
      .strictObject({
        num_machines: integer.min(1),
        processes_per_machine: integer.min(1),
        machine_rank: integer.min(0),
      })
      .refine(
        (value) =>
          value.machine_rank < value.num_machines &&
          value.num_machines * value.processes_per_machine <= 1024,
        { params: { clientMessageKey: 'invalidTopologyRank' } },
      ),
  })
  .superRefine((request, context) => {
    if (
      request.backend === 'slurm' &&
      Object.keys(request.runtime.environment).some(
        (name) => name.toUpperCase() === 'CUDA_VISIBLE_DEVICES',
      )
    )
      context.addIssue({
        code: 'custom',
        path: ['runtime', 'environment'],
        params: {
          clientMessageKey: 'slurmGpuPolicy',
        },
      });
  });
export type TrainingRequest = z.infer<typeof trainingRequestSchema>;
export function reservedRuntimeVariable(name: string): boolean {
  const upper = name.toUpperCase();
  return (
    ['WAVEDL_', 'SLURM_', 'SRUN_', 'SBATCH_', 'SCANCEL_', 'SQUEUE_', 'SACCT_', 'SCONTROL_'].some(
      (prefix) => upper.startsWith(prefix),
    ) ||
    [
      'RANK',
      'LOCAL_RANK',
      'WORLD_SIZE',
      'LOCAL_WORLD_SIZE',
      'GROUP_RANK',
      'ROLE_RANK',
      'ROLE_WORLD_SIZE',
      'MASTER_ADDR',
      'MASTER_PORT',
    ].includes(upper)
  );
}
export const customModelSchema = z.strictObject({
  modelId: z.string().regex(/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/, {
    error: () => validationMessage('invalidModelRegistryId'),
  }),
});
export const trainingFieldMetadata = options.filter(
  (option) => option.name !== 'data_path' && option.name !== 'model',
);
export const daemonOwnedTrainingFields = {
  config: 'Daemon plans use this typed configuration, not a host YAML filename.',
  output_dir: 'The daemon owns isolated run output directories.',
  list_models: 'Model discovery is the authenticated catalogue.',
  mixed_precision: 'CLI alias of precision; use the single precision field.',
};
