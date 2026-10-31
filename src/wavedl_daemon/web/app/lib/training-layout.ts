import { trainingDefaults, type TrainingConfig } from '../../shared/schemas/training-request';
import type { DraftConfig } from '../../shared/schemas/project-draft';
import { trainingPresets } from './training-presets';

export type TrainingFieldName = keyof DraftConfig;

export type TrainingSectionId =
  | 'optimizer'
  | 'schedule'
  | 'loss'
  | 'data'
  | 'checkpoints'
  | 'validation'
  | 'constraints'
  | 'code'
  | 'logging';

/** Shown on the Training block itself; everything else lives in a section. */
export const essentialTrainingFields = [
  'epochs',
  'patience',
  'batch_size',
  'lr',
  'loss',
  'optimizer',
  'scheduler',
  'precision',
] as const satisfies readonly TrainingFieldName[];

/** `pretrained` is not listed: the Model block owns it alongside the weights variant. */
export const trainingSections: readonly {
  readonly id: TrainingSectionId;
  readonly fields: readonly TrainingFieldName[];
}[] = [
  {
    id: 'optimizer',
    fields: ['weight_decay', 'grad_clip', 'grad_accum_steps', 'momentum', 'nesterov', 'betas'],
  },
  {
    id: 'schedule',
    fields: [
      'scheduler_patience',
      'scheduler_factor',
      'min_lr',
      'step_size',
      'milestones',
      'warmup_epochs',
    ],
  },
  { id: 'loss', fields: ['huber_delta', 'loss_weights'] },
  {
    id: 'data',
    fields: ['workers', 'cache_validate', 'single_channel', 'seed', 'deterministic', 'compile'],
  },
  { id: 'checkpoints', fields: ['save_every', 'resume', 'fresh'] },
  { id: 'validation', fields: ['cv', 'cv_stratify', 'cv_bins'] },
  {
    id: 'constraints',
    fields: ['constraint', 'constraint_file', 'constraint_weight', 'constraint_reduction'],
  },
  { id: 'code', fields: ['import_modules'] },
  { id: 'logging', fields: ['wandb', 'wandb_watch', 'project_name', 'run_name'] },
];

const adamFamily: readonly TrainingConfig['optimizer'][] = ['adamw', 'adam', 'nadam', 'radam'];

const decayingSchedulers: readonly TrainingConfig['scheduler'][] = [
  'plateau',
  'step',
  'multistep',
  'exponential',
];

const flooredSchedulers: readonly TrainingConfig['scheduler'][] = [
  'plateau',
  'cosine',
  'cosine_restarts',
];

const constrained = (config: TrainingConfig) =>
  config.constraint.length > 0 || !!config.constraint_file;

// Mirrors which arguments wavedl.utils.{optimizers,schedulers,losses} actually read
// for the chosen optimizer, scheduler and loss (scheduler_factor doubles as gamma).
const relevance: Partial<Record<TrainingFieldName, (config: TrainingConfig) => boolean>> = {
  momentum: (config) => config.optimizer === 'sgd' || config.optimizer === 'rmsprop',
  nesterov: (config) => config.optimizer === 'sgd',
  betas: (config) => adamFamily.includes(config.optimizer),
  scheduler_patience: (config) => config.scheduler === 'plateau',
  scheduler_factor: (config) => decayingSchedulers.includes(config.scheduler),
  min_lr: (config) => flooredSchedulers.includes(config.scheduler),
  step_size: (config) => config.scheduler === 'step',
  milestones: (config) => config.scheduler === 'multistep',
  warmup_epochs: (config) => config.scheduler === 'linear_warmup',
  huber_delta: (config) => config.loss === 'huber',
  loss_weights: (config) => config.loss === 'weighted_mse',
  cv_stratify: (config) => config.cv > 0,
  cv_bins: (config) => config.cv > 0 && config.cv_stratify,
  constraint_weight: constrained,
  constraint_reduction: constrained,
  wandb_watch: (config) => config.wandb,
  project_name: (config) => config.wandb,
  run_name: (config) => config.wandb,
};

export function changedFromDefault(name: TrainingFieldName, config: TrainingConfig): boolean {
  return JSON.stringify(config[name]) !== JSON.stringify(trainingDefaults[name]);
}

/** A changed value is never hidden, even when the current choice ignores it. */
export function trainingFieldVisible(name: TrainingFieldName, config: TrainingConfig): boolean {
  return (relevance[name]?.(config) ?? true) || changedFromDefault(name, config);
}

/**
 * Recipes are the presets plus the CLI's own defaults, which a fresh draft starts
 * from; each sets the same handful of values and leaves everything else alone.
 */
export const trainingRecipes = [
  {
    id: 'defaults',
    config: Object.fromEntries(
      Object.keys(trainingPresets[0]?.config ?? {}).map((name) => [
        name,
        Object.entries(trainingDefaults).find(([key]) => key === name)?.[1],
      ]),
    ),
  },
  ...trainingPresets.map(({ id, config }) => ({ id, config })),
];

/** The recipe whose every value the draft still matches, or null for a custom mix. */
export function matchingPreset(config: TrainingConfig): string | null {
  const recipe = trainingRecipes.find((candidate) =>
    Object.entries(candidate.config).every(
      ([name, value]) =>
        JSON.stringify(Object.entries(config).find(([key]) => key === name)?.[1]) ===
        JSON.stringify(value),
    ),
  );

  return recipe?.id ?? null;
}

/** Compact, locale-independent numbers: 0.0003 → 3e-4, 1000 → 1000. */
export function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return '—';

  if (value !== 0 && (Math.abs(value) < 0.01 || Math.abs(value) >= 1e6))
    return value
      .toExponential()
      .replace(/\.?0+e/, 'e')
      .replace('e+', 'e');

  return String(Number(value.toPrecision(6)));
}
