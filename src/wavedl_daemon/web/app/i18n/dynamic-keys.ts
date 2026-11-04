import trainingOptions from '../../shared/training-options.json';
import type { RunMonitorSnapshot } from '../lib/run-monitor';
import type { RunState, RunView } from '../lib/daemon-api';

export const workstationNavigationMessageKeys = {
  experiments: 'shell.experiments',
  project: 'shell.newTrainingRun',
  datasets: 'shell.datasets',
  results: 'shell.results',
  connection: 'shell.connection',
  settings: 'shell.settings',
} as const;

export const datasetPreviewTabMessageKeys = {
  input: 'datasets.inputDistribution',
  output: 'datasets.outputDistribution',
  sample: 'datasets.samples',
  correlation: 'datasets.correlation',
  statistics: 'datasets.statistics',
  scatter: 'datasets.scatter',
} as const;

export const projectBlockMessageKeys = {
  dataset: 'builder.blockDataset',
  model: 'builder.blockModel',
  training: 'builder.blockTraining',
  runtime: 'builder.blockRuntime',
  plan: 'builder.blockPlan',
} as const;

export const trainingSectionMessageKeys = {
  optimizer: 'builder.sectionOptimizer',
  schedule: 'builder.sectionSchedule',
  loss: 'builder.sectionLoss',
  data: 'builder.sectionData',
  checkpoints: 'builder.sectionCheckpoints',
  validation: 'builder.sectionValidation',
  constraints: 'builder.sectionConstraints',
  code: 'builder.sectionCode',
  logging: 'builder.sectionLogging',
} as const;

/** Translatable choice names, keyed `field.value`; technical names stay as written. */
export const trainingChoiceMessageKeys = {
  'loss.weighted_mse': 'builder.choiceWeightedMse',
  'scheduler.plateau': 'builder.choicePlateau',
  'scheduler.cosine': 'builder.choiceCosine',
  'scheduler.cosine_restarts': 'builder.choiceCosineRestarts',
  'scheduler.onecycle': 'builder.choiceOneCycle',
  'scheduler.step': 'builder.choiceStep',
  'scheduler.multistep': 'builder.choiceMultiStep',
  'scheduler.exponential': 'builder.choiceExponential',
  'scheduler.linear_warmup': 'builder.choiceLinearWarmup',
  'precision.no': 'builder.choiceFullPrecision',
  'cache_validate.fast': 'builder.choiceFast',
  'cache_validate.size': 'builder.choiceSizeOnly',
};

const technicalChoiceLabels = {
  mse: 'MSE',
  mae: 'MAE',
  huber: 'Huber',
  smooth_l1: 'Smooth L1',
  log_cosh: 'Log-cosh',
  adamw: 'AdamW',
  adam: 'Adam',
  sgd: 'SGD',
  nadam: 'NAdam',
  radam: 'RAdam',
  rmsprop: 'RMSprop',
  bf16: 'bf16',
  fp16: 'fp16',
  sha256: 'SHA-256',
};

export function trainingChoiceLabel(
  field: string,
  choice: string,
  t: (key: string) => string,
): string {
  const key = `${field}.${choice}`;

  const messageKey = Object.entries(trainingChoiceMessageKeys).find(([name]) => name === key)?.[1];

  if (messageKey) return t(messageKey);

  return Object.entries(technicalChoiceLabels).find(([name]) => name === choice)?.[1] ?? choice;
}

export const runtimeFieldMessageKeys = {
  pythonInterpreter: 'project.pythonInterpreter',
  interpreterDescription: 'project.interpreterHelp',
  backend: 'project.configuredBackend',
  mode: 'project.runtimeMode',
  device: 'project.device',
  machines: 'project.machines',
  processesPerMachine: 'project.processesPerMachine',
  machineRank: 'project.machineRank',
} as const;

export const inferenceFieldMessageKeys = {
  batchSize: 'results.batchSize',
  workers: 'results.workers',
  inputChannels: 'results.inputChannels',
  outputSize: 'results.outputSize',
  opset: 'results.opset',
  model: 'results.model',
  inputKey: 'results.inputKey',
  outputKey: 'results.outputKey',
  predictions: 'results.predictions',
  plot: 'results.plot',
  verbose: 'results.verbose',
  normalized: 'results.normalized',
} as const;

export const appearanceControlMessageKeys = {
  appearance: 'shell.appearance',
  system: 'shell.system',
  light: 'shell.light',
  dark: 'shell.dark',
  accent: 'shell.accent',
  neutral: 'shell.neutral',
  blue: 'shell.blue',
  violet: 'shell.violet',
  green: 'shell.green',
  interfaceScale: 'shell.interfaceScale',
} as const;

export const datasetPreviewMessageKeys = {
  correlationMissing: 'datasets.correlationMissing',
  correlationInsufficientRows: 'datasets.correlationInsufficientRows',
} as const;

export const datasetExportMessageKeys = {
  unavailable: 'datasets.pngExportUnavailable',
  preparing: 'datasets.pngExportPreparing',
  preparationFailed: 'datasets.pngExportPreparationFailed',
  requested: 'datasets.pngExportRequested',
  requestFailed: 'datasets.pngExportRequestFailed',
} as const;

export const trainingPresetMessageKeys = {
  defaults: {
    label: 'builder.recipeDefaults',
    description: 'builder.recipeDefaultsHelp',
  },
  fast: {
    label: 'project.fastIteration',
    description: 'project.fastPresetHelp',
  },
  balanced: {
    label: 'project.balanced',
    description: 'project.balancedPresetHelp',
  },
  high_accuracy: {
    label: 'project.longerTraining',
    description: 'project.accuracyPresetHelp',
  },
} as const;

export function trainingPresetMessageKeysFor(presetId: string) {
  return Object.entries(trainingPresetMessageKeys).find(([name]) => name === presetId)?.[1] ?? null;
}

export const trainingFieldMessageKeys = Object.fromEntries(
  trainingOptions.map(
    ({ name }) =>
      [
        name,
        {
          label: `training.${name}.label`,
          description: `training.${name}.description`,
        },
      ] as const,
  ),
);

export const runtimeValues = [
  'local',
  'slurm',
  'existing',
  'selected',
  'bundled',
  'container',
  'cpu',
  'mps',
  'cuda',
  'auto',
] as const;

export type RuntimeValue = (typeof runtimeValues)[number];

export const runtimeValueMessageKeys: Readonly<Record<RuntimeValue, string | null>> = {
  local: 'project.local',
  slurm: null,
  existing: 'project.existing',
  selected: 'project.selected',
  bundled: 'project.bundled',
  container: 'project.container',
  cpu: null,
  mps: null,
  cuda: null,
  auto: 'project.auto',
};

export const runStateMessageKeys: Readonly<Record<RunState, string>> = {
  planned: 'runs.planned',
  pending: 'runs.pending',
  submission_unknown: 'runs.submissionUnknown',
  submitted: 'runs.submitted',
  queued: 'runs.queued',
  held: 'runs.held',
  running: 'runs.running',
  stopping: 'runs.stopping',
  completed: 'runs.completed',
  failed: 'runs.failed',
  cancelled: 'runs.cancelled',
  terminal_unknown: 'runs.terminalUnknown',
  observation_unavailable: 'runs.observationUnavailable',
  supervision_lost: 'runs.supervisionLost',
};

export const runOperationMessageKeys: Readonly<Record<RunView['operation'], string>> = {
  train: 'runs.operationTrain',
  test: 'runs.operationTest',
  plot: 'runs.operationPlot',
};

export const runBackendMessageKeys = {
  local: 'runs.backendLocal',
  slurm: 'runs.backendSlurm',
} as const;

export const runStreamMessageKeys: Readonly<Record<RunMonitorSnapshot['stream'], string>> = {
  connecting: 'runs.connecting',
  live: 'runs.live',
  reconnecting: 'runs.reconnecting',
  stopped: 'runs.stopped',
  error: 'runs.error',
};

export const runCancellationMessageKeys: Readonly<
  Record<RunMonitorSnapshot['cancellation'], string>
> = {
  idle: 'runs.idle',
  sending: 'runs.sending',
  requested: 'runs.requested',
  unknown: 'runs.unknown',
};

export const modelCategoryMessageKeys = {
  'Classic CNNs': 'catalog.classic',
  'Efficient CNNs': 'catalog.efficient',
  'Modern CNNs': 'catalog.modern',
  'Vision Transformers': 'catalog.transformers',
  'Hybrid Architectures': 'catalog.hybrid',
  'State Space Models': 'catalog.stateSpace',
  'Temporal Networks': 'catalog.temporal',
  '3D & Specialized': 'catalog.specialized',
} as const;

export const shapeGuidanceMessageKeys = {
  described: 'catalog.shape_described',
  mismatch: 'catalog.shape_mismatch',
  unknown: 'catalog.shape_unknown',
  ambiguous: 'catalog.shape_ambiguous',
} as const;

export const metricHealthLabelMessageKeys = {
  Overfitting: 'runs.overfitting',
  'Learning rate': 'runs.lr',
  'Gradient health': 'runs.gradientHealth',
  'Patience used': 'runs.patienceUsed',
  Convergence: 'runs.convergence',
} as const;

export const metricHealthStatusMessageKeys = {
  Severe: 'runs.severe',
  Moderate: 'runs.moderate',
  Healthy: 'runs.healthy',
  Underfitting: 'runs.underfitting',
  'Very Low': 'runs.veryLow',
  High: 'runs.high',
  Normal: 'runs.normal',
  Exploding: 'runs.exploding',
  Vanishing: 'runs.vanishing',
  Stable: 'runs.stable',
  Critical: 'runs.critical',
  Elevated: 'runs.elevated',
  OK: 'runs.ok',
  Converging: 'runs.converging',
  Plateau: 'runs.plateau',
  Diverging: 'runs.diverging',
} as const;

export const evaluationMetricMessageKeys = {
  mae_avg: 'runs.evalMae',
  val_mae_avg: 'runs.evalValMae',
  r2: 'runs.evalR2',
  val_r2: 'runs.evalValR2',
  pearson: 'runs.evalPearson',
  val_pearson: 'runs.evalValPearson',
  epoch_time: 'runs.evalEpochTime',
  total_time: 'runs.evalTotalTime',
  mse: 'runs.evalMse',
  rmse: 'runs.evalRmse',
  mae_per_param: 'runs.evalMaePerParam',
  per_param_mae: 'runs.evalPerParamMae',
} as const;

export const plotMetricMessageKeys = {
  train_loss: 'runs.trainLoss',
  val_loss: 'runs.valLoss',
  r2: 'runs.trainR2',
  val_r2: 'runs.valR2',
  lr: 'runs.lr',
} as const;

export const dynamicMessageKeys: readonly string[] = [
  ...Object.values(workstationNavigationMessageKeys),
  ...Object.values(datasetPreviewTabMessageKeys),
  ...Object.values(projectBlockMessageKeys),
  ...Object.values(trainingSectionMessageKeys),
  ...Object.values(trainingChoiceMessageKeys),
  ...Object.values(runtimeFieldMessageKeys),
  ...Object.values(inferenceFieldMessageKeys),
  ...Object.values(appearanceControlMessageKeys),
  ...Object.values(datasetPreviewMessageKeys),
  ...Object.values(datasetExportMessageKeys),
  ...Object.values(trainingPresetMessageKeys).flatMap(({ label, description }) => [
    label,
    description,
  ]),
  ...Object.values(trainingFieldMessageKeys).flatMap(({ label, description }) => [
    label,
    description,
  ]),
  ...Object.values(runtimeValueMessageKeys).filter((key): key is string => key !== null),
  ...Object.values(runStateMessageKeys),
  ...Object.values(runOperationMessageKeys),
  ...Object.values(runBackendMessageKeys),
  ...Object.values(runStreamMessageKeys),
  ...Object.values(runCancellationMessageKeys),
  ...Object.values(modelCategoryMessageKeys),
  ...Object.values(shapeGuidanceMessageKeys),
  ...Object.values(metricHealthLabelMessageKeys),
  ...Object.values(metricHealthStatusMessageKeys),
  ...Object.values(evaluationMetricMessageKeys),
  ...Object.values(plotMetricMessageKeys),
];
