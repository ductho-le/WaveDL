import { z } from 'zod';
import { jsonObjectSchema } from './json';

const cvMetricSchema = z.number().finite().nullable();

const cvNonnegativeSchema = z.number().finite().nonnegative().nullable();

const cvIntegerSchema = (minimum: number, maximum: number) =>
  z.number().refine(Number.isSafeInteger).min(minimum).max(maximum);

const text = z.string().max(256);
const config = z.strictObject({
  model: text,
  folds: cvIntegerSchema(2, 256),
  stratify: z.boolean(),
  stratify_bins: cvIntegerSchema(1, 1_000_000),
  batch_size: cvIntegerSchema(1, 1_000_000),
  lr: z.number().finite().positive(),
  epochs: cvIntegerSchema(1, 1_000_000),
  patience: cvIntegerSchema(0, 1_000_000),
  loss: text,
  optimizer: text,
  scheduler: text,
});

const foldFields = z.object({
  fold: cvIntegerSchema(1, 256),
  best_val_loss: cvNonnegativeSchema,
  best_epoch: cvIntegerSchema(0, 1_000_000),
  r2: cvMetricSchema,
  mae_normalized: cvNonnegativeSchema,
  mae_physical: cvNonnegativeSchema,
  epochs_trained: cvIntegerSchema(1, 1_000_000),
}).catchall(cvMetricSchema);

const targetMetrics = z.array(z.object({
  index: cvIntegerSchema(0, 255),
  r2: cvMetricSchema,
  mae: cvNonnegativeSchema,
})).min(1).max(256);

const foldSchema = jsonObjectSchema.transform((value, ctx) => {
  const base = foldFields.safeParse(value);
  const targetKeys = Object.keys(value).filter((key) => !Object.hasOwn(foldFields.shape, key));
  const count = targetKeys.length / 2;

  if (!base.success || !Number.isInteger(count) || count < 1 || count > 256 ||
    targetKeys.some((key) => !/^(r2|mae)_target_(0|[1-9]\d*)$/.test(key))) {
    ctx.addIssue({ code: 'custom', message: 'Invalid fold metrics' });

    return z.NEVER;
  }

  const targets = targetMetrics.safeParse(Array.from({ length: count }, (_, index) => ({
    index, r2: value[`r2_target_${index}`], mae: value[`mae_target_${index}`],
  })));

  if (!targets.success || base.data.best_epoch > base.data.epochs_trained) {
    ctx.addIssue({ code: 'custom', message: 'Invalid fold targets or epoch' });

    return z.NEVER;
  }

  const { fold, best_val_loss, best_epoch, r2, mae_normalized, mae_physical, epochs_trained } = base.data;

  return { fold, best_val_loss, best_epoch, r2, mae_normalized, mae_physical, epochs_trained, targets: targets.data };
});

export const cvSummarySchema = z.strictObject({
  config,
  timestamp: z.string().max(64).refine((value) => Number.isFinite(Date.parse(value))),
  folds: cvIntegerSchema(2, 256),
  r2_mean: cvMetricSchema,
  r2_std: cvNonnegativeSchema,
  mae_mean: cvNonnegativeSchema,
  mae_std: cvNonnegativeSchema,
  val_loss_mean: cvNonnegativeSchema,
  val_loss_std: cvNonnegativeSchema,
  fold_results: z.array(foldSchema),
}).refine((value) => {
  const targetCount = value.fold_results[0]?.targets.length ?? 0;

  return value.config.folds === value.folds && value.fold_results.length === value.folds &&
    new Set(value.fold_results.map((fold) => fold.fold)).size === value.folds &&
    targetCount * value.folds <= 4096 && value.fold_results.every((fold) =>
      fold.fold <= value.folds && fold.targets.length === targetCount,
    );
}).transform(({ config: _config, ...summary }) => summary);
