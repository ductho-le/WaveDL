import { z } from 'zod';
import { jsonObjectSchema, jsonSchema } from './json';

const integer = z.number().refine(Number.isSafeInteger).nonnegative();

const metricValueSchema = z.number().finite().nullable();

const numericFields = {
  train_loss: metricValueSchema.optional(),
  val_loss: metricValueSchema.optional(),
  best_val_loss: metricValueSchema.optional(),
  lr: metricValueSchema.optional(),
  r2: metricValueSchema.optional(),
  pearson: metricValueSchema.optional(),
  patience: metricValueSchema.optional(),
  patience_ctr: metricValueSchema.optional(),
  epoch_time: metricValueSchema.optional(),
  elapsed_time: metricValueSchema.optional(),
  grad_norm: metricValueSchema.optional(),
  mae_avg: metricValueSchema.optional(),
  val_r2: metricValueSchema.optional(),
  val_pearson: metricValueSchema.optional(),
  val_mae_avg: metricValueSchema.optional(),
  total_time: metricValueSchema.optional(),
};

const integerFields = {
  epoch: integer.optional(),
  total_epochs: integer.optional(),
  patience_counter: integer.optional(),
  max_patience: integer.optional(),
};

const parameterMetrics = jsonObjectSchema.transform((value) => Object.entries(value))
  .pipe(z.array(z.tuple([z.string(), metricValueSchema])))
  .transform((entries) => Object.fromEntries(entries));

const metricParameterListSchema = z.array(z.tuple([z.string(), metricValueSchema]));

const metricDetailsFieldsSchema = z.object({
  ...numericFields,
  ...integerFields,
  mae_per_param: z.array(metricValueSchema).optional(),
  per_param_mae: parameterMetrics.optional(),
}).catchall(jsonSchema);

export const metricDetailsSchema = jsonObjectSchema.transform((value, ctx) => {
  const fields = metricDetailsFieldsSchema.safeParse(value);
  const physical = metricParameterListSchema.safeParse(
    Object.entries(value).filter(([key]) => key.startsWith('MAE_Phys_P')),
  );
  const keys = Object.keys(value);
  const failures: { field: string; message: string }[] = [];

  if (!fields.success) {
    for (const issue of fields.error.issues) {
      const field = String(issue.path[0]);
      const message = Object.hasOwn(integerFields, field)
        ? 'invalidIntegerMetric'
        : field === 'mae_per_param' || field === 'per_param_mae'
          ? 'invalidPerParameterMetrics'
          : 'invalidNumericMetric';
      failures.push({ field, message });
    }
  }

  if (!physical.success) {
    const entries = Object.entries(value).filter(([key]) => key.startsWith('MAE_Phys_P'));

    for (const issue of physical.error.issues) {
      const entry = entries.find((_, index) => index === issue.path[0]);

      if (entry) failures.push({ field: entry[0], message: 'invalidNumericMetric' });
    }
  }

  failures.sort((left, right) => keys.indexOf(left.field) - keys.indexOf(right.field));
  const first = failures[0];

  if (first) {
    ctx.addIssue({ code: 'custom', message: first.message, path: [first.field] });

    return z.NEVER;
  }

  if (!fields.success) return z.NEVER;

  if (Object.hasOwn(value, '__proto__')) {
    Object.defineProperty(fields.data, '__proto__', {
      value: value['__proto__'], enumerable: true, writable: true, configurable: true,
    });
  }

  return fields.data;
});

const metricHeader = z.object({
  name: z.string(),
  value: metricValueSchema,
  step: integer,
  unit: z.string().optional(),
  metrics: z.unknown().optional(),
});

export const metricPayloadSchema = metricHeader.pipe(z.object({
  ...metricHeader.shape,
  metrics: z.preprocess((value) => value ?? {}, metricDetailsSchema),
})).transform((payload) => {
  const numeric = new Map<string, z.output<typeof metricValueSchema>>();

  for (const [name, value] of Object.entries(payload.metrics)) {
    const parsed = metricValueSchema.safeParse(value);

    if (parsed.success) numeric.set(name, parsed.data);
  }

  return { ...payload, numeric };
});

export const metricEventScopeSchema = z.object({
  run_id: z.string(),
  attempt: integer.min(1),
  seq: integer,
});

export type MetricDetails = z.output<typeof metricDetailsSchema>;

export type RuntimeMetricField = keyof typeof numericFields;

export type NumericMetrics = z.output<typeof metricPayloadSchema>['numeric'];
