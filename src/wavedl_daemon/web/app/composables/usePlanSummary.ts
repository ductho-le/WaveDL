import { type Json, type PlanView } from '../lib/daemon-api';
import { planManifestSchema, planPreviewContextSchema } from '../../shared/schemas/plan-summary';
import { trainingDefaults } from '../../shared/schemas/training-request';
import { inferenceDefaults } from '../../shared/schemas/result-request';
import { trainingFieldMessageKeys } from '../i18n/dynamic-keys';

function equal(left: Json, right: Json | undefined): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

/** Reads a persisted plan for review: its manifest, context and non-default values. */
export function usePlanSummary(source: MaybeRefOrGetter<PlanView | null>) {
  const { t } = useI18n({ useScope: 'global' });

  function display(value: Json | undefined): string {
    if (value === undefined || value === null) return t('project.none');

    if (Array.isArray(value))
      return value.length ? value.map((item) => display(item)).join(', ') : t('project.none');

    if (value instanceof Object) return JSON.stringify(value);

    return String(value);
  }

  function text(value: string | undefined): string {
    return value ?? t('project.notAvailable');
  }

  const document = computed(() => {
    const raw = toValue(source)?.preview.manifest_json;

    if (raw === undefined) return null;

    try {
      const parsed = planManifestSchema.safeParse(JSON.parse(raw));

      return parsed.success ? parsed.data : null;
    } catch {
      return null;
    }
  });

  const manifest = computed(() => document.value?.manifest ?? null);
  const config = computed(() => document.value?.review.resolved_config ?? {});

  const differences = computed(() => {
    const defaults = Object.entries(
      manifest.value?.operation === 'train' ? trainingDefaults : inferenceDefaults,
    );

    return Object.entries(config.value).flatMap(([name, value]) => {
      const fallback = defaults.find(([key]) => key === name);

      if (!fallback || equal(value, fallback[1])) return [];

      return [
        {
          name,
          label: trainingFieldMessageKeys[name] ? t(trainingFieldMessageKeys[name].label) : name,
          value: display(value),
          defaultValue: display(fallback[1]),
        },
      ];
    });
  });

  const context = computed(() => {
    const plan = toValue(source);
    const preview = planPreviewContextSchema.parse(plan?.preview ?? {});
    const runtime = document.value?.review.runtime;
    const topology = preview.topology;

    return [
      { label: t('project.contextBackend'), value: text(preview.target) },
      { label: t('project.contextOperation'), value: text(document.value?.review.operation) },
      { label: t('project.contextDevice'), value: text(runtime?.device) },
      {
        label: t('project.contextTopology'),
        value: `${display(topology.num_machines)} × ${display(topology.processes_per_machine)}`,
      },
    ];
  });

  const rawPlan = computed(() => JSON.stringify(toValue(source)?.preview ?? null, null, 2));

  return { manifest, differences, context, rawPlan };
}
