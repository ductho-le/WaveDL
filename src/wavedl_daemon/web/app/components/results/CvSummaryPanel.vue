<script setup lang="ts">
import type { AcceptableValue } from 'reka-ui';
import { segmentedSelectionSchema } from '../../../shared/schemas/workstation-controls';
import type { Artifact } from '../../lib/artifact-client';
import type { ConnectionSession } from '../../lib/connection-session';
import { loadCvSummary, type CvMetric, type CvSummary } from '../../lib/cv-summary';

const props = defineProps<{ session: ConnectionSession; artifacts: readonly Artifact[] }>();
const { t, locale } = useI18n({ useScope: 'global' });
const id = useId();

const publications = computed(() =>
  props.artifacts.filter((artifact) => artifact.path === 'cv_summary.json'),
);

const selectionSchema = computed(() =>
  segmentedSelectionSchema(publications.value.map((artifact) => artifact.generation)),
);

const selected = shallowRef<Artifact | null>(null);
const summary = shallowRef<CvSummary | null>(null);
const busy = shallowRef(false);
const failed = shallowRef(false);
let scope = new AbortController();
let epoch = 0;

function reset() {
  scope.abort();
  scope = new AbortController();
  epoch++;
  selected.value = null;
  summary.value = null;
  busy.value = false;
  failed.value = false;
}

watch(() => props.artifacts, reset);

watch(
  () => toRaw(props.session),
  (session, _prior, onCleanup) => {
    reset();
    let revision = session.snapshot().revision;

    const unsubscribe = session.subscribe((state) => {
      if (state.status !== 'connected' || revision !== state.revision) reset();
      revision = state.revision;
    });

    onCleanup(unsubscribe);
  },
  { immediate: true },
);

function choose(generation: AcceptableValue) {
  const parsed = selectionSchema.value.safeParse(generation);
  const artifact = parsed.success ? publications.value.find((item) => item.generation === parsed.data) : undefined;
  scope.abort();
  scope = new AbortController();
  epoch++;
  selected.value = artifact ?? null;
  summary.value = null;
  busy.value = false;
  failed.value = false;
}

async function load() {
  if (!selected.value || busy.value) return;
  const current = epoch;
  busy.value = true;
  failed.value = false;

  try {
    const result = await loadCvSummary(props.session, selected.value, scope.signal);

    if (current === epoch) summary.value = result;
  } catch {
    if (current === epoch) failed.value = true;
  } finally {
    if (current === epoch) busy.value = false;
  }
}

function number(value: CvMetric) {
  return value === null
    ? t('cv.unavailable')
    : new Intl.NumberFormat(locale.value === 'vi' ? 'vi-VN' : 'en-US', {
        maximumSignificantDigits: 6,
      }).format(value);
}

const overall = computed(() =>
  summary.value
    ? [
        { label: t('cv.r2'), mean: summary.value.r2_mean, std: summary.value.r2_std },
        { label: t('cv.mae'), mean: summary.value.mae_mean, std: summary.value.mae_std },
        { label: t('cv.loss'), mean: summary.value.val_loss_mean, std: summary.value.val_loss_std },
      ]
    : [],
);

onBeforeUnmount(() => {
  reset();
});
</script>
<template>
  <section v-if="publications.length" class="flex flex-col gap-4" :aria-labelledby="`${id}-title`">
    <h3 :id="`${id}-title`" class="text-lg font-semibold">{{ t('cv.title') }}</h3>
    <InfoChip :label="t('cv.publicationEvidence')" :text="t('cv.help')" uncertain />
    <label :for="`${id}-source`">{{ t('cv.publication') }}</label>
    <Select :model-value="selected?.generation ?? ''" :disabled="busy" @update:model-value="choose">
      <SelectTrigger
        :id="`${id}-source`"
        :aria-label="t('cv.publication')"
        :data-value="selected?.generation ?? ''"
        ><SelectValue :placeholder="t('cv.choose')"
      /></SelectTrigger>
      <SelectContent
        ><SelectGroup>
          <SelectItem
            v-for="artifact in publications"
            :key="artifact.generation"
            :value="artifact.generation"
          >
            {{ t('cv.source', { generation: artifact.generation, attempt: artifact.attempt }) }}
          </SelectItem>
        </SelectGroup></SelectContent
      >
    </Select>
    <Button variant="outline" :disabled="!selected || busy" @click="load">{{
      busy ? t('cv.loading') : t('cv.load')
    }}</Button>
    <p v-if="failed" role="alert">{{ t('cv.failed') }}</p>
    <template v-if="summary">
      <p>
        {{
          t('cv.timestamp', {
            timestamp: new Date(summary.timestamp).toLocaleString(
              locale === 'vi' ? 'vi-VN' : 'en-US',
            ),
          })
        }}
      </p>
      <h4 class="font-semibold">{{ t('cv.overall') }}</h4>
      <section :aria-label="t('cv.metric')" class="grid gap-4 sm:grid-cols-3">
        <div v-for="row in overall" :key="row.label" class="rounded-xl border p-4">
          <h4 class="text-sm text-muted-foreground">{{ row.label }}</h4>
          <p class="mt-2 text-xl font-semibold tabular-nums" :aria-label="t('cv.mean')">
            {{ number(row.mean) }}
          </p>
          <p class="text-xs text-muted-foreground">{{ t('cv.std') }} {{ number(row.std) }}</p>
        </div>
      </section>
      <section :aria-label="t('cv.folds')" class="grid gap-4">
        <article
          v-for="fold in summary.fold_results"
          :key="fold.fold"
          class="rounded-xl border p-4"
        >
          <h4 class="font-semibold">{{ t('cv.fold') }} {{ fold.fold }}</h4>
          <dl class="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-3">
            <div>
              <dt class="text-sm text-muted-foreground">{{ t('cv.r2') }}</dt>
              <dd>{{ number(fold.r2) }}</dd>
            </div>
            <div>
              <dt class="text-sm text-muted-foreground">{{ t('cv.mae') }}</dt>
              <dd>{{ number(fold.mae_physical) }}</dd>
            </div>
            <div>
              <dt class="text-sm text-muted-foreground">{{ t('cv.normalized') }}</dt>
              <dd>{{ number(fold.mae_normalized) }}</dd>
            </div>
            <div>
              <dt class="text-sm text-muted-foreground">{{ t('cv.loss') }}</dt>
              <dd>{{ number(fold.best_val_loss) }}</dd>
            </div>
            <div>
              <dt class="text-sm text-muted-foreground">{{ t('cv.bestEpoch') }}</dt>
              <dd>{{ fold.best_epoch }}</dd>
            </div>
            <div>
              <dt class="text-sm text-muted-foreground">{{ t('cv.epochs') }}</dt>
              <dd>{{ fold.epochs_trained }}</dd>
            </div>
          </dl>
          <ul class="mt-4 flex flex-col gap-2" :aria-label="t('cv.targets')">
            <li
              v-for="target in fold.targets"
              :key="target.index"
              class="flex flex-wrap justify-between gap-3 rounded-md bg-muted p-2 text-sm"
            >
              <span>{{ t('cv.parameter', { index: target.index }) }}</span>
              <span>{{ t('cv.r2') }} {{ number(target.r2) }}</span>
              <span>{{ t('cv.mae') }} {{ number(target.mae) }}</span>
            </li>
          </ul>
        </article>
      </section>
    </template>
  </section>
</template>
