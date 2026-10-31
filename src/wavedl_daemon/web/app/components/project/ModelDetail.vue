<script setup lang="ts">
import { cn } from '@/lib/utils';
import { Check, CircleAlert, CircleCheck, Info } from '@lucide/vue';
import type { DatasetInspection } from '../../lib/dataset-inspection';
import { entryModel, type ModelEntry } from '../../lib/model-entries';
import { modelShapeGuidance } from '../../lib/model-shape-guidance';
import { modelPresentation } from '../../i18n/model-descriptions';
import { modelCategoryMessageKeys, shapeGuidanceMessageKeys } from '../../i18n/dynamic-keys';
import {
  mappedMessageKey,
  modelFamilyMessageKeys,
  modelTierMessageKeys,
} from '../catalog/model-facets';

const props = defineProps<{
  entry: ModelEntry;
  weights: boolean;
  selected: boolean;
  inspection: DatasetInspection | null;
  singleChannel: boolean;
  disabled: boolean;
}>();

const emit = defineEmits<{ 'update:weights': [value: boolean]; choose: [] }>();
const { t, locale } = useI18n({ useScope: 'global' });
const weightsId = useId();
const tiers = ['Edge', 'Low', 'Medium', 'High'];

const model = computed(() => entryModel(props.entry, props.weights));

const text = computed(() => modelPresentation(model.value, locale.value === 'vi' ? 'vi' : 'en', t));

const guidance = computed(() =>
  modelShapeGuidance(props.inspection, model.value, props.singleChannel),
);

const dataDims = computed(() => props.inspection?.inputShape.length ?? 0);

const category = computed(() => {
  const key = mappedMessageKey(modelCategoryMessageKeys, model.value.category);

  return key ? t(key) : model.value.category;
});

const family = computed(() => {
  const key = mappedMessageKey(modelFamilyMessageKeys, model.value.architectureFamily);

  return key ? t(key) : model.value.architectureFamily;
});

const tier = computed(() => {
  const key = mappedMessageKey(modelTierMessageKeys, model.value.computeTier);

  return key ? t(key) : model.value.computeTier;
});

const title = computed(() => props.entry.primary.displayName.replace(/ \(Pretrained\)$/, ''));
const tierLevel = computed(() => tiers.indexOf(model.value.computeTier) + 1);

// Bundled translations cover pros only; cons stay in the daemon's English.
const cons = computed(() => (locale.value === 'en' ? model.value.cons.slice(0, 3) : []));

const shapeNote = computed(() =>
  t(shapeGuidanceMessageKeys[guidance.value.status], {
    shape: guidance.value.shape,
    dims: guidance.value.dimensions,
  }),
);

const megabytes = computed(() =>
  model.value.sizeMb >= 1000
    ? `${(model.value.sizeMb / 1000).toFixed(1)} GB`
    : `${Math.round(model.value.sizeMb)} MB`,
);

const projectStyles = {
  modelDetailChoose: cn('min-w-40 rounded-[0.8rem]'),
  modelStat: cn(
    'flex flex-col gap-[0.3rem] py-[0.7rem] px-[0.85rem] rounded-[0.8rem]',
    '[background:color-mix(in_srgb,_var(--foreground)_4.5%,_transparent)] [&_dt]:text-muted-foreground',
    '[&_dt]:text-[0.72rem] [&_dd]:m-0 [&_dd]:text-[1.05rem] [&_dd]:font-semibold [&_dd]:tracking-[-0.01em]',
  ),
  modelTier: cn(
    'block w-[0.9rem] h-[0.35rem] rounded-full [background:color-mix(in_srgb,_var(--foreground)_14%,_transparent)]',
    "[&[data-on='true']]:bg-primary",
  ),
  modelDim: cn(
    'py-[0.05rem] px-[0.4rem] rounded-[0.4rem] text-[0.85rem]',
    "[background:color-mix(in_srgb,_var(--foreground)_7%,_transparent)] [&[data-match='true']]:text-primary-foreground",
    "[&[data-match='true']]:bg-primary",
  ),
  modelFit: cn(
    'flex gap-[0.6rem] py-3 px-[0.9rem] rounded-[0.8rem] text-[0.82rem] leading-[1.45]',
    '[background:color-mix(in_srgb,_var(--foreground)_4%,_transparent)] [&_svg]:flex-none [&_svg]:size-4',
    "[&_svg]:mt-[0.1rem] [&[data-status='described']]:[background:color-mix(in_srgb,_var(--success)_9%,_transparent)]",
    "[&[data-status='described']_svg]:text-success",
    "[&[data-status='mismatch']]:[background:color-mix(in_srgb,_var(--warning)_11%,_transparent)]",
    "[&[data-status='mismatch']_svg]:text-warning",
  ),
  modelListTitle: cn('mb-2 text-[0.7rem] font-[650] tracking-[0.06em] uppercase'),
  modelPoints: cn(
    'flex flex-col gap-[0.45rem] text-[0.85rem] leading-[1.45] [&_li]:relative [&_li]:pl-[0.9rem]',
    "[&_li::before]:[content:''] [&_li::before]:absolute [&_li::before]:left-0 [&_li::before]:top-[0.55em]",
    '[&_li::before]:size-[0.3rem] [&_li::before]:rounded-full [&_li::before]:bg-current [&_li::before]:opacity-35',
  ),
};
</script>
<template>
  <article class="flex min-w-0 flex-col gap-6" :aria-label="title">
    <header class="flex flex-wrap items-start gap-4">
      <div class="min-w-[min(100%,16rem)] flex-1">
        <p class="text-xs font-medium tracking-wide text-muted-foreground uppercase">
          {{ category }} · {{ family }}
        </p>
        <h3 class="mt-1 text-[1.65rem] leading-tight font-semibold tracking-[-0.025em]">
          {{ title }}
        </h3>
        <p class="mt-1 text-sm text-muted-foreground">
          {{ model.authorInstitution }} · {{ model.yearPublished }}
          <span class="font-mono text-xs">· {{ model.id }}</span>
        </p>
      </div>
      <Button
        type="button"
        size="lg"
        :class="projectStyles.modelDetailChoose"
        :variant="selected ? 'secondary' : 'default'"
        :disabled="disabled || selected"
        data-testid="model-select"
        @click="emit('choose')"
      >
        <Check v-if="selected" aria-hidden="true" />
        {{ selected ? t('builder.modelInUse') : t('builder.useThisModel') }}
      </Button>
    </header>

    <p class="max-w-[62ch] text-[0.95rem] leading-relaxed">{{ text.bestFor }}</p>

    <dl class="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
      <div :class="projectStyles.modelStat">
        <dt>{{ t('builder.parameters') }}</dt>
        <dd>{{ t('builder.millions', { value: model.paramsM }) }}</dd>
      </div>
      <div :class="projectStyles.modelStat">
        <dt>{{ t('builder.weightsSize') }}</dt>
        <dd>{{ megabytes }}</dd>
      </div>
      <div :class="projectStyles.modelStat">
        <dt>{{ t('builder.compute') }}</dt>
        <dd class="flex items-center gap-2">
          <span class="flex gap-0.5" aria-hidden="true">
            <i
              v-for="level in 4"
              :key="level"
              :class="projectStyles.modelTier"
              :data-on="level <= tierLevel"
            />
          </span>
          <span class="text-sm">{{ tier }}</span>
        </dd>
      </div>
      <div :class="projectStyles.modelStat">
        <dt>{{ t('builder.inputs') }}</dt>
        <dd class="flex gap-1">
          <span
            v-for="dims in model.supportedDims"
            :key="dims"
            :class="projectStyles.modelDim"
            :data-match="dims === dataDims"
            >{{ dims }}D</span
          >
        </dd>
      </div>
    </dl>

    <div
      v-if="entry.weighted"
      class="surface-group surface-row"
      :data-testid="`model-weights-${entry.key}`"
    >
      <div class="min-w-0 flex-1">
        <label :for="weightsId" class="text-[0.9rem] font-medium">
          {{ t('builder.startFromImagenet') }}
        </label>
        <p class="text-xs text-muted-foreground">
          {{
            t('builder.weightsHelp', {
              dims: entry.weighted.supportedDims.map((value) => `${value}D`).join(' · '),
            })
          }}
        </p>
      </div>
      <Switch
        :id="weightsId"
        :model-value="weights"
        :disabled="disabled"
        @update:model-value="emit('update:weights', $event === true)"
      />
    </div>

    <div
      v-if="inspection"
      :class="projectStyles.modelFit"
      :data-status="guidance.status"
      role="note"
    >
      <CircleCheck v-if="guidance.status === 'described'" aria-hidden="true" />
      <CircleAlert v-else-if="guidance.status === 'mismatch'" aria-hidden="true" />
      <Info v-else aria-hidden="true" />
      <p>{{ shapeNote }}</p>
    </div>

    <div class="grid gap-6 sm:grid-cols-2">
      <section v-if="text.pros.length">
        <h4 :class="cn('text-success', projectStyles.modelListTitle)">{{ t('builder.goodAt') }}</h4>
        <ul :class="projectStyles.modelPoints">
          <li v-for="pro in text.pros" :key="pro">{{ pro }}</li>
        </ul>
      </section>
      <section v-if="cons.length">
        <h4 :class="cn('text-warning', projectStyles.modelListTitle)">
          {{ t('builder.watchOut') }}
        </h4>
        <ul :class="projectStyles.modelPoints">
          <li v-for="con in cons" :key="con">{{ con }}</li>
        </ul>
      </section>
    </div>
    <p v-if="text.untranslated" class="text-xs text-muted-foreground">
      {{ t('catalog.untranslated') }}
    </p>
  </article>
</template>
