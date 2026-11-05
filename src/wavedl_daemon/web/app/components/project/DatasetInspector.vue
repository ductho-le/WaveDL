<script setup lang="ts">
import { cn } from '@/lib/utils';
import { AnimatePresence, motion } from 'motion-v';
import { ArrowRight, ChevronRight, FileStack } from '@lucide/vue';
import type { ConnectionSession } from '../../lib/connection-session';
import type { DataSelection } from '../../lib/catalog-api';
import type { DatasetInspection } from '../../lib/dataset-inspection';
import { pathName } from '../../lib/run-overview';
import { datasetInputShape, datasetOutputValues } from '../../lib/dataset-suggestions';

const props = defineProps<{
  session: ConnectionSession;
  selection: DataSelection | null;
  inspection: DatasetInspection | null;
}>();
const emit = defineEmits<{
  select: [selection: DataSelection];
  inspected: [inspection: DatasetInspection | null, selection: DataSelection | null];
}>();
const { t, locale } = useI18n({ useScope: 'global' });
const editing = shallowRef(!props.selection);
const root = useTemplateRef<HTMLElement>('root');
const reduceMotion = useReducedMotion();
const { info: restoredInfo, busy, failed } = useDatasetInspection(
  () => props.session, () => props.inspection ? null : props.selection,
);
const info = computed(() => props.inspection ?? restoredInfo.value);
const output = computed(() => info.value ? datasetOutputValues(info.value.outputShape) : null);
const stats = computed(() => {
  const current = info.value;

  if (!current) return null;

  return {
    samples: current.numSamples.toLocaleString(locale.value),
    input: datasetInputShape(current.inputShape),
    output: output.value?.toLocaleString(locale.value) ?? '—',
  };
});

watch(restoredInfo, (value) => {
  if (value && props.selection) emit('inspected', value, toRaw(props.selection));
});
watch(() => props.selection, (value) => { editing.value = !value; });

function confirm(selection: DataSelection) {
  editing.value = false;
  emit('select', selection);
}

function focusPicker() {
  root.value?.querySelector<HTMLInputElement>('input[type="search"]')?.focus();
}

const styles = {
  stat: cn(
    'min-w-28 flex-1 rounded-xl px-4 py-3 text-center',
    'bg-[color-mix(in_srgb,var(--foreground)_4.5%,transparent)]',
    '[&_strong]:block [&_strong]:text-2xl [&_strong]:font-semibold [&_strong]:tracking-tight',
    '[&_strong]:tabular-nums [&_strong]:[overflow-wrap:anywhere]',
    '[&_span]:mt-0.5 [&_span]:block [&_span]:text-xs [&_span]:text-muted-foreground',
  ),
  file: cn('grid size-11 shrink-0 place-items-center rounded-xl bg-[color-mix(in_srgb,var(--primary)_10%,transparent)] text-primary [&_svg]:size-5'),
  trigger: cn(
    'flex w-full items-center gap-2 rounded-lg py-3 text-sm text-muted-foreground',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    '[&[data-state=open]_svg]:rotate-90',
  ),
};
</script>

<template>
  <InspectorPanel :title="t('builder.datasetTitle')" :description="t('builder.datasetDescription')">
    <div ref="root" class="min-w-0">
      <AnimatePresence mode="wait" :initial="false">
        <motion.div v-if="editing" key="picker" :initial="{ opacity: 0, height: 0 }" :animate="{ opacity: 1, height: 'auto' }" :exit="{ opacity: 0, height: 0 }" :transition="{ duration: reduceMotion ? 0 : 0.22 }" @animation-complete="focusPicker">
          <DatasetPathPicker :session="session" :label="t('project.trainingDataset')" :selected="selection" @select="confirm" @inspected="(inspection, current) => emit('inspected', inspection, current)" />
          <Button v-if="selection" type="button" variant="ghost" size="sm" class="mt-3" @click="editing = false">{{ t('datasets.keepCurrent') }}</Button>
        </motion.div>
        <motion.div v-else-if="selection" key="summary" class="surface-group p-5" :initial="{ opacity: 0, y: 6 }" :animate="{ opacity: 1, y: 0 }" :exit="{ opacity: 0, y: -6 }" :transition="{ duration: reduceMotion ? 0 : 0.18 }">
          <div class="flex items-start gap-3">
            <span :class="styles.file"><FileStack aria-hidden="true" /></span>
            <div class="min-w-0 flex-1">
              <p class="break-all text-base font-semibold" data-testid="selected-dataset">{{ pathName(selection.path) }}</p>
              <p class="mt-1 break-all font-mono text-xs text-muted-foreground">{{ selection.root }} / {{ selection.relativePath }}</p>
            </div>
            <Button type="button" variant="outline" size="sm" @click="editing = true">{{ t('datasets.change') }}</Button>
          </div>
          <div v-if="stats" class="mt-5 flex flex-wrap items-center gap-2">
            <p :class="styles.stat"><strong>{{ stats.samples }}</strong><span>{{ t('datasets.samplesLabel') }}</span></p>
            <p :class="styles.stat"><strong>{{ stats.input }}</strong><span>{{ t('datasets.eachInputLabel') }}</span></p>
            <ArrowRight class="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <p :class="styles.stat"><strong>{{ stats.output }}</strong><span>{{ t('datasets.predictLabel', output ?? 0) }}</span></p>
          </div>
          <Skeleton v-else-if="busy" class="mt-5 h-5 max-w-80" :aria-label="t('datasets.headerLoading')" />
          <p v-else-if="failed" role="status" class="mt-4 text-sm text-muted-foreground">{{ t('datasets.summaryUnavailable') }}</p>
          <Collapsible v-if="info" class="mt-3 border-t">
            <CollapsibleTrigger :class="styles.trigger"><ChevronRight class="size-4 transition-transform" aria-hidden="true" />{{ t('datasets.technicalDetails') }}</CollapsibleTrigger>
            <CollapsibleContent class="collapsible-motion"><DatasetInfoPanel :inspection="info" class="pb-2" /></CollapsibleContent>
          </Collapsible>
        </motion.div>
      </AnimatePresence>
    </div>
  </InspectorPanel>
</template>
