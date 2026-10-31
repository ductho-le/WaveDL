<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import { ChevronRight, FileStack } from '@lucide/vue';
import type { ConnectionSession } from '../../lib/connection-session';
import type { DataSelection } from '../../lib/catalog-api';
import type { DatasetInspection } from '../../lib/dataset-inspection';

const props = defineProps<{
  session: ConnectionSession;
  selection: DataSelection | null;
  inspection: DatasetInspection | null;
}>();

const emit = defineEmits<{
  select: [selection: DataSelection];
  inspected: [inspection: DatasetInspection | null, selection: DataSelection | null];
}>();

const { t } = useI18n({ useScope: 'global' });

const name = computed(() => props.selection?.path.split(/[\\/]/).filter(Boolean).at(-1) ?? '');

const stats = computed(() => {
  const info = props.inspection;

  if (!info) return [];

  return [
    { label: t('builder.samples'), value: info.numSamples.toLocaleString() },
    { label: t('builder.inputShape'), value: info.inputShape.join(' × ') || '—' },
    { label: t('builder.targetShape'), value: info.outputShape.join(' × ') || '—' },
    { label: t('builder.format'), value: info.format.toUpperCase() },
  ];
});

const projectStyles = {
  datasetLayout: cn(
    'grid grid-cols-[minmax(0,_1.5fr)_minmax(16rem,_1fr)] [align-items:start] gap-4',
    '[@media(max-width:_56rem)]:grid-cols-1',
  ),
  datasetFile: cn(
    'grid flex-none place-items-center rounded-[0.65rem] size-[2.6rem] text-white',
    '[background:linear-gradient(160deg,_#35c46a,_#1f9d55)] shadow-[inset_0_1px_0_rgb(255_255_255_/_0.25)] [&_svg]:size-5',
  ),
  datasetEmptyGlyph: cn(
    'grid flex-none place-items-center rounded-[0.65rem] size-11 mb-[0.4rem] text-muted-foreground',
    '[background:color-mix(in_srgb,_var(--foreground)_5%,_transparent)] [&_svg]:size-[1.3rem]',
  ),
  datasetCard: cn('p-[1.1rem]'),
  datasetStat: cn(
    'py-[0.6rem] px-3 rounded-lg [background:color-mix(in_srgb,_var(--foreground)_4.5%,_transparent)]',
    '[&_dt]:text-muted-foreground [&_dt]:text-[0.7rem] [&_dd]:m-[0.15rem_0_0] [&_dd]:text-[0.95rem] [&_dd]:font-semibold',
    '[&_dd]:tabular-nums [&_dd]:[overflow-wrap:anywhere]',
  ),
  datasetEmpty: cn(
    'flex min-h-52 flex-col items-center justify-center gap-[0.35rem] p-6 rounded-[1rem]',
    '[border:1.5px_dashed_color-mix(in_srgb,_var(--foreground)_16%,_transparent)] text-center',
  ),
  datasetDetailsTrigger: cn(
    'flex w-full items-center gap-[0.6rem] py-3 px-4 text-[0.85rem] font-medium',
    '[&:hover]:[background:color-mix(in_srgb,_var(--foreground)_3%,_transparent)]',
    "[&[data-state='open']_.dataset-details-chevron]:[transform:rotate(90deg)]",
  ),
  datasetDetailsChevron: cn(
    'size-4 text-muted-foreground [transition:transform_260ms_cubic-bezier(0.22,_1,_0.36,_1)]',
  ),
};
</script>
<template>
  <InspectorPanel :title="t('builder.datasetTitle')" :description="t('builder.datasetDescription')">
    <div :class="projectStyles.datasetLayout">
      <DatasetPathPicker
        :session="session"
        :label="t('project.trainingDataset')"
        :selected="selection"
        @select="emit('select', $event)"
      />

      <aside class="flex min-w-0 flex-col gap-3" :aria-label="t('project.dataset')">
        <motion.div
          v-if="selection"
          :key="selection.path"
          :class="cn('surface-group', projectStyles.datasetCard)"
          :initial="{ opacity: 0, y: 10, scale: 0.98 }"
          :animate="{ opacity: 1, y: 0, scale: 1 }"
          :transition="{ type: 'spring', stiffness: 380, damping: 32 }"
        >
          <div class="flex items-start gap-3">
            <span :class="projectStyles.datasetFile"><FileStack aria-hidden="true" /></span>
            <div class="min-w-0">
              <p class="truncate text-[1.05rem] font-semibold" data-testid="selected-dataset">
                {{ name }}
              </p>
              <p class="mt-0.5 font-mono text-[0.7rem] break-all text-muted-foreground">
                {{ selection.root }} / {{ selection.relativePath }}
              </p>
            </div>
          </div>
          <dl v-if="stats.length" class="mt-4 grid grid-cols-2 gap-2">
            <div v-for="item in stats" :key="item.label" :class="projectStyles.datasetStat">
              <dt>{{ item.label }}</dt>
              <dd>{{ item.value }}</dd>
            </div>
          </dl>
        </motion.div>
        <div v-else :class="projectStyles.datasetEmpty" data-testid="selected-dataset">
          <span :class="projectStyles.datasetEmptyGlyph"><FileStack aria-hidden="true" /></span>
          <p class="text-[0.95rem] font-medium">{{ t('builder.noDataset') }}</p>
          <p class="max-w-[28ch] text-xs text-muted-foreground">{{ t('builder.noDatasetHelp') }}</p>
        </div>

        <Collapsible class="surface-group overflow-hidden" :unmount-on-hide="false">
          <CollapsibleTrigger :class="projectStyles.datasetDetailsTrigger">
            <ChevronRight
              :class="cn('dataset-details-chevron', projectStyles.datasetDetailsChevron)"
              aria-hidden="true"
            />
            {{ t('builder.headerDetails') }}
          </CollapsibleTrigger>
          <CollapsibleContent class="collapsible-motion">
            <div class="border-t border-[color-mix(in_srgb,var(--foreground)_10%,transparent)] p-4">
              <DatasetInfoPanel
                :session="session"
                :selection="selection"
                @inspected="(inspection, current) => emit('inspected', inspection, current)"
              />
            </div>
          </CollapsibleContent>
        </Collapsible>
      </aside>
    </div>
  </InspectorPanel>
</template>
