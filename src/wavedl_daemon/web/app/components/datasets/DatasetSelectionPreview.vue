<script setup lang="ts">
import { FileStack, Folder } from '@lucide/vue';
import { cn } from '@/lib/utils';
import type { DataSelection } from '../../lib/catalog-api';
import type { DatasetInspection } from '../../lib/dataset-inspection';
import { datasetFormat, datasetInputShape, datasetOutputValues } from '../../lib/dataset-suggestions';
import { pathName } from '../../lib/run-overview';

const props = defineProps<{
  selection: DataSelection | null;
  inspection: DatasetInspection | null;
  busy: boolean;
  failed: boolean;
}>();
defineEmits<{ confirm: [] }>();
const { t, locale } = useI18n({ useScope: 'global' });
const outputValues = computed(() => props.inspection ? datasetOutputValues(props.inspection.outputShape) : null);
const samples = computed(() => props.inspection?.numSamples.toLocaleString(locale.value));
const styles = {
  glyph: cn('grid size-11 place-items-center rounded-xl bg-[color-mix(in_srgb,var(--primary)_10%,transparent)] text-primary [&_svg]:size-5'),
  pill: cn('rounded-md bg-[color-mix(in_srgb,var(--foreground)_5%,transparent)] px-2.5 py-1 text-xs'),
};
</script>

<template>
  <section class="flex h-full min-w-0 flex-col gap-5 p-5" :aria-label="t('datasets.filePreview')" :aria-busy="busy">
    <template v-if="selection">
      <span :class="styles.glyph"><Folder v-if="selection.kind === 'directory'" aria-hidden="true" /><FileStack v-else aria-hidden="true" /></span>
      <div class="min-w-0">
        <div class="flex items-start justify-between gap-2">
          <h3 class="break-all text-base font-semibold">{{ pathName(selection.path) || selection.root }}</h3>
          <Badge v-if="selection.kind === 'file'" variant="outline">{{ datasetFormat(selection.path) }}</Badge>
        </div>
        <p class="mt-1 break-all font-mono text-[0.65rem] text-muted-foreground">{{ selection.path }}</p>
      </div>
      <div v-if="busy" class="space-y-3" role="status" :aria-label="t('datasets.headerLoading')">
        <Skeleton class="h-10 w-32" /><Skeleton class="h-5 w-20" /><Skeleton class="h-7 w-full" />
      </div>
      <div v-else-if="inspection" class="space-y-4" aria-live="polite">
        <p><strong class="block text-4xl font-semibold tracking-tight tabular-nums">{{ samples }}</strong><span class="text-sm text-muted-foreground">{{ t('datasets.samplesLabel') }}</span></p>
        <div class="flex flex-wrap gap-2">
          <span :class="styles.pill">{{ t('datasets.inputSummary', { shape: datasetInputShape(inspection.inputShape) }) }}</span>
          <span :class="styles.pill">{{ outputValues === null ? t('datasets.outputUnknown') : t('datasets.predictSummary', { count: outputValues.toLocaleString(locale) }, outputValues) }}</span>
        </div>
      </div>
      <p v-else-if="failed" role="status" class="text-sm text-muted-foreground">{{ t('datasets.inspectionShort') }}</p>
      <p v-else-if="selection.kind === 'directory'" class="text-sm text-muted-foreground">{{ t('datasets.folderSelected') }}</p>
      <Button type="button" class="mt-auto w-full" :disabled="busy" @click="$emit('confirm')">{{ selection.kind === 'directory' ? t('datasets.useFolder') : t('datasets.useDataset') }}</Button>
    </template>
    <div v-else-if="busy" class="space-y-4" role="status" :aria-label="t('datasets.headerLoading')"><Skeleton class="size-11 rounded-xl" /><Skeleton class="h-5 w-40" /><Skeleton class="h-10 w-32" /><Skeleton class="h-8 w-full" /></div>
    <p v-else class="m-auto max-w-[22ch] text-center text-sm text-muted-foreground">{{ t('datasets.previewPrompt') }}</p>
  </section>
</template>
