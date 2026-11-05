<script setup lang="ts">
import type { DatasetInspection } from '../../lib/dataset-inspection';

const props = defineProps<{ inspection: DatasetInspection }>();
const { t } = useI18n({ useScope: 'global' });
const details = computed(() => [
  { label: t('datasets.format'), value: props.inspection.format.toUpperCase() },
  { label: t('datasets.inputKey'), value: props.inspection.inputKey },
  { label: t('datasets.outputKey'), value: props.inspection.outputKey },
  { label: t('datasets.inputDtype'), value: props.inspection.inputDtype },
  { label: t('datasets.outputDtype'), value: props.inspection.outputDtype },
]);
</script>

<template>
  <dl class="grid grid-cols-1 gap-x-6 gap-y-3 text-xs min-[30rem]:grid-cols-2" :aria-label="t('datasets.technicalDetails')">
    <div v-for="detail in details" :key="detail.label" class="min-w-0">
      <dt class="text-muted-foreground">{{ detail.label }}</dt>
      <dd class="mt-1 break-all font-mono">{{ detail.value }}</dd>
    </div>
  </dl>
</template>
