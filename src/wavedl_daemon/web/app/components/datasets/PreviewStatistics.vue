<script setup lang="ts">
import type { DatasetPreview } from '../../lib/dataset-preview';

const { t } = useI18n({ useScope: 'global' });

defineProps<{ statistics: DatasetPreview['input_statistics'] }>();
</script>

<template>
  <div class="statistics-plots">
    <PreviewPlot
      :title="t('datasets.meanTitle')"
      :x="statistics.feature_indices"
      :y="statistics.means"
    />
    <PreviewPlot
      :title="t('datasets.stdTitle')"
      :x="statistics.feature_indices"
      :y="statistics.stds"
    />
  </div>
  <section class="flex flex-col gap-3" :aria-label="t('datasets.statsCaption')">
    <h4 class="text-sm font-medium">{{ t('datasets.statsCaption') }}</h4>
    <ul class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <li
        v-for="(feature, index) in statistics.feature_indices"
        :key="feature"
        data-testid="preview-statistic"
        :data-feature-index="feature"
        class="flex flex-col gap-3 rounded-xl border p-4"
      >
        <Badge variant="outline" class="self-start">
          {{ t('datasets.feature') }} {{ feature }}
        </Badge>
        <dl class="grid grid-cols-2 gap-3 text-sm">
          <div>
            <dt class="text-muted-foreground">{{ t('datasets.mean') }}</dt>
            <dd class="break-all font-medium tabular-nums">
              {{ statistics.means[index] ?? t('datasets.missing') }}
            </dd>
          </div>
          <div>
            <dt class="text-muted-foreground">{{ t('datasets.std') }}</dt>
            <dd class="break-all font-medium tabular-nums">
              {{ statistics.stds[index] ?? t('datasets.missing') }}
            </dd>
          </div>
        </dl>
      </li>
    </ul>
  </section>
</template>
