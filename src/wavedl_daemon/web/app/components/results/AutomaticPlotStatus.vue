<script setup lang="ts">
import type { AutomaticPlotState } from '../../lib/autoplot-controller';

const { t } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();

defineProps<{ state: AutomaticPlotState }>();

defineEmits<{ retry: []; stop: []; monitor: [runId: string] }>();
</script>

<template>
  <Alert v-if="state.status !== 'stopped'" class="mb-6" aria-live="polite">
    <AlertTitle>{{ t('results.diagnosticPlots') }}</AlertTitle>
    <AlertDescription>
      <p>{{ clientMessage(state.message) }}</p>
      <InfoChip
        v-if="state.status === 'armed' || state.status === 'planning'"
        :label="t('results.diagnosticPlots')"
        :text="t('results.automaticHelp')"
        uncertain
      />
      <div class="mt-3 flex flex-wrap gap-3">
        <Button
          v-if="state.status === 'error'"
          type="button"
          variant="outline"
          @click="$emit('retry')"
          >{{ t('results.retryPlots') }}</Button
        >
        <Button
          v-if="state.plotRunId"
          type="button"
          variant="outline"
          @click="$emit('monitor', state.plotRunId)"
          >{{ t('results.monitorPlot') }}</Button
        >
        <Button type="button" variant="outline" @click="$emit('stop')">{{
          state.status === 'submitted' ? t('results.dismiss') : t('results.disablePlots')
        }}</Button>
      </div>
    </AlertDescription>
  </Alert>
</template>
