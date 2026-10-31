<script setup lang="ts">
import { DaemonApiError } from '../../lib/daemon-api';

defineOptions({ name: 'WorkstationResultsPage' });

// The kept-alive root: descendants that mount while hidden inherit its state.
usePageActive();

definePageMeta({
  key: 'results',
  title: 'shell.results',
  breadcrumb: 'shell.results',
  fullWidth: true,
  fillHeight: false,
});

const { state, api, session, reportFailure, monitorRun } = useWorkstation();
const { runId, unknown } = useRunRoute('results');

watch(
  [runId, api, () => state.value.revision, () => state.value.health?.daemon_id],
  async ([id, client], _prior, onCleanup) => {
    if (!id || !client) return;
    const controller = new AbortController();
    onCleanup(() => controller.abort());
    const failure = reportFailure.value;

    try {
      await client.getRun(id, AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]));
    } catch (error) {
      if (controller.signal.aborted) return;

      if (error instanceof DaemonApiError) {
        if (error.code === 'http' && error.status === 404) unknown(id);
        else failure(error);
      }
    }
  },
  { immediate: true },
);
</script>

<template>
  <ShellPage data-work-page="results">
    <ResultsContainer
      v-if="!!api"
      :key="`${state.health?.daemon_id}:${state.revision}`"
      :session="session"
      :run-id="runId"
      @run-selected="monitorRun"
    />
  </ShellPage>
</template>
