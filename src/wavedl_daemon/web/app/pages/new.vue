<script setup lang="ts">
import type { RunView } from '../lib/daemon-api';

definePageMeta({
  title: 'shell.newTrainingRun',
  breadcrumb: 'shell.newTrainingRun',
  fullWidth: true,
  fillHeight: false,
});

defineOptions({ name: 'WorkstationNewPage' });

// The kept-alive root: descendants that mount while hidden inherit its state.
usePageActive();

const { api, session, monitorRun, automaticPlots } = useWorkstation();

function submitted(run: RunView): void {
  void monitorRun(run.run_id);
}
</script>

<template>
  <ShellPage data-work-page="project">
    <ProjectContainer
      v-show="!!api"
      :session="session"
      @submitted="submitted"
      @submitted-plan="automaticPlots.capture"
    />
  </ShellPage>
</template>
