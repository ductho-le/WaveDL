<script setup lang="ts">
definePageMeta({
  title: 'shell.datasets',
  breadcrumb: 'shell.datasets',
  fullWidth: true,
  fillHeight: false,
});

defineOptions({ name: 'WorkstationDatasetsPage' });

// The kept-alive root: descendants that mount while hidden inherit its state.
usePageActive();

const { state, api, session } = useWorkstation();
</script>

<template>
  <ShellPage data-work-page="datasets">
    <DatasetsWorkspace
      v-if="!!api"
      :key="`${state.health?.daemon_id}:${state.revision}`"
      :session="session"
    />
  </ShellPage>
</template>
