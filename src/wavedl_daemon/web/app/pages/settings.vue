<script setup lang="ts">
definePageMeta({
  keepalive: false,
  title: 'shell.settings',
  breadcrumb: 'shell.settings',
  fullWidth: true,
  fillHeight: false,
});

const { state, appearance, appearanceChanged } = useWorkstation();
</script>

<template>
  <ShellPage data-settings-page>
    <SettingsWorkspace
      :daemon-id="state.health?.daemon_id ?? null"
      :appearance="appearance"
      @appearance-changed="appearanceChanged"
    />
  </ShellPage>
</template>
