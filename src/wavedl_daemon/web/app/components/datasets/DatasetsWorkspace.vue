<script setup lang="ts">
import type { ConnectionSession } from '../../lib/connection-session';
import type { DataSelection } from '../../lib/catalog-api';

const { t } = useI18n({ useScope: 'global' });

defineProps<{ session: ConnectionSession }>();

const selection = shallowRef<DataSelection | null>(null);
</script>

<template>
  <section class="flex flex-col gap-7 min-w-0 [&_p]:text-muted-foreground">
    <header>
      <h1>{{ t('datasets.workspaceTitle') }}</h1>
      <p>{{ t('datasets.workspaceHelp') }}</p>
    </header>
    <DatasetPathPicker :session="session" @select="selection = $event" />
    <DatasetInfoPanel :session="session" :selection="selection" />
    <DatasetPreviewPanel :session="session" :selection="selection" />
  </section>
</template>
