<script setup lang="ts">
import { ChevronRight } from '@lucide/vue';
import type { ConnectionSession } from '../../lib/connection-session';
import type { DataSelection } from '../../lib/catalog-api';
import type { DatasetInspection } from '../../lib/dataset-inspection';

const { t } = useI18n({ useScope: 'global' });
const props = defineProps<{ session: ConnectionSession }>();
const selection = shallowRef<DataSelection | null>(null);
const inspection = shallowRef<DatasetInspection | null>(null);
let unsubscribe: (() => void) | undefined;

watch(() => props.session, (session) => {
  unsubscribe?.();
  let revision = -1;
  unsubscribe = toRaw(session).subscribe((state) => {
    if (state.revision !== revision || state.status !== 'connected') {
      selection.value = null;
      inspection.value = null;
    }

    revision = state.revision;
  });
}, { immediate: true });
onBeforeUnmount(() => unsubscribe?.());
</script>

<template>
  <section class="flex min-w-0 flex-col gap-7">
    <header>
      <h1>{{ t('datasets.workspaceTitle') }}</h1>
      <p class="text-muted-foreground">{{ t('datasets.workspaceHelp') }}</p>
    </header>
    <DatasetPathPicker :session="session" :selected="selection" @select="selection = $event" @inspected="inspection = $event" />
    <Collapsible v-if="inspection">
      <CollapsibleTrigger class="flex items-center gap-2 rounded-md text-sm text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring [&[data-state=open]_svg]:rotate-90"><ChevronRight class="size-4 transition-transform" aria-hidden="true" />{{ t('datasets.technicalDetails') }}</CollapsibleTrigger>
      <CollapsibleContent class="collapsible-motion pt-4"><DatasetInfoPanel :inspection="inspection" /></CollapsibleContent>
    </Collapsible>
    <DatasetPreviewPanel :session="session" :selection="selection" />
  </section>
</template>
