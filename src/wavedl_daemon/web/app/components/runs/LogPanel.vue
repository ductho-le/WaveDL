<script setup lang="ts">
import { cn } from '@/lib/utils';
import type { RunEvent } from '../../lib/event-stream';
import { eventDiagnostic } from '../../lib/run-monitor';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  logs: cn('min-w-0'),
  logsPanel: cn(
    'flex flex-col gap-3 h-full min-h-0 [&_.log-text]:max-h-none [&_.log-text]:flex-1',
    '[&_.log-text]:min-h-0',
  ),
};

const { t } = useI18n({ useScope: 'global' });
const props = defineProps<{ events: readonly RunEvent[]; truncated: boolean; panel?: boolean }>();
const diagnostics = computed(() => props.events.map(eventDiagnostic));
const text = computed(() => diagnostics.value.map((event) => event.text).join('\n'));
const shortened = computed(() => diagnostics.value.some((event) => event.truncated));
</script>
<template>
  <section
    :class="cn(styles.logs, panel && styles.logsPanel)"
    :aria-label="t('runs.runtimeDiagnostics')"
  >
    <h3>{{ t('runs.runtimeDiagnostics') }}</h3>
    <InfoChip
      v-if="truncated"
      :label="t('runs.boundedDiagnostics')"
      :text="t('runs.omittedDiagnostics')"
      uncertain
    />
    <InfoChip
      v-if="shortened"
      :label="t('runs.shortDiagnostics')"
      :text="t('runs.shortenedDiagnostics')"
    />
    <pre
      class="log-text whitespace-pre-wrap wrap-anywhere max-h-104 overflow-auto [font:0.8rem/1.6_ui-monospace,_monospace] p-4 bg-muted rounded-[0.75rem]"
      role="log"
      tabindex="0"
      aria-live="off"
      :aria-label="t('runs.runtimeDiagnostics')"
      >{{ text || t('runs.noEvents') }}</pre>
  </section>
</template>
