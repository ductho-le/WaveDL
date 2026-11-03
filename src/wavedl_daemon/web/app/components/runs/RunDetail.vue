<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import { Check, Copy, OctagonX, RotateCcw, ScrollText, X } from '@lucide/vue';
import { canRequestCancellation, type RunMonitorSnapshot } from '../../lib/run-monitor';
import { pathName, runCreatedAt, shortRunId } from '../../lib/run-overview';
import {
  runBackendMessageKeys,
  runCancellationMessageKeys,
  runStreamMessageKeys,
} from '../../i18n/dynamic-keys';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  runHero: cn(
    'relative flex flex-col gap-[0.85rem] overflow-hidden p-[1.25rem_1.4rem_1.3rem] rounded-[1.4rem]',
    'bg-surface',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_10%,_transparent),_0_24px_48px_-32px_rgb(0_0_0_/_0.35),_0_2px_6px_rgb(0_0_0_/_0.04)]',
    "[&::before]:[content:''] [&::before]:absolute [&::before]:[inset:0_0_auto] [&::before]:h-[3px]",
    '[&::before]:[background:linear-gradient(90deg,_#0a63d8,_#5aa9ff,_#35c46a)]',
  ),
  runStream: cn(
    'inline-flex items-center gap-[0.4rem] p-[0.2rem_0.6rem] rounded-full text-muted-foreground',
    'text-[0.72rem] font-[550] bg-[color-mix(in_srgb,_var(--foreground)_5%,_transparent)]',
    '[&_>_span[aria-hidden]::first-letter]:uppercase [&_i]:w-[0.45rem] [&_i]:h-[0.45rem]',
    '[&_i]:rounded-full [&_i]:bg-[color-mix(in_srgb,_var(--foreground)_25%,_transparent)]',
    "[&[data-stream='live']_i]:bg-success",
    "[&[data-stream='live']_i]:shadow-[0_0_0_3px_color-mix(in_srgb,_var(--success)_22%,_transparent)]",
    "[&[data-stream='connecting']_i]:bg-warning",
    "[&[data-stream='connecting']_i]:[animation:run-stream-blink_1.2s_ease-in-out_infinite]",
    "[&[data-stream='reconnecting']_i]:bg-warning",
    "[&[data-stream='reconnecting']_i]:[animation:run-stream-blink_1.2s_ease-in-out_infinite]",
    "[&[data-stream='error']_i]:bg-destructive motion-reduce:[&_i]:animate-none",
  ),
  runHeroFacts: cn(
    'flex flex-wrap gap-[0.35rem] m-0 p-0 list-none [&_li]:max-w-full [&_li]:overflow-hidden',
    '[&_li]:p-[0.2rem_0.6rem] [&_li]:rounded-full [&_li]:text-muted-foreground [&_li]:text-xs',
    '[&_li]:font-medium [&_li]:text-ellipsis [&_li]:whitespace-nowrap',
    '[&_li]:bg-[color-mix(in_srgb,_var(--foreground)_5%,_transparent)]',
  ),
};

const { t, locale } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();
const shellLayout = useShellLayout();
const { isActive } = usePageActive();
const labels = useRunLabels();

function backendLabel(backend: string): string {
  const key = Object.entries(runBackendMessageKeys).find(([name]) => name === backend)?.[1];

  return key === undefined ? backend : t(key);
}

const props = defineProps<{ snapshot: RunMonitorSnapshot; now?: Date }>();

defineEmits<{ cancel: []; replay: []; close: [] }>();

const logTarget = shallowRef<HTMLElement | null>(null);
const parkedLogs = shallowRef<HTMLElement | null>(null);

onMounted(() => {
  parkedLogs.value = document.createElement('div');
  logTarget.value = document.querySelector<HTMLElement>('#shell-panel-tab-logs');
});

const { copy, copied, isSupported: canCopy } = useClipboard({ copiedDuring: 1600 });
const metricsUnavailable = computed(
  () => props.snapshot.freshness !== 'fresh' || props.snapshot.run?.attempt == null,
);

const chartRecords = computed(() => (metricsUnavailable.value ? [] : props.snapshot.metrics));

const metricsUnconfirmed = computed(
  () =>
    props.snapshot.run?.uncertainty === true || props.snapshot.run?.state === 'supervision_lost',
);

const metricSummary = computed(() =>
  metricsUnavailable.value ? { progress: null, best: null, latest: null } : props.snapshot.summary,
);

const canCancel = computed(() =>
  canRequestCancellation(props.snapshot.run, props.snapshot.cancellation),
);

const created = computed(() => {
  const run = props.snapshot.run;
  const date = run ? runCreatedAt(run) : null;

  if (!run || !date) return null;

  return {
    text: new Intl.DateTimeFormat(locale.value, { dateStyle: 'medium', timeStyle: 'short' }).format(
      date,
    ),
    relative: labels.created(run, props.now ?? new Date()),
  };
});

/** Approved inputs first, then where and how it executes. */
const facts = computed(() => {
  const run = props.snapshot.run;

  if (!run) return [];
  const summary = run.summary;

  return [
    summary?.name ? summary.model : null,
    pathName(summary?.data_path) || null,
    summary?.epochs ? t('runs.epochCount', { count: summary.epochs }) : null,
    summary?.cv ? t('runs.cvFolds', { count: summary.cv }) : null,
    summary?.source_run_id ? t('runs.fromRun', { id: shortRunId(summary.source_run_id) }) : null,
    labels.operation(run),
    backendLabel(run.backend),
    t('runs.attempt', { attempt: run.attempt ?? t('runs.notAllocated') }),
  ].filter((fact): fact is string => fact !== null && fact !== undefined && fact !== '');
});

const streamLabel = computed(() => t(runStreamMessageKeys[props.snapshot.stream]));
</script>
<template>
  <section
    class="flex flex-col gap-4 min-w-0 max-w-full [&_[data-slot='card']]:min-w-0 [&_[data-slot='card']]:max-w-full"
    :aria-label="t('runs.selected')"
  >
    <motion.header
      :class="cn(styles.runHero)"
      :initial="{ opacity: 0, y: 12 }"
      :animate="{ opacity: 1, y: 0 }"
      :transition="{ type: 'spring', stiffness: 300, damping: 30 }"
    >
      <div class="flex items-center gap-[0.6rem]">
        <h2 class="flex-1 text-muted-foreground text-[0.78rem] font-semibold tracking-[0.01em]">
          {{ t('runs.detail') }}
        </h2>
        <span
          :class="cn(styles.runStream)"
          :data-stream="snapshot.stream"
          :title="t('runs.eventConnection', { status: streamLabel })"
          ><i aria-hidden="true" /><span class="sr-only">{{
            t('runs.eventConnection', { status: streamLabel })
          }}</span
          ><span aria-hidden="true">{{ streamLabel }}</span></span
        >
        <Button
          variant="ghost"
          size="icon"
          class="rounded-full text-muted-foreground"
          :aria-label="t('runs.close')"
          :title="t('runs.close')"
          @click="$emit('close')"
          ><X aria-hidden="true"
        /></Button>
      </div>
      <template v-if="snapshot.run">
        <h3
          class="-mt-1 wrap-anywhere text-[clamp(1.35rem,_2.4vw,_1.75rem)] font-[650] tracking-[-0.03em] leading-[1.15]"
        >
          {{ labels.title(snapshot.run) }}
        </h3>
        <ul :class="cn(styles.runHeroFacts)">
          <li v-for="fact in facts" :key="fact">{{ fact }}</li>
          <li v-if="created" :title="created.relative ?? undefined">
            {{ t('runs.createdAt', { time: created.text }) }}
          </li>
        </ul>
        <div class="flex flex-wrap items-start justify-between gap-3">
          <RunStateChip
            :run="snapshot.run"
            :stale="snapshot.freshness !== 'fresh'"
            :progress="metricSummary.progress"
          />
          <span class="inline-flex items-center gap-[0.15rem] min-w-0 max-w-full"
            ><code
              class="identity overflow-hidden text-muted-foreground [font:0.75rem/1.4_ui-monospace,_'SF_Mono',_monospace] text-ellipsis whitespace-nowrap"
              >{{ snapshot.run.run_id }}</code
            ><Button
              v-if="canCopy"
              variant="ghost"
              size="icon"
              class="size-7 flex-none rounded-[0.5rem] text-muted-foreground"
              :aria-label="t('runs.copyRunId')"
              :title="t('runs.copyRunId')"
              @click="copy(snapshot.run.run_id)"
              ><Check v-if="copied" class="size-[0.85rem]" aria-hidden="true" /><Copy
                v-else
                class="size-[0.85rem]"
                aria-hidden="true" /></Button
          ></span>
          <span class="sr-only" role="status" aria-live="polite">{{
            copied ? t('runs.runIdCopied') : ''
          }}</span>
        </div>
      </template>
      <div
        class="flex flex-wrap gap-2 pt-[0.85rem] [border-top:1px_solid_color-mix(in_srgb,_var(--border)_65%,_transparent)]! [&_button]:rounded-[0.7rem]"
      >
        <Button
          variant="outline"
          class="[&:not(:disabled)]:text-destructive dark:[&:not(:disabled)]:text-[#ff8a8a]"
          :disabled="!canCancel"
          @click="$emit('cancel')"
          ><OctagonX aria-hidden="true" />{{
            snapshot.cancellation === 'sending' ? t('runs.requesting') : t('runs.requestCancel')
          }}</Button
        ><Button variant="outline" @click="$emit('replay')"
          ><RotateCcw aria-hidden="true" />{{ t('runs.replay') }}</Button
        ><Button
          v-if="logTarget"
          variant="outline"
          data-testid="run-open-logs"
          @click="shellLayout.setActiveTab('logs')"
          ><ScrollText aria-hidden="true" />{{ t('runs.logs') }}</Button
        >
      </div>
      <div
        v-if="
          snapshot.message || snapshot.cancellation !== 'idle' || snapshot.freshness !== 'fresh'
        "
        class="flex flex-col gap-[0.35rem] text-muted-foreground text-[0.82rem] [&:not(:has(p:not(.sr-only)))]:contents"
      >
        <p v-if="snapshot.freshness !== 'fresh'" class="sr-only" role="status" aria-live="polite">
          {{ t('runs.lifecycleNote') }}
        </p>
        <p v-if="snapshot.message" role="status">
          {{ clientMessage(snapshot.message) }}
        </p>
        <p v-if="snapshot.cancellation !== 'idle'" role="status">
          {{
            t('runs.cancellation', { status: t(runCancellationMessageKeys[snapshot.cancellation]) })
          }}
        </p>
      </div>
    </motion.header>
    <MetricSummary
      :summary="metricSummary"
      :records="chartRecords"
      :unconfirmed="metricsUnconfirmed"
    />
    <MetricsChart :records="chartRecords" :unavailable="metricsUnavailable" />
    <EvaluationResultsPanel
      class="p-[1.25rem_1.4rem] rounded-[1.25rem] bg-surface batch-r-surface-shadow"
      :run="snapshot.run"
      :events="snapshot.events"
      :freshness="snapshot.freshness"
    />
    <Teleport
      v-if="logTarget && parkedLogs && snapshot.events.length"
      :to="isActive ? logTarget : parkedLogs"
      defer
    >
      <LogPanel :events="snapshot.events" :truncated="snapshot.truncated" panel />
    </Teleport>
    <LogPanel
      v-else-if="!logTarget"
      class="p-[1.25rem_1.4rem] rounded-[1.25rem] bg-surface batch-r-surface-shadow"
      :events="snapshot.events"
      :truncated="snapshot.truncated"
    />
  </section>
</template>
