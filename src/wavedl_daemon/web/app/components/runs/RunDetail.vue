<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion, useReducedMotion } from 'motion-v';
import { ArrowLeft, Check, Copy, Ellipsis, OctagonX, RotateCcw, ScrollText, X } from '@lucide/vue';
import { evaluationResults } from '../../lib/evaluation-results';
import { canRequestCancellation, type RunMonitorSnapshot } from '../../lib/run-monitor';
import { durationText, pathName, remainingSeconds, runCreatedAt, shortRunId } from '../../lib/run-overview';
import {
  runBackendMessageKeys,
  runCancellationMessageKeys,
  runStreamMessageKeys,
} from '../../i18n/dynamic-keys';

type RunDetailTab = 'overview' | 'metrics' | 'configuration' | 'logs' | 'evaluation';

const styles = {
  card: cn('@container h-full min-h-0 min-w-0 gap-0 overflow-hidden py-0'),
  header: cn('flex shrink-0 flex-col gap-3 px-4 py-3'),
  stream: cn(
    'size-2 shrink-0 rounded-full bg-muted-foreground/40',
    "[&[data-stream='live']]:bg-success",
    "[&[data-stream='connecting']]:bg-warning [&[data-stream='reconnecting']]:bg-warning",
    "[&[data-stream='error']]:bg-destructive",
  ),
  facts: cn('flex flex-wrap items-baseline gap-x-2 gap-y-1 text-xs text-muted-foreground [&>li+li]:before:mr-2 [&>li+li]:before:content-[\'·\'] [&>li]:min-w-0 [&>li]:wrap-anywhere'),
  tabs: cn('flex min-h-0 flex-1 flex-col gap-0'),
  tabList: cn('h-10 w-full shrink-0 justify-start overflow-x-auto rounded-none border-b bg-transparent p-0 px-2'),
  tabTrigger: cn(
    'h-full flex-none rounded-none border-0 border-b-2 border-transparent px-3 text-xs text-muted-foreground shadow-none',
    'data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:shadow-none',
    'dark:data-[state=active]:border-primary dark:data-[state=active]:bg-transparent',
  ),
  body: cn('min-h-0 flex-1 overflow-y-auto p-4'),
  configuration: cn('grid gap-0 text-sm [&>div]:grid [&>div]:grid-cols-[minmax(6rem,1fr)_minmax(0,2fr)] [&>div]:gap-4 [&>div]:border-b [&>div]:py-3 [&_dt]:text-muted-foreground [&_dd]:min-w-0 [&_dd]:wrap-anywhere'),
};

const { t, locale } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();
const shellLayout = useShellLayout();
const { isActive } = usePageActive();
const labels = useRunLabels();
const reducedMotion = useReducedMotion();

function backendLabel(backend: string): string {
  const key = Object.entries(runBackendMessageKeys).find(([name]) => name === backend)?.[1];

  return key === undefined ? backend : t(key);
}

const props = withDefaults(defineProps<{
  snapshot: RunMonitorSnapshot;
  now?: Date;
  back?: boolean;
  animateEntrance?: boolean;
}>(), { back: false, animateEntrance: true });

const title = useTemplateRef<HTMLElement>('title');

function focusTitle() {
  title.value?.focus({ preventScroll: true });
}

defineExpose({ focusTitle });

defineEmits<{ cancel: []; replay: []; close: [] }>();

const tab = shallowRef<RunDetailTab>('overview');
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

const healthIssues = computed(() => metricsUnavailable.value
  ? 0
  : (props.snapshot.summary.current?.health.filter((row) => row.severity !== 'good').length ?? 0));

const metricsUnconfirmed = computed(
  () =>
    props.snapshot.run?.uncertainty === true || props.snapshot.run?.state === 'supervision_lost',
);

const metricSummary = computed(() =>
  metricsUnavailable.value ? { progress: null, best: null, latest: null } : {
    ...props.snapshot.summary,
    progress: props.snapshot.summary.progress ?? props.snapshot.run?.progress ?? null,
  },
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

const percent = computed(() =>
  metricSummary.value.progress
    ? new Intl.NumberFormat(locale.value, { style: 'percent', maximumFractionDigits: 0 }).format(
        Math.min(1, Math.max(0, metricSummary.value.progress.fraction)),
      )
    : '',
);

const elapsed = computed(() => metricSummary.value.current?.elapsedSeconds ?? null);

// Last-known evidence must not read as a live countdown.
const remaining = computed(() =>
  metricsUnconfirmed.value ? null : remainingSeconds(metricSummary.value.progress, elapsed.value),
);

const hasEvaluation = computed(() => props.snapshot.run !== null
  && props.snapshot.freshness === 'fresh'
  && evaluationResults(props.snapshot.run, props.snapshot.events) !== null);

watch(hasEvaluation, (available) => {
  if (!available && tab.value === 'evaluation') tab.value = 'overview';
});

const configuration = computed(() => {
  const run = props.snapshot.run;
  const summary = run?.summary;
  const unavailable = t('runs.unavailable');

  return [
    { label: t('runs.configModel'), value: summary?.model ?? unavailable },
    { label: t('runs.configDataset'), value: summary?.data_path ?? unavailable },
    { label: t('runs.configEpochs'), value: summary?.epochs ?? unavailable },
    { label: t('runs.configFolds'), value: summary?.cv ?? unavailable },
    { label: t('runs.configSource'), value: summary?.source_run_id ?? unavailable },
    { label: t('runs.configOperation'), value: run ? labels.operation(run) : unavailable },
    { label: t('runs.configBackend'), value: run ? backendLabel(run.backend) : unavailable },
    { label: t('runs.configAttempt'), value: run?.attempt ?? t('runs.notAllocated') },
    { label: t('runs.configCreated'), value: created.value?.text ?? unavailable },
  ];
});
</script>
<template>
  <motion.div
    as-child
    :initial="animateEntrance ? { opacity: 0, y: 12 } : false"
    :animate="{ opacity: 1, y: 0 }"
    :transition="reducedMotion ? { duration: 0 } : { type: 'spring', stiffness: 300, damping: 30 }"
  >
    <Card :class="styles.card" :aria-label="t('runs.selected')">
    <header :class="styles.header">
      <div class="flex flex-wrap items-center gap-2">
        <h3 ref="title" tabindex="-1" class="min-w-[min(100%,10rem)] flex-1 truncate text-lg font-semibold tracking-tight outline-none focus-visible:ring-2 focus-visible:ring-ring" :title="snapshot.run ? labels.title(snapshot.run) : undefined">
          {{ snapshot.run ? labels.title(snapshot.run) : t('runs.loadingRuns') }}
        </h3>
        <RunStateChip v-if="snapshot.run" :run="snapshot.run" :stale="snapshot.freshness !== 'fresh'" />
        <Badge v-if="healthIssues > 0" as="button" type="button" variant="outline" class="cursor-pointer" @click="tab = 'metrics'">
          <span class="size-1.5 rounded-full bg-warning" aria-hidden="true" />
          {{ t('runs.healthWarnings', { count: healthIssues }, healthIssues) }}
        </Badge>
        <span :class="styles.stream" :data-stream="snapshot.stream" :title="t('runs.eventConnection', { status: streamLabel })">
          <span class="sr-only">{{ t('runs.eventConnection', { status: streamLabel }) }}</span>
        </span>
        <div class="ml-auto flex shrink-0 items-center gap-1">
          <Button variant="ghost" size="sm" :aria-label="t('runs.replay')" :title="t('runs.replay')" @click="$emit('replay')"><RotateCcw aria-hidden="true" /><span class="hidden @min-[48rem]:inline">{{ t('runs.replay') }}</span></Button>
          <Button variant="outline" size="sm" class="text-destructive" :disabled="!canCancel" :aria-label="snapshot.cancellation === 'sending' ? t('runs.requesting') : t('runs.requestCancel')" :title="t('runs.requestCancel')" @click="$emit('cancel')">
            <OctagonX aria-hidden="true" /><span class="hidden @min-[48rem]:inline">{{ snapshot.cancellation === 'sending' ? t('runs.requesting') : t('runs.requestCancel') }}</span>
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger as-child>
              <Button variant="ghost" size="icon" class="size-8" :aria-label="t('runs.moreActions')"><Ellipsis aria-hidden="true" /></Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem v-if="snapshot.run && canCopy" @select="copy(snapshot.run.run_id)"><Copy aria-hidden="true" />{{ t('runs.copyRunId') }}</DropdownMenuItem>
              <DropdownMenuItem v-if="logTarget" data-testid="run-open-logs" @select="shellLayout.setActiveTab('logs')"><ScrollText aria-hidden="true" />{{ t('runs.openLogsPanel') }}</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="ghost" size="icon" class="size-8 text-muted-foreground" :aria-label="back ? t('runs.back') : t('runs.close')" :title="back ? t('runs.back') : t('runs.close')" @click="$emit('close')">
            <component :is="back ? ArrowLeft : X" aria-hidden="true" />
          </Button>
        </div>
      </div>
      <ul v-if="snapshot.run" :class="styles.facts">
        <li v-for="fact in facts" :key="fact">{{ fact }}</li>
        <li v-if="created" :title="created.relative ?? undefined">{{ t('runs.createdAt', { time: created.text }) }}</li>
        <li><code>{{ shortRunId(snapshot.run.run_id) }}</code></li>
      </ul>
      <div v-if="metricSummary.progress" class="flex flex-wrap items-center gap-x-3 gap-y-2 text-xs tabular-nums">
        <span>{{ t('runs.epoch', { epoch: metricSummary.progress.epoch, total: metricSummary.progress.totalEpochs }) }}</span>
        <Progress class="h-1.5 min-w-20 flex-1" :model-value="metricSummary.progress.fraction * 100" :aria-label="t('runs.epochProgress')" :aria-valuetext="percent" />
        <span class="text-muted-foreground">
          <span v-if="elapsed !== null">{{ durationText(elapsed, locale) }}</span>
          <span v-if="elapsed !== null && remaining !== null" aria-hidden="true"> · </span>
          <span v-if="remaining !== null" :title="t('runs.remainingNote')">≈ {{ durationText(remaining, locale) }} {{ t('runs.remaining') }}</span>
        </span>
      </div>
      <span class="sr-only" role="status" aria-live="polite">{{ copied ? t('runs.runIdCopied') : '' }}</span>
      <div
        v-if="snapshot.message || snapshot.cancellation !== 'idle' || snapshot.freshness !== 'fresh'"
        class="flex flex-col gap-[0.35rem] text-muted-foreground text-[0.82rem] [&:not(:has(p:not(.sr-only)))]:contents"
      >
        <p v-if="snapshot.freshness !== 'fresh'" class="sr-only" role="status" aria-live="polite">{{ t('runs.lifecycleNote') }}</p>
        <p v-if="snapshot.message" role="status">{{ clientMessage(snapshot.message) }}</p>
        <p v-if="snapshot.cancellation !== 'idle'" role="status">{{ t('runs.cancellation', { status: t(runCancellationMessageKeys[snapshot.cancellation]) }) }}</p>
      </div>
    </header>
    <Separator />
    <Tabs v-model="tab" :class="styles.tabs">
      <TabsList :class="styles.tabList" :aria-label="t('runs.selected')">
        <TabsTrigger value="overview" :class="styles.tabTrigger">{{ t('runs.tabOverview') }}</TabsTrigger>
        <TabsTrigger value="metrics" :class="styles.tabTrigger">{{ healthIssues > 0 ? t('runs.tabMetricsCount', { count: healthIssues }) : t('runs.tabMetrics') }}</TabsTrigger>
        <TabsTrigger value="configuration" :class="styles.tabTrigger">{{ t('runs.tabConfiguration') }}</TabsTrigger>
        <TabsTrigger value="logs" :class="styles.tabTrigger">{{ t('runs.tabLogs') }}</TabsTrigger>
        <TabsTrigger v-if="hasEvaluation" value="evaluation" :class="styles.tabTrigger">{{ t('runs.tabEvaluation') }}</TabsTrigger>
      </TabsList>
      <TabsContent value="overview" class="flex min-h-0 flex-1 flex-col overflow-y-auto">
        <RunStats :summary="metricSummary" :records="chartRecords" />
        <Separator />
        <div class="flex min-h-0 flex-1 flex-col p-4">
          <MetricsChart :records="chartRecords" :unavailable="metricsUnavailable" />
        </div>
      </TabsContent>
      <TabsContent value="metrics" :class="styles.body">
        <MetricSummary :summary="metricSummary" :unconfirmed="metricsUnconfirmed" />
      </TabsContent>
      <TabsContent value="configuration" :class="styles.body">
        <dl :class="styles.configuration">
          <div v-for="fact in configuration" :key="fact.label"><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div>
          <div>
            <dt>{{ t('runs.configRunId') }}</dt>
            <dd v-if="snapshot.run" class="flex items-start gap-1">
              <code class="min-w-0 flex-1 wrap-anywhere">{{ snapshot.run.run_id }}</code>
              <Button v-if="canCopy" variant="ghost" size="icon" class="size-7 shrink-0" :aria-label="t('runs.copyRunId')" @click="copy(snapshot.run.run_id)"><Check v-if="copied" aria-hidden="true" /><Copy v-else aria-hidden="true" /></Button>
            </dd>
            <dd v-else>{{ t('runs.unavailable') }}</dd>
          </div>
        </dl>
      </TabsContent>
      <TabsContent value="logs" :class="styles.body">
        <LogPanel :events="snapshot.events" :truncated="snapshot.truncated" panel />
      </TabsContent>
      <TabsContent v-if="hasEvaluation" value="evaluation" :class="styles.body">
        <EvaluationResultsPanel :run="snapshot.run" :events="snapshot.events" :freshness="snapshot.freshness" />
      </TabsContent>
    </Tabs>
    <Teleport v-if="logTarget && parkedLogs && snapshot.events.length" :to="isActive ? logTarget : parkedLogs" defer>
      <LogPanel :events="snapshot.events" :truncated="snapshot.truncated" panel />
    </Teleport>
    </Card>
  </motion.div>
</template>
