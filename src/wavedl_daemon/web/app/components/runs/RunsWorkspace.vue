<script setup lang="ts">
import { cn } from '@/lib/utils';
import { CircleAlert } from '@lucide/vue';
import { DaemonApi, DaemonApiError } from '../../lib/daemon-api';
import type { RunFilter } from '../../lib/run-overview';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  runsWorkspace: cn(
    'grid grid-cols-[minmax(16rem,_22rem)_minmax(0,_1fr)] items-start gap-6 min-w-0 max-w-352 mx-auto',
    "[&_>_*]:min-w-0 [&_>_*]:max-w-full [&[data-stacked='true']]:grid-cols-[minmax(0,_1fr)]",
    "[&[data-stacked='true']_.runs-sidebar]:static",
    "[&[data-stacked='true']_.runs-sidebar]:max-h-[min(32rem,_70dvh)]",
  ),
};

const { clientMessage } = useClientMessages();
const props = defineProps<{ api: DaemonApi; runId: string | null }>();

const emit = defineEmits<{
  connectionError: [error: DaemonApiError];
  select: [runId: string | null];
  unknownRun: [runId: string];
}>();

const {
  runs,
  selected,
  loading,
  message,
  page,
  staleRunIds,
  lastRefreshed,
  refresh,
  select,
  replay,
  cancel,
  close,
  pause,
  resume,
} = useRuns(props.api, (error) => emit('connectionError', error), true);

const { isActive } = usePageActive(resume, pause);
const root = useTemplateRef<HTMLElement>('root');
const stage = useTemplateRef<HTMLElement>('stage');
const { width } = useElementSize(() => (isActive.value ? root.value : null));

// Zero until measured: keep the split layout rather than flashing the stacked one.
const stacked = computed(() => width.value > 0 && width.value < 720);

const clock = useNow({
  controls: true,
  scheduler: (tick) => useIntervalFn(tick, 30_000, { immediate: false }),
});

usePageActive(() => {
  clock.now.value = new Date();
  clock.resume();
}, clock.pause);

const now = clock.now;
const query = shallowRef('');
const filter = shallowRef<RunFilter>('all');

// The monitor's run id, known before its first snapshot arrives.
const activeRunId = shallowRef<string | null>(null);

function open(runId: string) {
  activeRunId.value = runId;
  select(runId);
}

/** Stacked layouts move the reader to the detail they asked for. */
async function choose(runId: string) {
  emit('select', runId);

  if (!stacked.value) return;
  await nextTick();
  const reduced = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  stage.value?.scrollIntoView({ block: 'start', behavior: reduced ? 'auto' : 'smooth' });
}

function dismiss() {
  emit('select', null);
}

watch(
  () => props.runId,
  (runId) => {
    if (runId) open(runId);
    else {
      activeRunId.value = null;
      close();
    }
  },
  { immediate: true },
);

watch(
  () => selected.value?.message,
  (error) => {
    if (error instanceof DaemonApiError && error.code === 'http' && error.status === 404) {
      const id = activeRunId.value;
      close();

      if (id) emit('unknownRun', id);
    }
  },
);
</script>
<template>
  <div ref="root" :class="cn(styles.runsWorkspace)" :data-stacked="stacked ? 'true' : undefined">
    <MotionConfig reduced-motion="user">
      <Alert v-if="message" variant="warning" role="status" class="col-span-full">
        <CircleAlert aria-hidden="true" />
        <AlertDescription>{{ clientMessage(message) }}</AlertDescription>
      </Alert>
      <RunList
        v-model:query="query"
        v-model:filter="filter"
        class="runs-sidebar sticky top-0 max-h-[calc(100dvh_-_9rem)]"
        :runs="runs"
        :loading="loading"
        :page="page"
        :stale-run-ids="staleRunIds"
        :last-refreshed="lastRefreshed"
        :selected-run-id="selected ? activeRunId : null"
        :now="now"
        @refresh="refresh()"
        @page="refresh"
        @select="choose"
      />
      <div ref="stage" class="min-w-0 scroll-mt-4">
        <RunDetail
          v-if="selected"
          :key="activeRunId ?? 'run'"
          :snapshot="selected"
          :now="now"
          @cancel="cancel"
          @replay="replay"
          @close="dismiss"
        />
        <RunOverview v-else :runs="runs" :loading="loading" :now="now" @select="choose" />
      </div>
    </MotionConfig>
  </div>
</template>
