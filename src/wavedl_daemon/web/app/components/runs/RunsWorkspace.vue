<script setup lang="ts">
import { cn } from '@/lib/utils';
import { CircleAlert } from '@lucide/vue';
import { DaemonApi, DaemonApiError } from '../../lib/daemon-api';

const styles = {
  runsWorkspace: cn('flex h-full min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-hidden'),
};

const { clientMessage } = useClientMessages();
const shellLayout = useShellLayout();
const props = defineProps<{ api: DaemonApi; runId: string | null }>();

const emit = defineEmits<{
  connectionError: [error: DaemonApiError];
  select: [runId: string | null];
  unknownRun: [runId: string];
}>();

const {
  runs, selected, loading, message, page, staleRunIds, lastRefreshed,
  refresh, select, replay, cancel, close, pause, resume,
} = useRuns(props.api, (error) => emit('connectionError', error), true);

const { isActive } = usePageActive(resume, pause);
const root = useTemplateRef<HTMLElement>('root');
const detail = useTemplateRef('detail');
const { width } = useElementSize(() => isActive.value ? root.value : null);
const stacked = computed(() => width.value > 0 && width.value < 720);
const activeRunId = computed(() => props.runId);
const entered = shallowRef(false);

const clock = useNow({
  controls: true,
  scheduler: (tick) => useIntervalFn(tick, 30_000, { immediate: false }),
});

usePageActive(() => {
  clock.now.value = new Date();
  clock.resume();
}, clock.pause);

const now = clock.now;

function choose(runId: string) {
  if (isActive.value && runId !== activeRunId.value) emit('select', runId);
}

function dismiss() {
  if (isActive.value) emit('select', null);
}

async function focusDetail() {
  await nextTick();

  if (isActive.value && activeRunId.value) detail.value?.focusTitle();
}

const overlayContent = '[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"], [role="combobox"], [role="tooltip"], [data-slot="popover-content"], [data-slot="dropdown-menu-content"], [data-slot="combobox-content"]';

function onEscape(event: KeyboardEvent) {
  if (event.key !== 'Escape' || !isActive.value || !activeRunId.value || event.defaultPrevented || event.isComposing) return;

  if (shellLayout.commandPaletteOpen.value || shellLayout.isFocusMode.value || (!shellLayout.isDockedLayout.value && shellLayout.isPanelSheetVisible.value)) return;

  if (event.target instanceof Element && event.target.closest(overlayContent)) return;

  const openOverlay = Array.from(document.querySelectorAll<HTMLElement>(overlayContent)).some((element) =>
    element.dataset.state !== 'closed' &&
    (element.getAttribute('role') !== 'combobox' || element.getAttribute('aria-expanded') === 'true') &&
    !element.closest('[inert], [aria-hidden="true"]') && element.getClientRects().length > 0,
  );

  if (openOverlay) return;
  event.preventDefault();
  dismiss();
}

watch(() => props.runId, (runId, previous) => {
  if (previous !== undefined) entered.value = true;

  if (runId) select(runId);
  else close();
}, { immediate: true });

watch(() => selected.value?.message, (error) => {
  if (error instanceof DaemonApiError && error.code === 'http' && error.status === 404) {
    const id = activeRunId.value;
    close();

    if (id) emit('unknownRun', id);
  }
});
</script>

<template>
  <div ref="root" :class="styles.runsWorkspace" :data-stacked="stacked" @keydown="onEscape">
    <MotionConfig reduced-motion="user">
      <Alert v-if="message" variant="warning" role="status" class="shrink-0">
        <CircleAlert aria-hidden="true" />
        <AlertDescription>{{ clientMessage(message) }}</AlertDescription>
      </Alert>
      <RunList
        :runs="runs"
        :loading="loading"
        :page="page"
        :stale-run-ids="staleRunIds"
        :last-refreshed="lastRefreshed"
        :selected-run-id="activeRunId"
        :folded="activeRunId !== null"
        :stacked="stacked"
        :now="now"
        @refresh="refresh()"
        @page="refresh"
        @select="choose"
        @focus-detail="focusDetail"
      >
        <template #detail>
          <RunDetail
            v-if="selected"
            ref="detail"
            :key="activeRunId ?? 'run'"
            :snapshot="selected"
            :now="now"
            :back="stacked"
            :animate-entrance="entered"
            @cancel="cancel"
            @replay="replay"
            @close="dismiss"
          />
        </template>
      </RunList>
    </MotionConfig>
  </div>
</template>
