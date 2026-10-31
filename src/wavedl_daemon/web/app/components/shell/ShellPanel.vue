<script setup lang="ts">
import { PanelRightClose } from '@lucide/vue';
import {
  SHELL_PANEL_DEFAULT_WIDTH,
  SHELL_PANEL_TABS,
  SHELL_PANEL_MAX_WIDTH,
  SHELL_PANEL_MIN_WIDTH,
  type ShellPanelTabDefinition,
  type ShellPanelTabId,
} from '../../shell/constants';

const props = defineProps<{
  tabs: readonly ShellPanelTabDefinition[];
  activeTab: ShellPanelTabId;
  width: number;
}>();

const emit = defineEmits<{
  'update:activeTab': [tab: ShellPanelTabId];
  resize: [width: number];
  collapse: [];
}>();

const { t } = useI18n({ useScope: 'global' });
const isResizing = shallowRef(false);
let startX = 0;
let startWidth = 0;

const expandableTabs = computed(() =>
  props.tabs.map((tab) => ({ id: tab.id, title: t(tab.labelKey), icon: tab.icon })),
);

const activeTabIndex = computed<number | null>({
  get: () => {
    const index = props.tabs.findIndex((tab) => tab.id === props.activeTab);

    return index < 0 ? null : index;
  },
  set: (index) => {
    if (index === null) return;
    const tab = props.tabs[index];

    if (tab) emit('update:activeTab', tab.id);
  },
});

function onResizeStart(event: PointerEvent): void {
  if (!(event.currentTarget instanceof HTMLElement)) return;
  isResizing.value = true;
  startX = event.clientX;
  startWidth = props.width;

  event.currentTarget.setPointerCapture?.(event.pointerId);
}

function onResizeMove(event: PointerEvent): void {
  if (isResizing.value) emit('resize', startWidth - (event.clientX - startX));
}

function onResizeEnd(event: PointerEvent): void {
  if (!isResizing.value) return;
  isResizing.value = false;

  if (
    event.currentTarget instanceof HTMLElement &&
    event.currentTarget.hasPointerCapture?.(event.pointerId)
  ) {
    event.currentTarget.releasePointerCapture(event.pointerId);
  }
}

function onResizeKey(event: KeyboardEvent): void {
  const step = 24;

  if (event.key === 'ArrowLeft') emit('resize', props.width + step);
  else if (event.key === 'ArrowRight') emit('resize', props.width - step);
  else if (event.key === 'Home') emit('resize', SHELL_PANEL_DEFAULT_WIDTH);
  else return;
  event.preventDefault();
}
</script>

<template>
  <aside
    class="shell-panel relative flex h-full min-h-0 min-w-0 flex-col"
    :style="{ width: `${props.width}px` }"
    :data-testid="'shell-companion'"
    :aria-label="t('shell.companionPanel')"
    @keydown.esc.prevent="emit('collapse')"
  >
    <div
      class="group absolute -left-1.5 inset-y-0 z-20 hidden w-3 cursor-col-resize touch-none focus-visible:outline-none lg:block"
      role="separator"
      tabindex="0"
      aria-orientation="vertical"
      :aria-label="t('shell.resizeCompanionPanel')"
      :aria-valuenow="props.width"
      :aria-valuemin="SHELL_PANEL_MIN_WIDTH"
      :aria-valuemax="SHELL_PANEL_MAX_WIDTH"
      @pointerdown="onResizeStart"
      @pointermove="onResizeMove"
      @pointerup="onResizeEnd"
      @pointercancel="onResizeEnd"
      @keydown="onResizeKey"
      @dblclick="emit('resize', SHELL_PANEL_DEFAULT_WIDTH)"
    >
      <span
        class="absolute inset-y-6 left-1/2 w-px bg-border opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100"
        :class="{ 'opacity-100 bg-primary': isResizing }"
      />
    </div>

    <div
      class="flex shrink-0 items-center gap-1 p-2.5 max-lg:pt-[max(0.625rem,env(safe-area-inset-top,0px))]"
    >
      <ExpandableTabs
        v-model="activeTabIndex"
        :tabs="expandableTabs"
        :aria-label="t('shell.companionPanel')"
        class="min-w-0 flex-1"
      />
      <Button
        type="button"
        variant="ghost"
        size="icon"
        class="size-8 shrink-0 rounded-lg lg:hidden"
        :aria-label="t('shell.collapseCompanionPanel')"
        :title="t('shell.collapseCompanionPanel')"
        @click="emit('collapse')"
      >
        <PanelRightClose class="size-4" aria-hidden="true" />
      </Button>
    </div>

    <div class="shell-panel-body min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <template v-for="tab in SHELL_PANEL_TABS" :key="tab.id">
        <section
          :id="`shell-panel-body-${tab.id}`"
          v-show="props.activeTab === tab.id"
          :class="['h-full', props.activeTab === tab.id ? 'shell-tab-enter' : undefined]"
          role="tabpanel"
          :aria-label="t(tab.labelKey)"
          :aria-hidden="props.activeTab !== tab.id"
          :inert="props.activeTab !== tab.id"
        >
          <slot :name="tab.id" />
        </section>
      </template>
    </div>
  </aside>
</template>
