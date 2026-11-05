<script setup lang="ts">
import { motion } from 'motion-v';
import { SHELL_RAIL_WIDTH, SHELL_RAIL_GUTTER } from '../../constants/shellRail';
import type { ConnectionStatus } from '../../lib/connection-session';
import {
  sectionPath,
  SHELL_ACTION_NAVIGATION,
  SHELL_PRIMARY_NAVIGATION,
} from '../../shell/sections';
import { type WorkstationSection } from '../../lib/shell-test-ids';

const props = defineProps<{
  status: ConnectionStatus;
  section: WorkstationSection;
}>();

const emit = defineEmits<{
  navigate: [section: WorkstationSection];
}>();

const { t } = useI18n({ useScope: 'global' });
const layout = useShellLayout();
const navigateTo = useShellNavigation();
const { isRailVisible, isCompanionShowing, panelWidth, commandPaletteOpen: paletteOpen } = layout;
const reducedMotion = useReducedMotion();
const isDesktop = useMediaQuery('(min-width: 1024px)');

watch(isDesktop, layout.setDockedLayout, { immediate: true });

const companionMotion = computed(() =>
  isDesktop.value
    ? {
        opacity: isCompanionShowing.value ? 1 : 0,
        width: isCompanionShowing.value ? panelWidth.value + 12 : 0,
      }
    : { opacity: 1, width: '100%' },
);

provideShellActions((section) => {
  void navigate(section);
});

const connectionAnnouncement = computed(() =>
  props.status === 'connected'
    ? t('shell.connected')
    : props.status === 'connecting'
      ? t('shell.connecting')
      : props.status === 'offline'
        ? t('shell.offline')
        : t('shell.notConnected'),
);

function navigate(section: WorkstationSection) {
  paletteOpen.value = false;
  const destination = SHELL_PRIMARY_NAVIGATION.find((item) => item.id === section);

  if (destination) {
    if (navigateTo) navigateTo(sectionPath(destination.id));
    else emit('navigate', destination.id);

    return;
  }

  const action = SHELL_ACTION_NAVIGATION.find((item) => item.id === section);

  if (!action) return;

  if (navigateTo) navigateTo(action.path);
  else emit('navigate', section);
}

function shortcut(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    if (event.defaultPrevented) return;

    if (layout.isFocusMode.value) layout.exitFocusMode();
    else if (!isDesktop.value) layout.closePanel();
  }

  if ((event.metaKey || event.ctrlKey) && event.key === '.') {
    event.preventDefault();
    layout.toggleFocusMode();
  }

  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    paletteOpen.value = !paletteOpen.value;
  }
}

useEventListener('keydown', shortcut);
</script>
<template>
  <div data-shell class="shell-ground fixed inset-x-0 top-0 h-dvh overflow-hidden">
    <span class="sr-only" aria-live="polite" aria-atomic="true">{{ connectionAnnouncement }}</span>
    <a class="shell-skip-link" href="#workspace">{{ t('shell.skipToWorkspace') }}</a>
    <div
      data-shell-frame
      class="app-safe-frame [--safe-gutter:0.75rem] flex size-full flex-col pb-0 md:pb-3"
    >
      <div class="flex min-h-0 min-w-0 flex-1">
        <AnimatePresence>
          <motion.div
            v-if="isRailVisible"
            class="hidden h-full shrink-0 overflow-visible md:block"
            :initial="{ opacity: 0, width: 0 }"
            :animate="{ opacity: 1, width: SHELL_RAIL_WIDTH + SHELL_RAIL_GUTTER }"
            :exit="{ opacity: 0, width: 0 }"
            :transition="{ duration: reducedMotion ? 0 : 0.26, ease: [0.16, 1, 0.3, 1] }"
          >
            <div class="h-full pr-3">
              <ShellRail
                :section="section"
                :status="status"
                @navigate="navigate"
                @command="paletteOpen = true"
              />
            </div>
          </motion.div>
        </AnimatePresence>
        <div class="flex min-h-0 min-w-0 flex-1 flex-col"><slot /></div>
        <div
          v-show="!isDesktop && isCompanionShowing"
          class="fixed inset-0 z-40 bg-neutral-950/20 backdrop-blur-xs dark:bg-neutral-950/45"
          aria-hidden="true"
          @pointerdown="layout.closePanel()"
        />
        <motion.div
          class="h-full shrink-0 max-lg:fixed max-lg:inset-y-0 max-lg:right-0 max-lg:z-50 max-lg:w-full! max-lg:max-w-sm max-lg:transition-transform max-lg:duration-300 motion-reduce:transition-none sm:max-lg:max-w-md"
          :class="[
            isCompanionShowing ? 'overflow-visible' : 'pointer-events-none overflow-hidden',
            layout.isPanelSheetVisible.value ? 'max-lg:translate-x-0' : 'max-lg:translate-x-full',
          ]"
          :inert="!isCompanionShowing"
          :aria-hidden="!isCompanionShowing"
          :initial="false"
          :animate="companionMotion"
          :transition="{ duration: reducedMotion ? 0 : 0.26, ease: [0.16, 1, 0.3, 1] }"
        >
          <div class="relative h-full max-lg:pl-0 lg:pl-3"><ShellCompanion /></div>
        </motion.div>
      </div>
      <ShellDock
        v-if="isRailVisible"
        data-testid="workstation-nav"
        :section="section"
        :status="status"
        @navigate="navigate"
        @command="paletteOpen = true"
      />
    </div>
    <ShellCommandPalette v-model:open="paletteOpen" @navigate="navigate" />
  </div>
</template>
