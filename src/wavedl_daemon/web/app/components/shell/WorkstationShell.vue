<script setup lang="ts">
import { motion } from 'motion-v';
import { SHELL_RAIL_WIDTH, SHELL_RAIL_GUTTER } from '../../constants/shellRail';
import { X } from '@lucide/vue';
import type { ConnectionStatus } from '../../lib/connection-session';
import {
  sectionForPath,
  sectionPath,
  SHELL_ACTION_NAVIGATION,
  SHELL_PRIMARY_NAVIGATION,
} from '../../shell/sections';
import { type WorkstationSection } from '../../lib/shell-test-ids';

type Overlay = 'adopt';

const props = defineProps<{
  status: ConnectionStatus;
  section: WorkstationSection;
  overlay: Overlay | null;
}>();

const emit = defineEmits<{
  navigate: [section: WorkstationSection];
  'update:overlay': [overlay: Overlay | null];
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

let overlayOpener: HTMLElement | null = null;
let focusObserver: MutationObserver | null = null;
const overlayTitle = computed(() => t('shell.adoptStandaloneRun'));

function setOverlay(value: Overlay | null) {
  if (
    value &&
    !props.overlay &&
    import.meta.client &&
    document.activeElement instanceof HTMLElement
  )
    overlayOpener = document.activeElement;
  emit('update:overlay', value);
}

function restoreOverlayFocus(event: Event) {
  focusObserver?.disconnect();
  event.preventDefault();

  if (overlayOpener?.isConnected) overlayOpener.focus();
}

function focusOverlay(event: Event) {
  event.preventDefault();
  void nextTick(() => {
    const body = document.querySelector<HTMLElement>('.shell-overlay-body');

    if (!body) return;

    const focusFirst = () => {
      const control =
        body.querySelector<HTMLElement>('[role="combobox"]') ??
        body.querySelector<HTMLElement>(
          'input:not([disabled]), button:not([disabled]), [tabindex="0"]',
        );

      if (!control) return false;
      control.focus();
      focusObserver?.disconnect();

      return true;
    };

    if (focusFirst()) return;
    body.focus();
    focusObserver = new MutationObserver(focusFirst);
    focusObserver.observe(body, { childList: true, subtree: true });
  });
}

async function navigate(section: WorkstationSection) {
  paletteOpen.value = false;
  const destination = SHELL_PRIMARY_NAVIGATION.find((item) => item.id === section);

  if (destination) {
    setOverlay(null);

    if (navigateTo) navigateTo(sectionPath(destination.id));
    else emit('navigate', destination.id);

    return;
  }

  const action = SHELL_ACTION_NAVIGATION.find((item) => item.id === section);

  if (!action) return;

  if (action.id === 'adopt') {
    // props.section lags behind the router, so trust where navigation landed.
    if (props.section !== 'experiments' && navigateTo) {
      if (sectionForPath(await navigateTo('/experiments', false)) !== 'experiments') return;
    }

    setOverlay('adopt');

    return;
  }

  setOverlay(null);

  if (navigateTo) navigateTo(action.path);
  else emit('navigate', section);
}

function shortcut(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    if (layout.isFocusMode.value) layout.exitFocusMode();
    else if (!isDesktop.value) layout.closePanel();
  }

  if ((event.metaKey || event.ctrlKey) && event.key === '.') {
    event.preventDefault();
    layout.toggleFocusMode();
  }

  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    setOverlay(null);
    paletteOpen.value = !paletteOpen.value;
  }
}

useEventListener('keydown', shortcut);

onScopeDispose(() => focusObserver?.disconnect());
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
    <Dialog :open="props.overlay !== null" @update:open="!$event && setOverlay(null)"
      ><DialogContent
        class="shell-overlay shell-overlay-wide"
        :show-close-button="false"
        :aria-describedby="undefined"
        @open-auto-focus="focusOverlay"
        @close-auto-focus="restoreOverlayFocus"
        ><div class="shell-overlay-header">
          <DialogTitle class="sr-only">{{ overlayTitle }}</DialogTitle>
          <DialogClose as-child
            ><Button
              class="shell-overlay-close"
              variant="ghost"
              size="icon"
              :aria-label="t('shell.closeOverlay')"
              ><X aria-hidden="true" /></Button
          ></DialogClose>
        </div>
        <div class="shell-overlay-body" tabindex="-1">
          <slot v-if="props.overlay === 'adopt'" name="adopt" /></div></DialogContent
    ></Dialog>
  </div>
</template>
