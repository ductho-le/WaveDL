<script setup lang="ts">
import { motion } from 'motion-v';
import { Activity, Search, Settings2, Unplug } from '@lucide/vue';
import { SHELL_PRIMARY_NAVIGATION, type WorkstationSection } from '../../shell/sections';
import type { ConnectionStatus } from '../../lib/connection-session';
import { connectionStatusKey } from '../../constants/shellConnection';

const props = defineProps<{ section: WorkstationSection; status: ConnectionStatus }>();
const emit = defineEmits<{ navigate: [section: WorkstationSection]; command: [] }>();
const { t } = useI18n({ useScope: 'global' });
const { connection } = useWorkstation();
const reducedMotion = useReducedMotion();
const collisionPadding = useSafeAreaCollisionPadding();
const statusLabel = computed(() => t(connectionStatusKey[props.status]));
const DOCK_BASE_SIZE = 36; // `size-9`, the size every rail control rests at
const DOCK_MAGNIFIED_SIZE = 64;
const DOCK_FALLOFF = 120; // How far along the rail the swell reaches from the cursor

/**
 * Doing two jobs at once: it is the icons' resting inset from the pill's left
 * edge — which is what anchors them there — and the slack their hover glow needs
 * to spill into before the track clips it.
 */
const DOCK_PADDING = 10;

/**
 * The scrolling track is sized for the largest icon and pinned to the pill's left
 * edge, so it hangs off the right only. It is the one box in the rail that clips
 * — an `overflow-y` of anything but `visible` clips X too — so a track cut to the
 * pill's width would shear the flank off every magnified icon.
 */
const DOCK_TRACK_WIDTH = DOCK_MAGNIFIED_SIZE + DOCK_PADDING * 2;
const DOCK_SPRING = { mass: 0.1, stiffness: 150, damping: 12 };

const pointerY = useMotionValue(Number.POSITIVE_INFINITY);

/**
 * Mouse only: a touch "hover" has no matching leave, so the dock would stay
 * swollen after the tap. Reduced-motion users keep the flat rail they asked for.
 */
function trackDockPointer(event: PointerEvent) {
  if (reducedMotion.value || event.pointerType !== 'mouse') return;

  pointerY.set(event.clientY);
}

/**
 * Keyboard focus gets the same swell as the cursor, so tabbing through the rail
 * reads as the same control and not as a dock that only exists for mouse users.
 */
function trackDockFocus(event: FocusEvent) {
  if (reducedMotion.value || !(event.target instanceof HTMLElement)) return;

  const rect = event.target.getBoundingClientRect();
  pointerY.set(rect.y + rect.height / 2);
}

function releaseDock() {
  pointerY.set(Number.POSITIVE_INFINITY);
}

function onNavigationKeydown(event: KeyboardEvent): void {
  if (!(event.currentTarget instanceof HTMLElement)) return;

  if (!(document.activeElement instanceof HTMLElement)) return;

  const controls = Array.from(
    event.currentTarget.querySelectorAll<HTMLButtonElement>('[data-testid^="nav-"]'),
  );

  const currentIndex = controls.findIndex((control) => control === document.activeElement);

  if (currentIndex < 0 || controls.length === 0) return;

  let nextIndex: number;

  if (event.key === 'ArrowDown') nextIndex = (currentIndex + 1) % controls.length;
  else if (event.key === 'ArrowUp')
    nextIndex = (currentIndex - 1 + controls.length) % controls.length;
  else if (event.key === 'Home') nextIndex = 0;
  else if (event.key === 'End') nextIndex = controls.length - 1;
  else return;

  event.preventDefault();
  controls[nextIndex]?.focus();
}
</script>
<template>
  <TooltipProvider :delay-duration="280">
    <nav
      class="shell-rail flex h-full shrink-0 flex-col gap-3 py-0.5 w-14"
      data-shell-rail-mode="dock"
      :aria-label="t('shell.sections')"
      data-testid="workstation-nav"
      @keydown="onNavigationKeydown"
    >
      <motion.div
        class="shell-rail-island flex shrink-0 flex-col items-center gap-0.5 py-2"
        role="group"
        :aria-label="t('shell.utilities')"
        :initial="{ opacity: 0, y: reducedMotion ? 0 : -8 }"
        :animate="{ opacity: 1, y: 0 }"
        :transition="{ duration: reducedMotion ? 0 : 0.24, ease: [0.16, 1, 0.3, 1] }"
      >
        <Tooltip>
          <TooltipTrigger as-child>
            <Button
              variant="ghost"
              size="icon"
              class="shell-rail-control rounded-full"
              :aria-label="t('shell.workstation')"
              @click="emit('navigate', 'experiments')"
              ><img src="/favicon.svg" alt="" class="size-7 rounded-[0.5rem]"
            /></Button>
          </TooltipTrigger>
          <TooltipContent side="right" :side-offset="10">{{
            t('shell.workstation')
          }}</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger as-child>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              class="shell-rail-control rounded-full"
              :aria-label="t('shell.commandPalette')"
              @click="emit('command')"
            >
              <Search />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right" :side-offset="10">{{
            t('shell.commandPalette')
          }}</TooltipContent>
        </Tooltip>
      </motion.div>

      <div class="flex min-h-0 flex-1 items-center">
        <motion.div
          class="shell-rail-island flex max-h-full w-full flex-col items-start py-2"
          role="group"
          :aria-label="t('shell.sections')"
          :initial="{ opacity: 0, y: reducedMotion ? 0 : 8 }"
          :animate="{ opacity: 1, y: 0 }"
          :transition="{
            duration: reducedMotion ? 0 : 0.24,
            delay: reducedMotion ? 0 : 0.04,
            ease: [0.16, 1, 0.3, 1],
          }"
          @pointermove="trackDockPointer"
          @pointerleave="releaseDock"
          @focusin="trackDockFocus"
          @focusout="releaseDock"
        >
          <!--
            Wider than the pill but flush with its left edge, so the overhang is
            all on the right. `items-start` then pins every icon to its resting
            inset: a magnified one keeps its left edge exactly where it sat and
            grows out over the gutter, the way icons lift off a Dock pinned to
            the left of the screen rather than swelling around their own centre.
          -->
          <div
            class="shell-hide-scrollbar flex max-h-full flex-col items-start overflow-y-auto px-2.5"
            :style="{ width: `${DOCK_TRACK_WIDTH}px` }"
          >
            <div class="flex w-full flex-col items-start gap-0.5">
              <ShellRailDockItem
                v-for="item in SHELL_PRIMARY_NAVIGATION"
                :key="item.id"
                :item="item"
                :active="props.section === item.id"
                :pointer-y="pointerY"
                :base-size="DOCK_BASE_SIZE"
                :magnified-size="reducedMotion ? DOCK_BASE_SIZE : DOCK_MAGNIFIED_SIZE"
                :falloff="DOCK_FALLOFF"
                :spring="DOCK_SPRING"
                @select="emit('navigate', $event.id)"
              />
            </div>
          </div>
        </motion.div>
      </div>

      <motion.div
        class="shell-rail-island flex shrink-0 flex-col items-center gap-0.5 py-2"
        role="group"
        :aria-label="t('shell.connection')"
        :initial="{ opacity: 0, y: reducedMotion ? 0 : 8 }"
        :animate="{ opacity: 1, y: 0 }"
        :transition="{
          duration: reducedMotion ? 0 : 0.24,
          delay: reducedMotion ? 0 : 0.08,
          ease: [0.16, 1, 0.3, 1],
        }"
      >
        <!--
          No tooltip on this one: the menu it opens already names the account,
          and a tooltip and a dropdown sharing one trigger fight over focus.
        -->
        <DropdownMenu>
          <DropdownMenuTrigger as-child>
            <Button
              type="button"
              class="shell-rail-account rounded-full"
              variant="ghost"
              size="icon"
              :aria-label="`${t('shell.connectionStatus')}: ${statusLabel}`"
            >
              <span class="shell-status-dot" :data-status="props.status" aria-hidden="true" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            class="min-w-56 rounded-lg"
            side="right"
            align="end"
            :side-offset="12"
            :collision-padding="collisionPadding"
          >
            <DropdownMenuLabel>{{ statusLabel }}</DropdownMenuLabel>
            <DropdownMenuGroup>
              <DropdownMenuItem @select="emit('navigate', 'connection')"
                ><Activity aria-hidden="true" />{{ t('shell.connection') }}</DropdownMenuItem
              >
              <DropdownMenuItem @select="emit('navigate', 'settings')"
                ><Settings2 aria-hidden="true" />{{ t('shell.settings') }}</DropdownMenuItem
              >
              <DropdownMenuItem
                :disabled="props.status === 'disconnected'"
                @select="connection.detach()"
                ><Unplug aria-hidden="true" />{{ t('shell.signOut') }}</DropdownMenuItem
              >
            </DropdownMenuGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </motion.div>
    </nav>
  </TooltipProvider>
</template>
