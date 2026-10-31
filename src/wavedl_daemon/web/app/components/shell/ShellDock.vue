<script setup lang="ts">
import { motion } from 'motion-v';

import {
  getShellDockDestination,
  SHELL_DOCK_DESTINATIONS,
  type ShellDockDestination,
} from '../../constants/shellDock';
import { SPRING_DEFAULT } from '@/lib/animation';

import type { WorkstationSection } from '../../shell/sections';
import type { ConnectionStatus } from '../../lib/connection-session';

const props = defineProps<{ section: WorkstationSection; status: ConnectionStatus }>();
const emit = defineEmits<{ navigate: [section: WorkstationSection]; command: [] }>();
const { t } = useI18n({ useScope: 'global' });
const shellPreferences = useShellPreferencesStore();
const reducedMotion = useReducedMotion();
const isMenuOpen = shallowRef(false);

const primaryItems = computed(() => {
  return shellPreferences.dockSlots.slice(0, 3).flatMap((id) => {
    const destination = getShellDockDestination(id);

    if (!destination) return [];

    return [destination];
  });
});

const availableItems = computed(() => [...SHELL_DOCK_DESTINATIONS]);

/*
 * All five saved slots still carry meaning after the compact dock shrinks to
 * three controls: the full configured set leads the menu, followed by every
 * other destination this account can open.
 */
const menuItems = computed(() => {
  const availableById = new Map(
    availableItems.value.map((destination) => [destination.id, destination]),
  );

  const configured = shellPreferences.dockSlots.flatMap((id) => {
    const destination = availableById.get(id);

    return destination ? [destination] : [];
  });

  const configuredIds = new Set(configured.map((destination) => destination.id));

  return [
    ...configured,
    ...availableItems.value.filter((destination) => !configuredIds.has(destination.id)),
  ];
});

function isActive(item: ShellDockDestination) {
  return item.id === props.section;
}

const activeItemId = computed(() => menuItems.value.find((item) => isActive(item))?.id);

const dockInitial = computed(() =>
  reducedMotion.value ? { opacity: 1 } : { opacity: 0, scale: 0.97, y: 14 },
);

const dockTransition = computed(() => (reducedMotion.value ? { duration: 0 } : SPRING_DEFAULT));

function selectItem(item: ShellDockDestination) {
  emit('navigate', item.id);
}
</script>

<template>
  <nav class="shell-dock shrink-0 px-3 pt-2 md:hidden" :aria-label="t('shell.sections')">
    <motion.div
      class="shell-mobile-dock-frame mx-auto flex max-w-lg items-stretch gap-2"
      :style="{ paddingBottom: 'max(0.45rem, env(safe-area-inset-bottom))' }"
      :initial="dockInitial"
      :animate="{ opacity: 1, scale: 1, y: 0 }"
      :transition="dockTransition"
    >
      <div
        class="shell-mobile-dock min-w-0 flex-1"
        data-shell-dock-primary
        :aria-label="t('shell.pinnedDestinations')"
      >
        <ShellDockAction
          v-for="item in primaryItems"
          :key="item.id"
          :item="item"
          :active="isActive(item)"
          @select="selectItem"
        />
      </div>

      <ShellDockMenu
        v-model:open="isMenuOpen"
        :items="menuItems"
        :active-id="activeItemId"
        @select="selectItem"
        @command="emit('command')"
      />
    </motion.div>
  </nav>
</template>
