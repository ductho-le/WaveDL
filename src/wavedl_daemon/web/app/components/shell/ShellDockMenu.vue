<script setup lang="ts">
import { motion } from 'motion-v';
import { connectionStatusKey } from '../../constants/shellConnection';
import { Ellipsis, X } from '@lucide/vue';

import type { ShellDockDestination, ShellDockDestinationId } from '../../constants/shellDock';

interface Props {
  items: readonly ShellDockDestination[];
  activeId?: ShellDockDestinationId;
}

const props = defineProps<Props>();
const open = defineModel<boolean>('open', { default: false });

const emit = defineEmits<{
  select: [item: ShellDockDestination];
  command: [];
}>();

const { t } = useI18n({ useScope: 'global' });
const { state, connection } = useWorkstation();
const statusLabel = computed(() => t(connectionStatusKey[state.value.status]));
const contentMotion = useMenuContentMotion();
const itemMotion = useMenuItemMotion();

function selectItem(item: ShellDockDestination) {
  open.value = false;
  emit('select', item);
}
</script>

<template>
  <Dialog v-model:open="open">
    <DialogTrigger as-child>
      <Button
        type="button"
        variant="ghost"
        class="shell-dock-menu-trigger"
        :aria-label="t('shell.openNavigation')"
        data-testid="shell-dock-menu-trigger"
      >
        <Ellipsis aria-hidden="true" />
        <span class="shell-dock-action-label">{{ t('shell.menu') }}</span>
      </Button>
    </DialogTrigger>

    <DialogContent
      class="shell-dock-popover w-[calc(100vw-1.5rem)] max-w-md gap-0 overflow-hidden p-0 sm:max-w-md"
      :show-close-button="false"
      data-testid="shell-dock-menu"
    >
      <motion.div
        class="shell-dock-popover-body"
        :initial="contentMotion.initial.value"
        :animate="contentMotion.animate.value"
        :transition="contentMotion.transition.value"
      >
        <DialogHeader class="shell-dock-popover-header">
          <div class="min-w-0 text-left">
            <p class="shell-dock-popover-kicker">{{ t('shell.workstation') }}</p>
            <DialogTitle>{{ t('shell.allDestinations') }}</DialogTitle>
            <DialogDescription>{{ t('shell.navigationDescription') }}</DialogDescription>
          </div>

          <DialogClose as-child>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              class="shell-dock-menu-close rounded-full"
              :aria-label="t('shell.closeNavigation')"
            >
              <X aria-hidden="true" />
            </Button>
          </DialogClose>
        </DialogHeader>

        <ScrollArea class="shell-dock-popover-scroll">
          <nav class="shell-dock-menu-grid" :aria-label="t('shell.allDestinations')">
            <motion.div
              v-for="item in props.items"
              :key="item.id"
              class="min-w-0"
              :initial="itemMotion.initial.value"
              :animate="itemMotion.animate.value"
              :transition="itemMotion.transition.value"
            >
              <ShellDockAction
                :item="item"
                :active="item.id === props.activeId"
                appearance="menu"
                @select="selectItem"
              />
            </motion.div>
          </nav>
        </ScrollArea>
        <div class="flex flex-wrap items-center justify-between gap-2 px-4 pb-4">
          <p class="flex items-center gap-2 text-xs text-muted-foreground">
            <span class="shell-status-dot" :data-status="state.status" aria-hidden="true" />{{
              statusLabel
            }}
          </p>
          <Button
            variant="ghost"
            size="sm"
            @click="
              open = false;
              emit('command');
            "
          >
            {{ t('shell.commandPalette') }}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            :disabled="state.status === 'disconnected'"
            @click="
              open = false;
              connection.detach();
            "
          >
            {{ t('shell.signOut') }}
          </Button>
        </div>
      </motion.div>
    </DialogContent>
  </Dialog>
</template>
