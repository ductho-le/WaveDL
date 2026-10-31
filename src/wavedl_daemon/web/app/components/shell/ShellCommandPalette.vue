<script setup lang="ts">
import { X } from '@lucide/vue';
import {
  SHELL_ACTION_NAVIGATION,
  SHELL_PRIMARY_NAVIGATION,
  type ShellNavigationDescriptor,
} from '../../shell/sections';
import type { WorkstationSection } from '../../lib/shell-test-ids';

const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ navigate: [section: WorkstationSection] }>();
const { t } = useI18n({ useScope: 'global' });
const query = shallowRef('');
const selected = shallowRef(0);
let opener: HTMLElement | null = null;

const allItems: readonly ShellNavigationDescriptor[] = [
  ...SHELL_PRIMARY_NAVIGATION,
  ...SHELL_ACTION_NAVIGATION,
];

const filtered = computed(() => {
  const term = query.value.trim().toLocaleLowerCase();

  if (!term) return allItems;

  return allItems.filter((item) => t(item.labelKey).toLocaleLowerCase().includes(term));
});

const activeOptionId = computed(() => {
  const item = filtered.value[selected.value];

  return item ? `shell-command-option-${item.id}` : undefined;
});

function choose(): void {
  const item = filtered.value[selected.value];

  if (!item) return;
  emit('navigate', item.id);
  open.value = false;
}

function keydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') {
    open.value = false;

    return;
  }

  if (event.key === 'ArrowDown') {
    event.preventDefault();
    selected.value = Math.min(selected.value + 1, Math.max(0, filtered.value.length - 1));
  } else if (event.key === 'ArrowUp') {
    event.preventDefault();
    selected.value = Math.max(selected.value - 1, 0);
  } else if (event.key === 'Enter') {
    event.preventDefault();
    choose();
  }
}

watch(filtered, () => {
  selected.value = Math.min(selected.value, Math.max(0, filtered.value.length - 1));
});

watch(open, (value) => {
  if (value) {
    opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    query.value = '';
    selected.value = 0;
    void nextTick(() => document.getElementById('shell-command-input')?.focus());
  }
});

function restoreFocus(event: Event) {
  event.preventDefault();

  if (
    opener?.isConnected &&
    !document.querySelector('[role="dialog"]:not([data-testid="shell-command-palette"])')
  )
    opener.focus();
}
</script>

<template>
  <Dialog v-model:open="open">
    <DialogContent
      class="shell-command-palette"
      :data-testid="'shell-command-palette'"
      :show-close-button="false"
      @close-auto-focus="restoreFocus"
    >
      <DialogTitle class="sr-only">{{ t('shell.commandPalette') }}</DialogTitle>
      <DialogClose as-child
        ><Button
          class="shell-overlay-close"
          variant="ghost"
          size="icon"
          :aria-label="t('shell.closeOverlay')"
          ><X aria-hidden="true" /></Button
      ></DialogClose>
      <DialogDescription class="sr-only">{{
        t('shell.commandPaletteNavigation')
      }}</DialogDescription>
      <Input
        id="shell-command-input"
        v-model="query"
        type="search"
        role="combobox"
        :aria-label="t('shell.commandPalette')"
        :aria-expanded="open"
        :aria-activedescendant="activeOptionId"
        aria-autocomplete="list"
        aria-controls="shell-command-results"
        :placeholder="t('shell.commandPalettePlaceholder')"
        @keydown="keydown"
      />
      <div id="shell-command-results" role="listbox" class="shell-command-results">
        <Button
          v-for="(item, index) in filtered"
          :key="item.id"
          :id="`shell-command-option-${item.id}`"
          type="button"
          variant="ghost"
          class="shell-command-item"
          role="option"
          :aria-selected="index === selected"
          @mousemove="selected = index"
          @click="
            emit('navigate', item.id);
            open = false;
          "
        >
          <component :is="item.icon" class="size-4" aria-hidden="true" />
          {{ t(item.labelKey) }}
        </Button>
        <p v-if="filtered.length === 0" class="shell-command-empty">
          {{ t('shell.commandPaletteNoResults') }}
        </p>
      </div>
    </DialogContent>
  </Dialog>
</template>
