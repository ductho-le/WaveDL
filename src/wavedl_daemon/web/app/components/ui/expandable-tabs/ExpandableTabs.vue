<script setup lang="ts">
import type { Component, HTMLAttributes } from 'vue';
import { computed, shallowRef, useTemplateRef, watch } from 'vue';
import { onClickOutside } from '@vueuse/core';
import { AnimatePresence, motion, useReducedMotion } from 'motion-v';

import { cn } from '@/lib/utils';

export interface ExpandableTab {
  id: string;
  title: string;
  icon: Component;
  type?: never;
  disabled?: boolean;
}

export interface ExpandableTabSeparator {
  type: 'separator';
  id: string;
}

export type ExpandableTabItem = ExpandableTab | ExpandableTabSeparator;

const props = withDefaults(
  defineProps<{
    tabs: readonly ExpandableTabItem[];
    class?: HTMLAttributes['class'];
    activeClass?: HTMLAttributes['class'];
    ariaLabel?: string;
  }>(),
  {
    activeClass: '',
    ariaLabel: undefined,
  },
);

const emit = defineEmits<{
  select: [index: number];
  collapse: [];
}>();

const selectedIndex = defineModel<number | null>({ default: null });
const expandedIndex = shallowRef<number | null>(selectedIndex.value);
const rootRef = useTemplateRef<HTMLElement>('root');
const reducedMotion = useReducedMotion();

const labelTransition = computed(() =>
  reducedMotion.value
    ? { duration: 0 }
    : { delay: 0.1, type: 'spring', bounce: 0, duration: 0.6 },
);

watch(selectedIndex, (index) => {
  expandedIndex.value = index;
});

watch(
  () => props.tabs,
  (tabs) => {
    const index = selectedIndex.value;
    if (index === null) return;

    const selectedTab = tabs[index];
    if (!selectedTab || selectedTab.type === 'separator') selectedIndex.value = null;
  },
);

function isSeparator(tab: ExpandableTabItem): tab is ExpandableTabSeparator {
  return tab.type === 'separator';
}

function selectTab(index: number, tab: ExpandableTab) {
  if (tab.disabled) return;

  selectedIndex.value = index;
  expandedIndex.value = index;
  emit('select', index);
}

function collapseLabels() {
  if (expandedIndex.value === null) return;

  expandedIndex.value = null;
  emit('collapse');
}

onClickOutside(rootRef, collapseLabels);
</script>

<template>
  <div
    ref="root"
    role="toolbar"
    :aria-label="props.ariaLabel"
    :class="
      cn(
        'flex flex-wrap items-center gap-1 rounded-2xl border bg-background p-1 shadow-sm',
        props.class,
      )
    "
  >
    <template v-for="(tab, index) in props.tabs" :key="tab.id">
      <Separator
        v-if="isSeparator(tab)"
        orientation="vertical"
        class="mx-1 h-6"
        decorative
      />
      <Button
        v-else
        type="button"
        variant="ghost"
        size="sm"
        :disabled="tab.disabled"
        :class="
          cn(
            'rounded-xl py-2 transition-[background-color,color,gap,padding] duration-300',
            expandedIndex === index ? 'gap-2 px-4' : 'gap-0 px-2',
            selectedIndex === index
              ? cn('bg-muted', props.activeClass)
              : 'text-muted-foreground hover:bg-muted hover:text-foreground',
          )
        "
        :aria-label="tab.title"
        :aria-pressed="selectedIndex === index"
        :title="tab.title"
        @click="selectTab(index, tab)"
      >
        <component :is="tab.icon" class="size-4" />
        <AnimatePresence :initial="false">
          <motion.span
            v-if="expandedIndex === index"
            class="overflow-hidden whitespace-nowrap"
            :initial="{ width: 0, opacity: 0 }"
            :animate="{ width: 'auto', opacity: 1 }"
            :exit="{ width: 0, opacity: 0 }"
            :transition="labelTransition"
          >
            {{ tab.title }}
          </motion.span>
        </AnimatePresence>
      </Button>
    </template>
  </div>
</template>
