<script setup lang="ts">
import { cn } from '@/lib/utils';

/** `titleHidden` keeps the accessible name for panels whose controls speak for themselves. */
defineProps<{ title: string; description?: string; titleHidden?: boolean }>();

const titleId = useId();

const projectStyles = {
  inspectorPanel: cn(
    'flex flex-col gap-6 [animation:inspector-enter_420ms_cubic-bezier(0.22,_1,_0.36,_1)]',
  ),
};
</script>
<template>
  <section :class="projectStyles.inspectorPanel" :aria-labelledby="titleId">
    <h2 v-if="titleHidden" :id="titleId" class="sr-only">{{ title }}</h2>
    <header v-else class="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
      <div class="min-w-0">
        <h2 :id="titleId" class="text-[1.45rem] font-semibold tracking-[-0.02em]">{{ title }}</h2>
        <p v-if="description" class="mt-1 max-w-[60ch] text-sm text-muted-foreground">
          {{ description }}
        </p>
      </div>
      <div v-if="$slots.actions" class="flex flex-wrap items-center gap-2">
        <slot name="actions" />
      </div>
    </header>
    <slot />
  </section>
</template>
