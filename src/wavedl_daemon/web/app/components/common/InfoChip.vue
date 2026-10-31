<script setup lang="ts">
import { CircleHelp } from '@lucide/vue';

defineProps<{ label: string; text: string; uncertain?: boolean }>();

const open = shallowRef(false);
</script>

<template>
  <TooltipProvider>
    <Tooltip :open="open" @update:open="open = $event">
      <TooltipTrigger as-child>
        <button
          type="button"
          class="inline-flex w-fit max-w-full items-center gap-2 whitespace-normal break-words rounded-full border px-3 py-1 text-left text-xs text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
          :class="{ 'border-dashed': uncertain }"
          :aria-label="`${label}: ${text}`"
          @click="open = !open"
        >
          <CircleHelp class="size-3.5" aria-hidden="true" />{{ label }}
        </button>
      </TooltipTrigger>
      <TooltipContent class="max-w-[min(24rem,calc(100vw-2rem))] break-words"
        ><p>{{ text }}</p></TooltipContent
      >
    </Tooltip>
    <span class="sr-only">{{ text }}</span>
  </TooltipProvider>
</template>
