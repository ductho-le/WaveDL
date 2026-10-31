<script setup lang="ts">
import { Info } from '@lucide/vue';

defineProps<{ label: string; text?: string }>();
</script>

<template>
  <Popover>
    <PopoverTrigger as-child>
      <Button type="button" variant="ghost" size="icon" class="size-7 shrink-0" :aria-label="label">
        <Info aria-hidden="true" />
      </Button>
    </PopoverTrigger>
    <PopoverContent :aria-label="label" align="start" class="max-w-sm text-sm leading-relaxed">
      <p v-if="text">{{ text }}</p>
      <slot v-else />
    </PopoverContent>
  </Popover>
</template>
