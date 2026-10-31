<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { inject } from 'vue'
import { cn } from '@/lib/utils'
import { SHEET_SIDE_KEY } from './keys'

const props = defineProps<{ class?: HTMLAttributes['class'] }>()

const side = inject(SHEET_SIDE_KEY, 'right')

const hasTopInset = computed(() => side === 'right' || side === 'left' || side === 'top')
</script>

<template>
  <div
    data-slot="sheet-header"
    :class="cn(
      'flex flex-col gap-1.5 p-4',
      hasTopInset && '-mt-[env(safe-area-inset-top,0px)] pt-[calc(1rem+env(safe-area-inset-top,0px))]',
      props.class,
    )"
  >
    <slot />
  </div>
</template>
