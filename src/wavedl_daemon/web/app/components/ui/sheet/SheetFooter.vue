<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { inject } from 'vue'
import { cn } from '@/lib/utils'
import { SHEET_SIDE_KEY } from './keys'

const props = defineProps<{ class?: HTMLAttributes['class'] }>()

const side = inject(SHEET_SIDE_KEY, 'right')

const hasBottomInset = computed(() => side === 'right' || side === 'left' || side === 'bottom')
</script>

<template>
  <div
    data-slot="sheet-footer"
    :class="cn(
      'mt-auto flex flex-col gap-2 p-4',
      hasBottomInset && '-mb-[env(safe-area-inset-bottom,0px)] pb-[calc(1rem+env(safe-area-inset-bottom,0px))]',
      props.class,
    )"
  >
    <slot />
  </div>
</template>
