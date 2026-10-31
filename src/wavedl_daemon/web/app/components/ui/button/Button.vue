<script setup lang="ts">
import type { PrimitiveProps } from 'reka-ui'
import type { HTMLAttributes } from 'vue'
import type { ButtonVariants } from '.'
import { Primitive } from 'reka-ui'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'
import { buttonVariants } from '.'

interface Props extends PrimitiveProps {
  variant?: ButtonVariants['variant']
  size?: ButtonVariants['size']
  class?: HTMLAttributes['class']
  isLoading?: boolean
  loadingLabel?: string
  disabled?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  as: 'button',
  isLoading: false,
  loadingLabel: 'Loading',
  disabled: false,
})

function handleClick(event: MouseEvent) {
  if (props.disabled || props.isLoading) return
}
</script>

<template>
  <Primitive
    ref="buttonRef"
    data-slot="button"
    :as="as"
    :as-child="asChild"
    :class="cn(buttonVariants({ variant, size }), 'relative overflow-hidden', props.class)"
    :disabled="props.disabled || props.isLoading"
    @click="handleClick"
  >
    <Spinner
      v-if="isLoading"
      variant="default"
      size="sm"
      :label="loadingLabel"
      class="text-current"
    />
    <slot v-else />
  </Primitive>
</template>
