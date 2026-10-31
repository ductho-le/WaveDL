<script setup lang="ts">
import { computed } from 'vue'
import type { HTMLAttributes } from 'vue'

import { cn } from '@/lib/utils'

import type { SpinnerSize, SpinnerVariant } from './types'

const props = withDefaults(defineProps<{
  variant?: SpinnerVariant
  size?: SpinnerSize
  label?: string
  text?: string
  class?: HTMLAttributes['class']
}>(), {
  variant: 'default',
  size: 'md',
  label: 'Loading',
  text: 'Loading',
})

const rootClass = computed(() => cn(
  'inline-flex items-center justify-center text-muted-foreground',
  props.variant === 'terminal' && 'font-mono text-sm font-semibold',
  props.class,
))

const spinnerSizeClass = computed(() => {
  if (props.size === 'sm') return 'h-4 w-4'
  if (props.size === 'lg') return 'h-8 w-8'
  return 'h-6 w-6'
})

const ringInnerSizeClass = computed(() => {
  if (props.size === 'sm') return 'h-4 w-4'
  if (props.size === 'lg') return 'h-8 w-8'
  return 'h-6 w-6'
})

const pulseDotSizeClass = computed(() => {
  if (props.size === 'sm') return 'h-2 w-2'
  if (props.size === 'lg') return 'h-3.5 w-3.5'
  return 'h-3 w-3'
})

const dotSizeClass = computed(() => {
  if (props.size === 'sm') return 'h-1.5 w-1.5'
  if (props.size === 'lg') return 'h-2.5 w-2.5'
  return 'h-2 w-2'
})

const scaleClass = computed(() => {
  if (props.size === 'sm') return 'scale-90'
  if (props.size === 'lg') return 'scale-110'
  return 'scale-100'
})
</script>

<template>
  <span
    role="status"
    :aria-label="label"
    :class="rootClass"
  >
    <span
      v-if="variant === 'default'"
      :class="cn('inline-block animate-spin rounded-[999px] border-[3px] border-current border-t-transparent', spinnerSizeClass)"
      aria-hidden="true"
    />

    <svg
      v-else-if="variant === 'svg'"
      :class="cn('shrink-0 animate-spin text-current', spinnerSizeClass)"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M24 12C24 18.6274 18.6274 24 12 24C5.37258 24 0 18.6274 0 12C0 5.37258 5.37258 0 12 0C18.6274 0 24 5.37258 24 12ZM3.13375 12C3.13375 16.8967 7.10331 20.8662 12 20.8662C16.8967 20.8662 20.8662 16.8967 20.8662 12C20.8662 7.10331 16.8967 3.13375 12 3.13375C7.10331 3.13375 3.13375 7.10331 3.13375 12Z"
        fill="currentColor"
        opacity=".2"
      />
      <path
        d="M12 0C9.62662 -2.83022e-08 7.30655 0.703788 5.33316 2.02236C3.35977 3.34094 1.8217 5.21509 0.913446 7.4078C0.00519403 9.60051 -0.232446 12.0133 0.230577 14.3411C0.693599 16.6689 1.83649 18.8071 3.51472 20.4853L5.73062 18.2694C4.49065 17.0294 3.64622 15.4496 3.30412 13.7297C2.96201 12.0098 3.13759 10.2271 3.80866 8.60703C4.47972 6.98694 5.61613 5.60222 7.07418 4.62798C8.53222 3.65375 10.2464 3.13375 12 3.13375L12 0Z"
        fill="currentColor"
      />
    </svg>

    <svg
      v-else-if="variant === 'throbber'"
      :class="cn('shrink-0 animate-spin text-current', spinnerSizeClass)"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <g stroke="currentColor" stroke-linecap="round" stroke-width="2">
        <path d="M12 2.75V5.25" opacity="0.15" />
        <path d="M16.95 4.08L15.7 6.25" opacity="0.22" />
        <path d="M19.92 7.05L17.75 8.3" opacity="0.32" />
        <path d="M21.25 12H18.75" opacity="0.42" />
        <path d="M19.92 16.95L17.75 15.7" opacity="0.54" />
        <path d="M16.95 19.92L15.7 17.75" opacity="0.66" />
        <path d="M12 21.25V18.75" opacity="0.78" />
        <path d="M7.05 19.92L8.3 17.75" opacity="0.9" />
        <path d="M4.08 16.95L6.25 15.7" opacity="1" />
        <path d="M2.75 12H5.25" opacity="0.86" />
        <path d="M4.08 7.05L6.25 8.3" opacity="0.7" />
        <path d="M7.05 4.08L8.3 6.25" opacity="0.5" />
      </g>
    </svg>

    <span
      v-else-if="variant === 'ring'"
      :class="cn('relative flex items-center justify-center', spinnerSizeClass)"
      aria-hidden="true"
    >
      <span :class="cn('absolute inline-flex animate-ping rounded-full border border-current/50', ringInnerSizeClass)" />
      <span :class="cn('inline-flex rounded-full border border-current', ringInnerSizeClass)" />
    </span>

    <span
      v-else-if="variant === 'pulse-dot'"
      :class="cn('inline-flex shrink-0 origin-center rounded-full bg-current [animation:spinner-pulse-dot_1s_ease-in-out_infinite]', pulseDotSizeClass)"
      aria-hidden="true"
    />

    <span
      v-else-if="variant === 'wave'"
      :class="cn('flex items-center justify-center gap-0.5', scaleClass)"
      aria-hidden="true"
    >
      <span class="h-2.5 w-1 origin-center rounded-full bg-current [animation:spinner-wave_0.9s_ease-in-out_infinite_0s]" />
      <span class="h-3.5 w-1 origin-center rounded-full bg-current [animation:spinner-wave_0.9s_ease-in-out_infinite_0.12s]" />
      <span class="h-[1.125rem] w-1 origin-center rounded-full bg-current [animation:spinner-wave_0.9s_ease-in-out_infinite_0.24s]" />
      <span class="h-3.5 w-1 origin-center rounded-full bg-current [animation:spinner-wave_0.9s_ease-in-out_infinite_0.36s]" />
      <span class="h-2.5 w-1 origin-center rounded-full bg-current [animation:spinner-wave_0.9s_ease-in-out_infinite_0.48s]" />
    </span>

    <span
      v-else-if="variant === 'bars'"
      :class="cn('flex items-center justify-center gap-1', scaleClass)"
      aria-hidden="true"
    >
      <span class="h-[1.125rem] w-1.5 origin-bottom rounded-[2px] bg-current opacity-55 [animation:spinner-bars_1s_ease-in-out_infinite_0s]" />
      <span class="h-[1.125rem] w-1.5 origin-bottom rounded-[2px] bg-current opacity-75 [animation:spinner-bars_1s_ease-in-out_infinite_0.12s]" />
      <span class="h-[1.125rem] w-1.5 origin-bottom rounded-[2px] bg-current [animation:spinner-bars_1s_ease-in-out_infinite_0.24s]" />
    </span>

    <span
      v-else-if="variant === 'terminal'"
      :class="cn('flex items-center justify-center gap-0.5', scaleClass)"
      aria-hidden="true"
    >
      <span>&gt;</span>
      <span class="inline-block h-4 w-1.5 bg-current [animation:spinner-caret_1s_steps(1)_infinite]" />
    </span>

    <span
      v-else-if="variant === 'shimmer-text'"
      class="spinner-shimmer-text inline-block bg-clip-text text-transparent [animation:spinner-shimmer_1.5s_linear_infinite]"
      aria-hidden="true"
    >{{ text }}</span>

    <span
      v-else-if="variant === 'loading-dots'"
      class="flex items-end justify-center"
      aria-hidden="true"
    >
      <span class="pb-px">{{ text }}</span>
      <span class="ms-0.5 inline-flex items-end gap-px pb-px">
        <span class="inline-block [animation:spinner-ellipsis_1s_ease-in-out_infinite_0s]">.</span>
        <span class="inline-block [animation:spinner-ellipsis_1s_ease-in-out_infinite_0.12s]">.</span>
        <span class="inline-block [animation:spinner-ellipsis_1s_ease-in-out_infinite_0.24s]">.</span>
      </span>
    </span>

    <span
      v-else-if="variant === 'ellipsis'"
      class="inline-flex items-end gap-1.5"
      aria-hidden="true"
    >
      <span :class="cn('inline-block shrink-0 rounded-full bg-current [animation:spinner-ellipsis_0.7s_ease-in-out_infinite_0s]', dotSizeClass)" />
      <span :class="cn('inline-block shrink-0 rounded-full bg-current [animation:spinner-ellipsis_0.7s_ease-in-out_infinite_0.12s]', dotSizeClass)" />
      <span :class="cn('inline-block shrink-0 rounded-full bg-current [animation:spinner-ellipsis_0.7s_ease-in-out_infinite_0.24s]', dotSizeClass)" />
    </span>

    <span
      v-else-if="variant === 'typing-dots'"
      :class="cn('inline-flex gap-x-1', scaleClass)"
      aria-hidden="true"
    >
      <span :class="cn('rounded-full bg-current [animation:spinner-typing_1s_ease-in-out_infinite]', dotSizeClass)" />
      <span :class="cn('rounded-full bg-current [animation:spinner-typing_1s_ease-in-out_infinite_0.2s]', dotSizeClass)" />
      <span :class="cn('rounded-full bg-current [animation:spinner-typing_1s_ease-in-out_infinite_0.4s]', dotSizeClass)" />
    </span>

    <span class="sr-only">{{ label }}</span>
  </span>
</template>
