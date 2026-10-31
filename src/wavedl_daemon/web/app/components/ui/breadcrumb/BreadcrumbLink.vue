<script lang="ts" setup>
import type { HTMLAttributes } from 'vue'
import type { NuxtLinkProps } from '#app'

import { cn } from '@/lib/utils'

const props = withDefaults(defineProps<NuxtLinkProps & { class?: HTMLAttributes['class'] }>(), {})

/**
 * Everything NuxtLink should receive, minus `custom`.
 *
 * Vue gives every boolean prop a `false` default, while NuxtLink's typed props
 * only admit `custom: true` — the variant that renders nothing and hands its
 * slot the resolved route. Spreading the whole props object therefore fails to
 * typecheck. This component renders an anchor, so `custom` was never wanted.
 */
const linkProps = computed(() => {
  // oxlint-disable-next-line no-unused-vars
  const { custom, ...rest } = props
  return rest
})
</script>

<template>
  <NuxtLink
    data-slot="breadcrumb-link"
    v-bind="linkProps"
    :class="cn('hover:text-foreground transition-colors', props.class)"
    prefetch-on="interaction"
  >
    <slot />
  </NuxtLink>
</template>
