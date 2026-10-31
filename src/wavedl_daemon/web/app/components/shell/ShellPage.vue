<script setup lang="ts">
import { SHELL_PANEL_TABS } from '../../shell/constants';
import { cn } from '@/lib/utils';

const props = withDefaults(
  defineProps<{
    panelTabs?: readonly ShellPanelTabDefinition[];
  }>(),
  {
    panelTabs: () => SHELL_PANEL_TABS,
  },
);

const route = useRoute();

useShellPanelTabs(() => props.panelTabs);

/*
 * The page owns its scroller so retention and BackToTop address the same element.
 */
const scrollContainer = shallowRef<HTMLElement | null>(null);

provideShellScroll(scrollContainer);

useKeepAliveScroll(scrollContainer);

const fullWidth = computed(() => route.meta.fullWidth ?? true);

/*
 * Internally scrolling workspaces can opt into a bounded canvas through page meta.
 */
const fillHeight = computed(() => route.meta.fillHeight === true);

const bodyClass = computed(() =>
  cn(
    'flex flex-1 flex-col gap-4 px-4 py-4 md:gap-6 md:px-6 md:py-6',
    !fullWidth.value && 'mx-auto w-full max-w-screen-2xl',
    fillHeight.value && 'min-h-0 h-full overflow-hidden',
  ),
);

const scrollClass = computed(() =>
  cn('min-h-0 flex-1', fillHeight.value ? 'overflow-hidden' : 'overflow-x-hidden overflow-y-auto'),
);
</script>

<template>
  <ShellWorkspace>
    <template #canvas>
      <div class="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <slot name="header">
          <ShellPageHeader>
            <template v-if="$slots['title-controls']" #title-controls>
              <slot name="title-controls" />
            </template>
          </ShellPageHeader>
        </slot>
        <div
          id="workspace"
          ref="scrollContainer"
          data-shell-scroll
          data-testid="workstation-content"
          tabindex="-1"
          :class="scrollClass"
        >
          <div :class="bodyClass">
            <slot />
          </div>
        </div>
        <BackToTop />
        <slot name="scroll-overlay" />
      </div>
    </template>
  </ShellWorkspace>
</template>
