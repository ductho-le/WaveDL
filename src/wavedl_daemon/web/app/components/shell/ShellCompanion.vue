<script setup lang="ts">
import { FocusScope } from 'reka-ui';
import { SHELL_PANEL_TABS } from '../../shell/constants';

const {
  resolvedActiveTab,
  closePanel,
  isCompanionShowing,
  isDockedLayout,
  panelTabs,
  panelWidth,
  setActiveTab,
  setPanelWidth,
} = useShellLayout();

let opener: HTMLElement | null = null;

watch(isCompanionShowing, (visible) => {
  if (visible && !isDockedLayout.value) {
    opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    void nextTick(() => document.querySelector<HTMLElement>('#shell-companion button')?.focus());
  } else if (!visible && opener?.isConnected) {
    const target = opener;
    void nextTick(() => nextTick(() => target.focus()));
    opener = null;
  }
});
</script>

<template>
  <FocusScope
    as-child
    :loop="!isDockedLayout && isCompanionShowing"
    :trapped="!isDockedLayout && isCompanionShowing"
    @mount-auto-focus="$event.preventDefault()"
    @unmount-auto-focus="$event.preventDefault()"
  >
    <ShellPanel
      id="shell-companion"
      class="shell-companion"
      :role="isDockedLayout ? 'complementary' : 'dialog'"
      :aria-modal="!isDockedLayout && isCompanionShowing ? true : undefined"
      :tabs="panelTabs"
      :active-tab="resolvedActiveTab"
      :width="panelWidth"
      @update:active-tab="setActiveTab"
      @resize="setPanelWidth"
      @collapse="closePanel"
      @keydown.esc.prevent="closePanel()"
    >
      <template v-for="tab in SHELL_PANEL_TABS" :key="tab.id" #[tab.id]>
        <ShellPanelTarget :tab="tab">
          <slot :name="tab.id" />
        </ShellPanelTarget>
      </template>
    </ShellPanel>
  </FocusScope>
</template>
