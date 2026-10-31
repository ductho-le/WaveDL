<script setup lang="ts">
import { Command, Import, PanelRight, Maximize2, Minimize2 } from '@lucide/vue';
import { connectionStatusKey } from '../../constants/shellConnection';

const { t } = useI18n({ useScope: 'global' });
const navigate = useShellActions();
const layout = useShellLayout();
const { state } = useWorkstation();

const { activeSection, isDockedLayout, isCompanionShowing, isFocusMode, commandPaletteOpen } =
  layout;

const statusLabel = computed(() => t(connectionStatusKey[state.value.status]));
</script>

<template>
  <Button
    variant="ghost"
    :size="isDockedLayout ? 'sm' : 'icon'"
    data-testid="connection-status"
    :aria-label="`${t('shell.connectionStatus')}: ${statusLabel}`"
    @click="navigate('connection')"
  >
    <span class="shell-status-dot" :data-status="state.status" aria-hidden="true" />
    <span class="hidden lg:inline">{{ statusLabel }}</span>
  </Button>
  <Button
    v-if="activeSection === 'experiments'"
    class="hidden sm:inline-flex"
    variant="outline"
    size="icon"
    data-testid="nav-adopt"
    :aria-label="t('shell.adoptStandaloneRun')"
    :title="t('shell.adoptStandaloneRun')"
    @click="navigate('adopt')"
  >
    <Import aria-hidden="true" />
  </Button>
  <Button
    variant="ghost"
    size="icon"
    :aria-label="t('shell.commandPalette')"
    :title="t('shell.commandPaletteShortcut')"
    @click="commandPaletteOpen = true"
  >
    <Command aria-hidden="true" />
  </Button>
  <Button
    class="hidden md:inline-flex"
    variant="ghost"
    size="icon"
    :aria-label="t(isFocusMode ? 'shell.exitFocus' : 'shell.enterFocus')"
    :aria-pressed="isFocusMode"
    @click="layout.toggleFocusMode()"
  >
    <Minimize2 v-if="isFocusMode" aria-hidden="true" /><Maximize2 v-else aria-hidden="true" />
  </Button>
  <Button
    variant="ghost"
    size="icon"
    :aria-label="t('shell.openCompanionPanel')"
    :aria-pressed="isCompanionShowing"
    aria-controls="shell-companion"
    @click="layout.togglePanel()"
  >
    <PanelRight aria-hidden="true" />
  </Button>
</template>
