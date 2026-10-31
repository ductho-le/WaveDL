import { Package, ScrollText } from '@lucide/vue';

export const SHELL_PANEL_MIN_WIDTH = 320;
export const SHELL_PANEL_DEFAULT_WIDTH = 368;
export const SHELL_PANEL_MAX_WIDTH = 520;

export type ShellPanelTabId = 'logs' | 'artifacts';

export interface ShellPanelTabDefinition {
  id: ShellPanelTabId;
  labelKey: string;
  icon: Component;
}

export const SHELL_PANEL_TABS: readonly ShellPanelTabDefinition[] = [
  { id: 'logs', labelKey: 'shell.logs', icon: ScrollText },
  { id: 'artifacts', labelKey: 'shell.artifacts', icon: Package },
];

export function shellPanelTabTargetId(tab: ShellPanelTabId): string {
  return `shell-panel-tab-${tab}`;
}
