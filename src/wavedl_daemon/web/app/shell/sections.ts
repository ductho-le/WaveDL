import { isUuid } from '../lib/daemon-api-schemas';

/**
 * Section integration contract:
 * - The router selects the active work page; activeSection is derived from its path.
 * - Teleport run logs to #shell-panel-tab-logs and results to
 *   #shell-panel-tab-artifacts. Both targets stay mounted while the panel is closed.
 * - Use Teleport's `defer` (or wait for onMounted) because targets are siblings of
 *   the canvas. Gate teleported bodies by activeSection, not the section's v-show.
 * - useShellLayout().setActiveTab('logs' | 'artifacts') selects and opens a tab;
 *   useShellPanelTabs() can register a section-specific subset of the real tabs.
 * - WorkstationShell exposes an `actions` slot beside its Experiments Adopt action.
 *   Connection and Settings are pages; Adopt is an overlay on Experiments.
 */
import { FileChartColumn, Layers, Database, Plus, Cable, Settings2, Import } from '@lucide/vue';
import {
  WORKSTATION_SECTIONS,
  type PrimaryWorkstationSection,
  type WorkstationSection,
} from '../lib/shell-test-ids';

export type { PrimaryWorkstationSection, WorkstationSection };

export type ShellCommand = 'adopt' | 'connection' | 'settings';

export interface ShellNavigationDescriptor<
  Section extends WorkstationSection = WorkstationSection,
> {
  id: Section;
  labelKey: string;
  icon: Component;
  path: string;
  command?: ShellCommand;
}

export const SHELL_PRIMARY_NAVIGATION: readonly ShellNavigationDescriptor<PrimaryWorkstationSection>[] =
  [
    {
      id: WORKSTATION_SECTIONS.experiments,
      labelKey: 'shell.experiments',
      icon: Layers,
      path: '/experiments',
    },
    {
      id: WORKSTATION_SECTIONS.project,
      labelKey: 'shell.newTrainingRun',
      icon: Plus,
      path: '/new',
    },
    {
      id: WORKSTATION_SECTIONS.datasets,
      labelKey: 'shell.datasets',
      icon: Database,
      path: '/datasets',
    },
    {
      id: WORKSTATION_SECTIONS.results,
      labelKey: 'shell.results',
      icon: FileChartColumn,
      path: '/results',
    },
  ];

export const SHELL_ACTION_NAVIGATION: readonly ShellNavigationDescriptor<
  Exclude<WorkstationSection, PrimaryWorkstationSection>
>[] = [
  {
    id: WORKSTATION_SECTIONS.adopt,
    labelKey: 'shell.adoptStandaloneRun',
    icon: Import,
    path: '/experiments',
    command: 'adopt',
  },
  {
    id: WORKSTATION_SECTIONS.connection,
    labelKey: 'shell.connection',
    icon: Cable,
    path: '/login',
    command: 'connection',
  },
  {
    id: WORKSTATION_SECTIONS.settings,
    labelKey: 'shell.settings',
    icon: Settings2,
    path: '/settings',
    command: 'settings',
  },
];

export function sectionPath(section: PrimaryWorkstationSection, runId?: string): string {
  const navigation = SHELL_PRIMARY_NAVIGATION.find((item) => item.id === section);
  const path = navigation?.path ?? '/experiments';

  if (!runId || (section !== 'experiments' && section !== 'results')) return path;

  return `${path}/${encodeURIComponent(runId)}`;
}

export function sectionForPath(path: string): WorkstationSection {
  if (path === '/login' || path === '/login/') return 'connection';

  if (path === '/settings' || path.startsWith('/settings/')) return 'settings';

  return (
    SHELL_PRIMARY_NAVIGATION.find((item) => path === item.path || path.startsWith(`${item.path}/`))
      ?.id ?? WORKSTATION_SECTIONS.experiments
  );
}

/** The run named by an Experiments or Results URL, if it is a canonical run id. */
export function runIdForPath(path: string): string | null {
  const [, section, runId, ...rest] = path.split('/');

  if (rest.length || (section !== 'experiments' && section !== 'results')) return null;

  return isUuid(runId) ? runId : null;
}

/** Experiments and Results share the current run, as the single-canvas shell did. */
export function navigationPath(path: string, currentRun: string | null): string {
  if (!currentRun || (path !== '/experiments' && path !== '/results')) return path;

  return `${path}/${currentRun}`;
}
