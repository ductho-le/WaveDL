import type { ShallowRef } from 'vue';
import type { DaemonApiError } from '../lib/daemon-api';
import type { ConnectionSession } from '../lib/connection-session';
import type { AppearancePreferences } from '../lib/preferences';
import type { useAutomaticPlots } from './useAutomaticPlots';
import type { useDaemonConnection } from './useDaemonConnection';

export type WorkstationOverlay = 'adopt' | null;

export type WorkstationConnection = ReturnType<typeof useDaemonConnection>;

export type WorkstationAutomaticPlots = ReturnType<typeof useAutomaticPlots>;

export interface WorkstationContext {
  connection: WorkstationConnection;
  state: WorkstationConnection['state'];
  api: WorkstationConnection['api'];
  session: ConnectionSession;
  automaticPlots: WorkstationAutomaticPlots;
  reportFailure: ComputedRef<(error: DaemonApiError) => void>;
  revision: ComputedRef<number>;
  overlay: ShallowRef<WorkstationOverlay>;
  appearance: ShallowRef<AppearancePreferences>;
  appearanceChanged(value: AppearancePreferences): void;
  monitorRun(runId: string): Promise<void>;
}

const workstationKey: InjectionKey<WorkstationContext> = Symbol('workstation');

export function provideWorkstation(context: WorkstationContext): void {
  provide(workstationKey, context);
}

export function useWorkstation(): WorkstationContext {
  const context = inject(workstationKey);

  if (!context) throw new Error('useWorkstation() must be called inside the workstation layout.');

  return context;
}
