import type { ConnectionStatus } from '../lib/connection-session';

export interface WorkstationAuthState {
  status: ConnectionStatus;
  resumePending: boolean;
}

/** Middleware runs before the layout mounts. Defer until initial resume settles. */
export function useWorkstationAuth() {
  return useState('workstation-auth', (): WorkstationAuthState => ({
    status: 'disconnected',
    resumePending: true,
  }));
}
