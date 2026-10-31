import type { ConnectionStatus } from '../lib/connection-session';

export const connectionStatusKey: Record<ConnectionStatus, string> = {
  connected: 'shell.connected',
  connecting: 'shell.connecting',
  disconnected: 'shell.notConnected',
  authentication: 'shell.notConnected',
  skew: 'shell.notConnected',
  offline: 'shell.offline',
  error: 'shell.notConnected',
};
