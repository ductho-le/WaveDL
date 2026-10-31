import {
  ConnectionSession as BrowserConnectionSession,
  type ConnectionOptions,
} from '../../app/lib/connection-session';

export type { ConnectionState } from '../../app/lib/connection-session';
import { healthSchema } from '../../app/lib/daemon-api-schemas';

/** Add the session endpoints to domain fixtures without altering their API responses. */
export class ConnectionSession extends BrowserConnectionSession {
  constructor(options: ConnectionOptions) {
    const fetcher = options.fetcher ?? globalThis.fetch.bind(globalThis);
    let daemonId: string | undefined = 'daemon';
    let generation = 0;

    const sessionFetch: typeof fetch = async (input, init) => {
      const url = new URL(input instanceof Request ? input.url : String(input));

      if (url.pathname === '/api/session') {
        if (init?.method === 'DELETE') return new Response(null, { status: 204 });

        if (!daemonId) throw new Error('Session fixture requires a health handshake first.');
        generation += 1;

        return new Response(
          JSON.stringify({
            access_token: `session${generation.toString().padStart(36, '0')}`,
            token_type: 'Bearer',
            expires_at: Date.now() + 12 * 60 * 60 * 1000,
            daemon_id: daemonId,
          }),
          { headers: { 'content-type': 'application/json' } },
        );
      }

      const response = await fetcher(input, init);

      if (url.pathname === '/api/health' && response.ok) {
        const parsed = healthSchema.pick({ daemon_id: true }).safeParse(await response.clone().json());

        if (parsed.success) daemonId = parsed.data.daemon_id;
      }

      return response;
    };

    super({ ...options, fetcher: sessionFetch });
  }
}
