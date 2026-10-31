import { ConnectionSession, type ConnectionOptions } from '../lib/connection-session';
import { stripBootstrapUrl } from '../lib/daemon-api';
import { clearSessionCredentialArea, loadSessionCredential } from '../lib/session-credential';
import { loadKeepSignedIn, saveKeepSignedIn } from '../lib/preferences';

/** Owns browser storage and mount ordering for the current daemon session. */
export function useDaemonConnection(options: ConnectionOptions) {
  const router = useRouter();
  const auth = useWorkstationAuth();
  auth.value = { status: 'disconnected', resumePending: true };
  const session = new ConnectionSession(options);
  const state = shallowRef(session.snapshot());
  const api = shallowRef(session.api);
  const keepSignedIn = shallowRef(false);

  const unsubscribe = session.subscribe((next) => {
    state.value = next;
    api.value = session.api;
    auth.value = { ...auth.value, status: next.status };
  });

  const canMutate = computed(() => state.value.status === 'connected');

  function setKeepSignedIn(value: boolean): void {
    const next = value === true;

    if (keepSignedIn.value === next) return;
    keepSignedIn.value = next;
    saveKeepSignedIn(next);
    session.setKeepSignedIn(next);

    if (!next) clearSessionCredentialArea('local');
  }

  const online = () => session.setOnline(navigator.onLine);
  let disposed = false;
  onMounted(async () => {
    online();
    window.addEventListener('online', online);
    window.addEventListener('offline', online);

    try {
      const stripped = stripBootstrapUrl(window.location.href);
      const clean = new URL(stripped.url);
      // Complete router replacement before reading storage or starting a request.
      await router.replace(`${clean.pathname}${clean.search}`);

      if (disposed) return;
      keepSignedIn.value = loadKeepSignedIn();

      if (!keepSignedIn.value) clearSessionCredentialArea('local');

      if (stripped.code !== null || stripped.invalid) {
        await session.bootstrapCode(stripped.code, stripped.invalid, keepSignedIn.value);
      } else {
        const stored = loadSessionCredential({}, options.origin);

        if (stored) await session.resume(stored.credential, keepSignedIn.value);
      }
    } catch {
      session.detach();
    } finally {
      if (!disposed) auth.value = { status: session.snapshot().status, resumePending: false };
    }
  });
  onScopeDispose(() => {
    disposed = true;
    unsubscribe();
    session.dispose();

    if (import.meta.client) {
      window.removeEventListener('online', online);
      window.removeEventListener('offline', online);
    }
  });

  return {
    state: shallowReadonly(state),
    api: shallowReadonly(api),
    canMutate,
    keepSignedIn: shallowReadonly(keepSignedIn),
    setKeepSignedIn,
    session,
    connect: (token: string, persist = keepSignedIn.value) => session.connect(token, persist),
    reconnect: (token: string, persist = keepSignedIn.value) => session.reconnect(token, persist),
    detach: () => session.disconnect(),
    refresh: () => session.refresh(),
  };
}
