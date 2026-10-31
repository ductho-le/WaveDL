import type { ConnectionSession } from '../lib/connection-session';
import { CatalogApi } from '../lib/catalog-api';
import type { HostProfile } from '../../shared/schemas/host-profile';

/** One host profile per verified connection revision; null while unknown. */
export function useHostProfile(source: MaybeRefOrGetter<ConnectionSession>) {
  const profile = shallowRef<HostProfile | null>(null);
  const busy = shallowRef(false);
  let revision = 0;
  let request: AbortController | null = null;
  let unsubscribe: (() => void) | undefined;

  function cancel() {
    revision++;
    request?.abort();
    request = null;
    busy.value = false;
  }

  async function load() {
    cancel();
    const current = revision;
    request = new AbortController();
    busy.value = true;

    try {
      const value = await new CatalogApi(toRaw(toValue(source))).host(request.signal);

      if (current === revision) profile.value = value;
    } catch {
      if (current === revision) profile.value = null;
    } finally {
      if (current === revision) {
        busy.value = false;
        request = null;
      }
    }
  }

  watch(
    () => toRaw(toValue(source)),
    (session) => {
      unsubscribe?.();
      cancel();
      profile.value = null;
      let scope = -1;
      let connected = false;
      unsubscribe = session.subscribe((state) => {
        if (state.revision === scope && connected && state.status === 'connected') return;
        scope = state.revision;
        connected = state.status === 'connected';
        cancel();
        profile.value = null;

        if (state.status === 'connected') void load();
      });
    },
    { immediate: true },
  );
  onBeforeUnmount(() => {
    unsubscribe?.();
    cancel();
  });

  return { profile, busy, load };
}
