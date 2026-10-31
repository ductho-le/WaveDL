import type { ConnectionSession } from '../lib/connection-session';
import { CatalogApi, type ModelDescription, type ModelSelection } from '../lib/catalog-api';

export type CatalogueMessage = '' | 'catalog.modelsUnavailable' | 'catalog.connectionChanged';

/** One catalogue per verified connection revision; selections are branded by CatalogApi. */
export function useModelCatalogue(source: MaybeRefOrGetter<ConnectionSession>) {
  const models = shallowRef<readonly ModelDescription[]>([]);
  const busy = shallowRef(false);
  const message = shallowRef<CatalogueMessage>('');
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
    message.value = '';
    models.value = [];

    try {
      const value = await new CatalogApi(toRaw(toValue(source))).models(request.signal);

      if (current === revision) models.value = value;
    } catch {
      if (current === revision) message.value = 'catalog.modelsUnavailable';
    } finally {
      if (current === revision) {
        busy.value = false;
        request = null;
      }
    }
  }

  /** `model` must be the catalogue's own frozen object: CatalogApi brands it. */
  function select(model: ModelDescription): ModelSelection | null {
    try {
      return new CatalogApi(toRaw(toValue(source))).selectModel(model, models.value);
    } catch {
      models.value = [];
      message.value = 'catalog.connectionChanged';

      return null;
    }
  }

  watch(
    () => toRaw(toValue(source)),
    (session) => {
      unsubscribe?.();
      cancel();
      models.value = [];
      let scope = -1;
      let connected = false;
      unsubscribe = session.subscribe((state) => {
        if (state.revision === scope && connected && state.status === 'connected') return;
        scope = state.revision;
        connected = state.status === 'connected';
        cancel();
        models.value = [];

        if (state.status === 'connected') {
          message.value = '';
          void load();
        } else if (state.status !== 'disconnected') message.value = 'catalog.modelsUnavailable';
      });
    },
    { immediate: true },
  );
  onBeforeUnmount(() => {
    unsubscribe?.();
    cancel();
  });

  return { models, busy, message, load, cancel, select };
}
