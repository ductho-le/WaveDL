import { isUuid } from '../lib/daemon-api-schemas';
import { sectionPath } from '../shell/sections';

/** Route params are the only selection authority; corrections never add history. */
export function useRunRoute(section: 'experiments' | 'results') {
  const router = useRouter();
  const root = sectionPath(section);

  // Nuxt freezes useRoute() for cached pages. A retired page must never correct
  // the URL belonging to the page the user is actually looking at.
  const active = computed(() => {
    const path = router.currentRoute.value.path;

    return path === root || path.startsWith(`${root}/`);
  });

  const routeRunId = computed(() => {
    const value = router.currentRoute.value.params.runId;

    return Array.isArray(value) ? null : (value ?? null);
  });

  const runId = shallowRef<string | null>(null);
  watch(
    () => (active.value ? routeRunId.value : undefined),
    (value) => {
      if (!active.value) return;
      runId.value = isUuid(value) ? value : null;

      if (value && !isUuid(value)) void router.replace(root);
    },
    { immediate: true },
  );

  function select(id: string | null): void {
    void router.push(sectionPath(section, id ?? undefined));
  }

  function unknown(id: string): void {
    if (active.value && runId.value === id) void router.replace(root);
  }

  return { runId, select, unknown };
}
