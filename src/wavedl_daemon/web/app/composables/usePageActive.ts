const pageActiveKey: InjectionKey<ComputedRef<boolean>> = Symbol('page-active');

/**
 * Tracks whether the enclosing kept-alive page is shown; outside KeepAlive it
 * stays active. Activity is inherited: KeepAlive never deactivates a child that
 * first mounts inside an already-hidden page (e.g. a reconnect remount), so the
 * child must start from its ancestors' state. `resume`/`pause` run once per
 * transition while mounted.
 */
export function usePageActive(resume?: () => void, pause?: () => void) {
  const parent = inject(pageActiveKey, null);
  const shown = shallowRef(true);
  const mounted = shallowRef(false);
  const isActive = computed(() => shown.value && (parent?.value ?? true));
  provide(pageActiveKey, isActive);
  let running = false;
  watch(
    () => mounted.value && isActive.value,
    (active) => {
      if (active === running) return;
      running = active;

      if (active) resume?.();
      else pause?.();
    },
    { flush: 'sync' },
  );
  onMounted(() => {
    mounted.value = true;
  });
  onActivated(() => {
    shown.value = true;
  });
  onDeactivated(() => {
    shown.value = false;
  });
  onBeforeUnmount(() => {
    mounted.value = false;
  });

  return { isActive };
}
