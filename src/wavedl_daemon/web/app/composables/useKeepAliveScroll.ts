/**
 * Preserve the scroll offset across `<KeepAlive>` deactivation.
 *
 * Nuxt's router already restores `savedPosition`, but it does so exactly one
 * animation frame after `page:loading:end`. A kept-alive page that is still
 * settling at that instant — rebuilding a list, filling in cached rows — can be
 * shorter than the offset being restored, and the browser silently clamps to the
 * bottom. Re-applying across a bounded frame loop, stopping as soon as the
 * scroller is tall enough, makes the restore independent of that timing.
 *
 * Record while the scroller is connected: KeepAlive detaches the DOM before
 * onDeactivated runs, at which point browsers report an element's scrollTop as 0.
 *
 * The scroll owner can pass its own ref; injection only reaches descendants,
 * so a page cannot inject the canvas provided by its child ShellPage.
 *
 * @param scrollTarget - The owned scroller, or the injected canvas/window fallback.
 * @param maxFrames - How many frames to keep retrying before giving up. The cap
 *   matters: a page whose content never returns would otherwise fight the user's
 *   own scrolling indefinitely.
 */
export function useKeepAliveScroll(
  scrollTarget: Readonly<Ref<ShellScrollTarget | null>> = useShellScroll().scrollTarget,
  maxFrames = 10,
): void {
  let savedY = 0;
  let active = false;
  let restoring = false;
  let restoreFrame: number | null = null;

  const savePosition = () => {
    if (!import.meta.client || !active || restoring) return;

    const target = scrollTarget.value;

    if (!target || (isShellScrollElement(target) && !target.isConnected)) return;

    savedY = Math.max(0, getShellScrollTop(target));
  };

  const cancelRestore = () => {
    if (restoreFrame !== null) {
      cancelAnimationFrame(restoreFrame);
      restoreFrame = null;
    }

    restoring = false;
  };

  useEventListener(scrollTarget, 'scroll', savePosition, { passive: true });
  useEventListener(scrollTarget, ['wheel', 'touchstart', 'keydown'], cancelRestore, {
    passive: true,
  });
  onBeforeRouteLeave(savePosition);

  onDeactivated(() => {
    savePosition();
    active = false;
    cancelRestore();
  });

  onActivated(() => {
    if (!import.meta.client) return;

    active = true;
    cancelRestore();

    if (savedY <= 0) return;

    const targetY = savedY;
    let frames = 0;
    restoring = true;

    const apply = () => {
      restoreFrame = null;

      if (!active || !restoring) return;

      frames += 1;
      const target = scrollTarget.value;

      if (target && (!isShellScrollElement(target) || target.isConnected)) {
        const maxScrollTop = getShellScrollMax(target);
        target.scrollTo({ left: 0, top: Math.min(targetY, maxScrollTop), behavior: 'instant' });

        if (maxScrollTop >= targetY) {
          restoring = false;

          return;
        }
      }

      if (frames >= maxFrames) {
        restoring = false;

        return;
      }

      restoreFrame = requestAnimationFrame(apply);
    };

    restoreFrame = requestAnimationFrame(apply);
  });

  onScopeDispose(cancelRestore);
}
