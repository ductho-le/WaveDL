/**
 * Floating content is placed against the physical viewport, which `viewport-fit=cover`
 * extends under the notch — so a menu tall enough to be shifted or flipped can land in
 * the cutout, and a menu sized against the available height can grow into one. Reka
 * hands this padding to floating-ui, which uses it for both, so a single value keeps
 * every menu positioned and sized inside the safe area.
 *
 * This is the floating counterpart to the `app-safe-*` rules in `tailwind.css`: those
 * cover overlays whose geometry is CSS, this covers the ones whose geometry is computed.
 */

/** How far a floating panel stands off a viewport edge before any cutout is counted. */
const FLOATING_EDGE_GUTTER = 8;

interface SafeAreaCollisionPadding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/**
 * One reader for the whole app. The first menu to ask would otherwise own the resize
 * listener and take it away again on unmount, leaving every later menu reading insets
 * frozen at whichever orientation that first menu happened to open in.
 */
let sharedPadding: ComputedRef<SafeAreaCollisionPadding> | null = null;

/** `useScreenSafeArea` reports CSS lengths; anything unresolved counts as no cutout. */
function toPixels(value: string): number {
  const parsed = Number.parseFloat(value);

  return Number.isFinite(parsed) ? parsed : 0;
}

function gutterOnly(): SafeAreaCollisionPadding {
  return {
    top: FLOATING_EDGE_GUTTER,
    right: FLOATING_EDGE_GUTTER,
    bottom: FLOATING_EDGE_GUTTER,
    left: FLOATING_EDGE_GUTTER,
  };
}

export function useSafeAreaCollisionPadding(): ComputedRef<SafeAreaCollisionPadding> {
  // Nothing to read without a device, and a module-level cache must not outlive a request.
  if (import.meta.server) return computed(gutterOnly);

  if (sharedPadding) return sharedPadding;

  // Detached, so the listener outlives whichever component opened the first menu.
  const scope = effectScope(true);

  const created = scope.run(() => {
    const safeArea = useScreenSafeArea();

    return computed<SafeAreaCollisionPadding>(() => ({
      top: FLOATING_EDGE_GUTTER + toPixels(safeArea.top.value),
      right: FLOATING_EDGE_GUTTER + toPixels(safeArea.right.value),
      bottom: FLOATING_EDGE_GUTTER + toPixels(safeArea.bottom.value),
      left: FLOATING_EDGE_GUTTER + toPixels(safeArea.left.value),
    }));
  });

  if (!created) {
    scope.stop();

    return computed(gutterOnly);
  }

  sharedPadding = created;

  return sharedPadding;
}
