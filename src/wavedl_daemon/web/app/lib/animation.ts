export const SPRING_DEFAULT = {
  bounce: 0.15,
  duration: 0.4,
  type: 'spring',
};

/**
 * Fast-out, slow-in. The house curve for chrome that appears over content: it
 * covers most of its distance immediately, so a panel reads as already-there
 * rather than as arriving.
 */
export const EASE_OUT_EXPO: [number, number, number, number] = [0.16, 1, 0.3, 1];
