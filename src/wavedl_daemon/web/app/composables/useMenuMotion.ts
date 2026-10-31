import { SPRING_DEFAULT } from '@/lib/animation';

export function useMenuContentMotion() {
  const shouldReduceMotion = useReducedMotion();

  const initial = computed(() =>
    shouldReduceMotion.value ? { opacity: 1 } : { opacity: 0.8, scale: 0.95, y: -4 },
  );

  const animate = computed(() =>
    shouldReduceMotion.value ? { opacity: 1 } : { opacity: 1, scale: 1, y: 0 },
  );

  const transition = computed(() => (shouldReduceMotion.value ? { duration: 0 } : SPRING_DEFAULT));

  return { initial, animate, transition };
}

export function useMenuItemMotion() {
  const shouldReduceMotion = useReducedMotion();

  const initial = computed(() =>
    shouldReduceMotion.value ? { opacity: 1 } : { opacity: 1, y: -4 },
  );

  const animate = computed(() =>
    shouldReduceMotion.value ? { opacity: 1 } : { opacity: 1, y: 0 },
  );

  const transition = computed(() => (shouldReduceMotion.value ? { duration: 0 } : SPRING_DEFAULT));

  return { initial, animate, transition };
}
