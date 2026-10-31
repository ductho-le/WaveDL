export const shellScrollKey: InjectionKey<Ref<HTMLElement | null>> = Symbol('shell-scroll');

export type ShellScrollTarget = Window | HTMLElement;

export function provideShellScroll(container: Ref<HTMLElement | null>): void {
  provide(shellScrollKey, container);
}

export function useShellScroll() {
  const injected = inject(shellScrollKey, undefined);

  const scrollTarget = computed<ShellScrollTarget | null>(() => {
    if (injected) return injected.value;

    if (import.meta.client) return window;

    return null;
  });

  return { scrollTarget };
}

export function isShellScrollElement(target: ShellScrollTarget): target is HTMLElement {
  return target instanceof HTMLElement;
}

export function getShellScrollTop(target: ShellScrollTarget): number {
  return isShellScrollElement(target) ? target.scrollTop : target.scrollY;
}

export function getShellScrollMax(target: ShellScrollTarget): number {
  if (isShellScrollElement(target)) {
    return Math.max(0, target.scrollHeight - target.clientHeight);
  }

  return Math.max(0, document.documentElement.scrollHeight - target.innerHeight);
}

export function scrollShellTo(target: ShellScrollTarget, options: ScrollToOptions): void {
  target.scrollTo(options);
}
