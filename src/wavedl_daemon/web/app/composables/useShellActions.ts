import type { WorkstationSection } from '../shell/sections';

type ShellActions = (section: WorkstationSection) => void;

const key: InjectionKey<ShellActions> = Symbol('shell-actions');

export function provideShellActions(navigate: ShellActions): void {
  provide(key, navigate);
}

export function useShellActions(): ShellActions {
  const navigate = inject(key);

  if (!navigate) throw new Error('Shell actions require WorkstationShell');

  return navigate;
}
