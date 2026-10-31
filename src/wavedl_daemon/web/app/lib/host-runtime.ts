import type { HostProfile } from '../../shared/schemas/host-profile';
import type { TrainingRequest } from '../../shared/schemas/training-request';

export type AutomaticRuntime = Pick<TrainingRequest, 'backend' | 'runtime' | 'topology'>;

/**
 * The setup automatic mode plans: one Local process on the daemon's own interpreter.
 * `auto` lets the trainer pick CUDA, Apple GPU or CPU itself; Local refuses explicit
 * MPS and requires named devices for explicit CUDA, so auto is the one choice that
 * always plans.
 */
export function automaticRuntime(profile: HostProfile): AutomaticRuntime {
  return {
    backend: 'local',
    runtime: { python: profile.python, mode: 'existing', device: 'auto', environment: {} },
    topology: { num_machines: 1, processes_per_machine: 1, machine_rank: 0 },
  };
}

/** Whether the request already holds exactly the automatic setup (storage remaps included). */
export function matchesAutomatic(request: TrainingRequest, profile: HostProfile): boolean {
  const expected = automaticRuntime(profile);

  return (
    request.backend === expected.backend &&
    JSON.stringify(request.runtime) === JSON.stringify(expected.runtime) &&
    JSON.stringify(request.topology) === JSON.stringify(expected.topology) &&
    !request.storage_map?.length
  );
}

export type HostAccelerator = 'cuda' | 'mps' | 'cpu' | 'unknown';

/** What `auto` resolves to on this host, in the trainer's order of preference. */
export function hostAccelerator(profile: HostProfile): HostAccelerator {
  const found = profile.accelerators;

  if (!found || !found.torch) return 'unknown';

  if (found.cuda.length) return 'cuda';

  return found.mps ? 'mps' : 'cpu';
}

/** Binary gigabytes as hardware reports them: 25769803776 → "24 GB". */
export function gigabytes(bytes: number): string {
  const value = bytes / 1024 ** 3;

  return `${value >= 10 ? Math.round(value) : Number(value.toFixed(1))} GB`;
}

/** CUDA_VISIBLE_DEVICES naming one distinct device per Local process: 2 → "0,1". */
export function visibleDevices(processes: number): string {
  return Array.from({ length: Math.max(1, processes) }, (_, index) => String(index)).join(',');
}
