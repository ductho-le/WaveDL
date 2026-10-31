import { inferenceDefaults, type ResultRequest } from '../../shared/schemas/result-request';

/** Fresh draft defaults; interpreter, dataset and checkpoint require selection. */
export function createResultOperationDefaults(): ResultRequest {
  return {
    backend: 'local',
    storage_map: [],
    operation: 'test',
    config: structuredClone(inferenceDefaults),
    checkpoint_source: { run_id: '', generation: '' },
    runtime: { python: '', mode: 'existing', device: 'cpu', environment: {} },
    topology: { num_machines: 1, processes_per_machine: 1, machine_rank: 0 },
  };
}
