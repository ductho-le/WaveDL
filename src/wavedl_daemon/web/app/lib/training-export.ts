import { clientError } from '../i18n/client-messages';
import options from '../../shared/training-options.json';
import { trainingRequestSchema, type TrainingRequest } from '../../shared/schemas/training-request';

/** JSON scalar syntax is also safe YAML syntax; no implicit YAML typing. */
export function trainingYaml(draft: TrainingRequest): string {
  const { config } = trainingRequestSchema.parse(draft);

  return (
    '# WaveDL training configuration. Runtime credentials and environment are omitted.\n' +
    Object.entries(config)
      .map(([name, value]) => `${name}: ${JSON.stringify(value)}`)
      .join('\n') +
    '\n'
  );
}

/** Quote only words the shell would otherwise split or expand. */
export function quotePosix(value: string): string {
  if (/^[\w@%+=:,./-]+$/.test(value)) return value;

  return `'${value.replaceAll("'", "'\\''")}'`;
}

/** Always shown so the command reads on its own, even when they match defaults. */
const essentialOptions = ['model', 'data_path'];

/**
 * A standalone `wavedl-train` command for the installed CLI, listing only the
 * options that differ from upstream defaults. It is not a daemon plan.
 */
export function trainingCommand(draft: TrainingRequest): string {
  const request = trainingRequestSchema.parse(draft);

  // The launcher needs a rendezvous address this editor does not collect.
  if (request.topology.num_machines !== 1 || request.topology.machine_rank !== 0)
    throw clientError('exportMultiNode', {});

  const environment = Object.entries(request.runtime.environment).map(
    ([name, value]) => `${name}=${quotePosix(value)}`,
  );

  // Explicit, like the daemon's plan, so launcher auto-detection cannot add processes.
  const argv = ['wavedl-train', '--num_gpus', String(request.topology.processes_per_machine)];

  // auto, mps and cuda all leave device selection to the launcher.
  if (request.runtime.device === 'cpu') argv.push('--cpu');

  for (const option of options) {
    const entry = Object.entries(request.config).find(([name]) => name === option.name);

    if (!entry) throw clientError('incompleteTrainingSchema', {});
    const value = entry[1];

    if (
      !essentialOptions.includes(option.name) &&
      JSON.stringify(value) === JSON.stringify(option.default)
    )
      continue;

    if (value === null || value === false || (Array.isArray(value) && !value.length)) continue;
    const flag = option.name === 'import_modules' ? '--import' : `--${option.name}`;

    if (value === true) argv.push(flag);
    else if (Array.isArray(value)) {
      const items = value.map(String);

      // argparse reads "-1e-7" as a flag; only a plain negative decimal is a value.
      if (items.some((item) => item.startsWith('-') && !/^-\d*\.?\d+$/.test(item)))
        throw clientError('exportOptionPrefix', {});
      argv.push(flag, ...items);
    } else argv.push(`${flag}=${value}`);
  }

  argv.push('--output_dir', './results');

  return [...environment, ...argv.map(quotePosix)].join(' ');
}
