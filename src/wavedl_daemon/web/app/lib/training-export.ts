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

export function quotePosix(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

/** A standalone host command is not a daemon plan or a scheduler submission. */
export function trainingCommand(draft: TrainingRequest): string {
  const request = trainingRequestSchema.parse(draft);

  if (!request.runtime.python.startsWith('/')) throw clientError('exportWindowsHost', {});

  if (request.topology.num_machines !== 1 || request.topology.machine_rank !== 0)
    throw clientError('exportMultiNode', {});

  if (!['cpu', 'cuda'].includes(request.runtime.device))
    throw clientError('exportDeviceRequired', {});

  if (request.runtime.mode === 'container') throw clientError('exportContainerStartup', {});

  if (Object.keys(request.runtime.environment).length)
    throw clientError('envValuesNotExported', {});

  const argv = [
    request.runtime.python,
    '-m',
    'wavedl.launcher',
    '--num_gpus',
    String(request.topology.processes_per_machine),
    '--mixed_precision',
    request.config.precision,
  ];

  if (request.runtime.device === 'cpu') argv.push('--cpu');

  for (const option of options) {
    const entry = Object.entries(request.config).find(([name]) => name === option.name);

    if (!entry) throw clientError('incompleteTrainingSchema', {});
    const value = entry[1];

    if (value === null || value === false || (Array.isArray(value) && !value.length)) continue;
    const flag = option.name === 'import_modules' ? '--import' : `--${option.name}`;

    if (value === true) argv.push(flag);
    else if (Array.isArray(value)) {
      if (value.some((item) => !Number.isFinite(item) && String(item).startsWith('-')))
        throw clientError('exportOptionPrefix', {});
      argv.push(flag, ...value.map(String));
    } else argv.push(`${flag}=${value}`);
  }

  argv.push('--output_dir', './results');

  return argv.map(quotePosix).join(' ');
}
