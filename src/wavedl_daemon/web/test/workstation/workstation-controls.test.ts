import { describe, expect, it } from 'vitest';
import {
  locateRequestSchema,
  resultExportSchema,
  runtimeDeviceSchema,
  segmentedSelectionSchema,
  storedDockSlotsSchema,
} from '../../shared/schemas/workstation-controls';
import { planManifestSchema, planPreviewContextSchema } from '../../shared/schemas/plan-summary';
import { customIssueParamsSchema } from '../../shared/schemas/client-diagnostic';
import {
  configureValidationLocale,
  translateClientDiagnostic,
} from '../../app/i18n/client-messages';
import {
  trainingConfigBaseSchema,
  trainingDefaults,
  trainingFieldMetadata,
} from '../../shared/schemas/training-request';
import { inferenceConfigSchema } from '../../shared/schemas/result-request';
import { trainingCommand } from '../../app/lib/training-export';
import { loadShellDockSlots } from '../../app/lib/preferences';
import { trainingChoiceLabel, trainingPresetMessageKeysFor } from '../../app/i18n/dynamic-keys';

describe('workstation control boundaries', () => {
  it('accepts declared select values without coercing other emit values', () => {
    for (const device of ['cpu', 'cuda', 'mps', 'auto'])
      expect(runtimeDeviceSchema.parse(device)).toBe(device);

    for (const value of ['', 1, false, null, {}, ['cpu']])
      expect(runtimeDeviceSchema.safeParse(value).success).toBe(false);

    expect(resultExportSchema.parse(null)).toBeNull();
    expect(resultExportSchema.parse('onnx')).toBe('onnx');
    expect(resultExportSchema.safeParse('other').success).toBe(false);
    expect(segmentedSelectionSchema(['cpu', 'cuda']).safeParse('').success).toBe(false);
    expect(segmentedSelectionSchema(['cpu', 'cuda']).parse('cuda')).toBe('cuda');
    expect(segmentedSelectionSchema([]).safeParse('cpu').success).toBe(false);
    expect(storedDockSlotsSchema.parse(['project', {}, null, 2, 'results'])).toEqual([
      'project',
      null,
      null,
      null,
      'results',
    ]);
    expect(storedDockSlotsSchema.safeParse(['project']).success).toBe(false);
  });

  it('keeps additive locate request behavior', () => {
    expect(locateRequestSchema.parse({ root: '', path: '', extra: true })).toEqual({
      root: '',
      path: '',
    });
    expect(locateRequestSchema.safeParse({ root: 'data', path: 1 }).success).toBe(false);
  });

  it('rejects unknown, stale, empty and non-string checkpoint choices', () => {
    const choices = segmentedSelectionSchema(['best_checkpoint', 'epoch_5_checkpoint']);

    expect(choices.parse('epoch_5_checkpoint')).toBe('epoch_5_checkpoint');

    for (const value of ['interrupted_checkpoint', '', 2, {}, null])
      expect(choices.safeParse(value).success).toBe(false);

    expect(
      segmentedSelectionSchema(['best_checkpoint']).safeParse('epoch_5_checkpoint').success,
    ).toBe(false);
  });

  it('keeps every selectable training default among its declared choices', () => {
    const choices = trainingFieldMetadata.filter((field) => field.choices.length > 0);

    expect(choices.map((field) => field.name)).toEqual([
      'loss',
      'optimizer',
      'scheduler',
      'cache_validate',
      'precision',
      'constraint_reduction',
    ]);

    for (const field of choices) expect(field.choices).toContain(trainingDefaults[field.name]);
  });

  it('retains dock recovery for malformed stored JSON and mixed or duplicate slots', () => {
    const defaults = ['experiments', 'project', 'datasets', 'results', 'settings'];

    for (const raw of [null, 'broken', 'null', '{}', '[]', '["project"]']) {
      const slots = loadShellDockSlots({
        getItem: () => raw,
        setItem: () => {},
        removeItem: () => {},
      });

      expect(slots).toEqual(defaults);
    }

    for (const stored of [
      ['project', {}, null, 2, 'results'],
      ['project', 'project', 'project', 'project', 'project'],
    ]) {
      const slots = loadShellDockSlots({
        getItem: () => JSON.stringify(stored),
        setItem: () => {},
        removeItem: () => {},
      });

      expect(slots).toEqual(['project', 'experiments', 'datasets', 'results', 'settings']);
    }
  });

  it('retains the original manifest while decoding review fields once', () => {
    const manifest = { operation: 1, resolved_config: [], runtime: { device: false }, extra: true };
    const parsed = planManifestSchema.parse(manifest);

    expect(parsed.manifest).toEqual(manifest);
    expect(parsed.review).toEqual({
      operation: undefined,
      resolved_config: {},
      runtime: { device: undefined },
    });
    expect(planManifestSchema.safeParse([]).success).toBe(false);
    expect(planPreviewContextSchema.parse({ target: null, topology: false })).toEqual({
      target: undefined,
      topology: {},
    });
    expect(
      planManifestSchema.parse({ runtime: { device: 'cuda' }, resolved_config: { epochs: 4 } })
        .review,
    ).toMatchObject({ runtime: { device: 'cuda' }, resolved_config: { epochs: 4 } });
  });

  it('uses catalog keys and only string/number interpolation params', () => {
    const params = customIssueParamsSchema.parse({
      clientMessageKey: 'nulNotAllowed',
      count: 2,
      ignored: true,
    });

    expect(params).toEqual({ clientMessageKey: 'nulNotAllowed', count: 2 });
    expect(customIssueParamsSchema.safeParse({ clientMessageKey: 'not-a-key' }).success).toBe(
      false,
    );
    expect(translateClientDiagnostic('en', 'ordinary text')).toBe('ordinary text');
    expect(translateClientDiagnostic('en', new Error('ordinary error'))).toBe('ordinary error');
    expect(trainingPresetMessageKeysFor('__proto__')).toBeNull();
    expect(trainingChoiceLabel('unknown', 'unknown', (key) => key)).toBe('unknown');
    expect(trainingChoiceLabel('loss', 'weighted_mse', (key) => key)).toBe(
      'builder.choiceWeightedMse',
    );

    configureValidationLocale('en');
    const result = trainingConfigBaseSchema.shape.model.safeParse('\0');

    expect(result.success).toBe(false);

    if (!result.success) expect(result.error.issues[0]?.message).not.toBe('Invalid input');
  });

  it('refuses only list entries that upstream would read as a flag', () => {
    const config = trainingConfigBaseSchema.shape;
    const listed = config.constraint.safeParse(['y0 - y1', '-y0']);

    expect(listed.success).toBe(false);

    if (!listed.success)
      expect(listed.error.issues[0]?.message).toBe('List entries cannot start with "-".');
    expect(config.import_modules.safeParse(['-m']).success).toBe(false);
    expect(inferenceConfigSchema.shape.param_names.safeParse(['-a']).success).toBe(false);
    expect(inferenceConfigSchema.shape.import_modules.safeParse(['-m']).success).toBe(false);
    expect(config.run_name.parse('-x')).toBe('-x');
    expect(config.grad_clip.parse(-1e-7)).toBe(-1e-7);
    expect(inferenceConfigSchema.shape.input_key.parse('-X')).toBe('-X');
  });

  it('exports negative list numbers only in a form argparse reads as values', () => {
    const request = (constraint_weight: number[]) =>
      trainingCommand({
        backend: 'local',
        operation: 'train',
        config: { ...trainingDefaults, grad_clip: -1e-7, constraint_weight },
        runtime: { python: '/usr/bin/python3', mode: 'existing', device: 'cpu', environment: {} },
        topology: { num_machines: 1, processes_per_machine: 1, machine_rank: 0 },
      });

    expect(request([-0.5, -2])).toContain("'--constraint_weight' '-0.5' '-2'");
    expect(request([-0.5])).toContain("'--grad_clip=-1e-7'");
    expect(() => request([-1e-7])).toThrow(
      'Standalone CLI list entries cannot start with an option prefix; use the YAML export.',
    );
  });
});
