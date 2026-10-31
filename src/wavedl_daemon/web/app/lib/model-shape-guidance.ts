import type { ModelDescription } from './catalog-api';
import type { DatasetInspection } from './dataset-inspection';

export type ModelShapeGuidance = {
  status: 'described' | 'mismatch' | 'unknown' | 'ambiguous';
  shape: string;
  dimensions: number;
};

/** Header-only advice using prepare_data's spatial-shape convention, not model validation. */
export function modelShapeGuidance(
  inspection: DatasetInspection | null | undefined,
  model: Pick<ModelDescription, 'supportedDims'>,
  singleChannel = false,
): ModelShapeGuidance {
  const shape = inspection?.inputShape ?? [];
  const dimensions = shape.length;
  const unknown: ModelShapeGuidance = { status: 'unknown', shape: '', dimensions: 0 };

  if (
    !inspection ||
    dimensions < 1 ||
    dimensions > 3 ||
    shape.some((size) => !Number.isSafeInteger(size) || size < 1)
  )
    return unknown;
  const result = { shape: shape.join(' × '), dimensions };
  const [first, second, third] = shape;

  // Training adds a channel; it does not strip a leading size of 1 or a small axis.
  // This exact rank-three heuristic requires explicit single_channel confirmation.
  if (
    dimensions === 3 &&
    first !== undefined &&
    first <= 16 &&
    second !== undefined &&
    second > 16 &&
    third !== undefined &&
    third > 16 &&
    !singleChannel
  )
    return { ...result, status: 'ambiguous' };

  if (!model.supportedDims.length) return unknown;

  return {
    ...result,
    status: model.supportedDims.includes(dimensions) ? 'described' : 'mismatch',
  };
}
