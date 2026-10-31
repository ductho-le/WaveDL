export const modelFamilyMessageKeys = {
  CNN: 'catalog.familyCnn',
  Hybrid: 'catalog.familyHybrid',
  Transformer: 'catalog.familyTransformer',
  SSM: 'catalog.familySsm',
  'Encoder-Decoder': 'catalog.familyEncoderDecoder',
};

export const modelTierMessageKeys = {
  Edge: 'catalog.tierEdge',
  Low: 'catalog.tierLow',
  Medium: 'catalog.tierMedium',
  High: 'catalog.tierHigh',
};

export const modelShapeLabelMessageKeys = {
  described: 'catalog.shapeDescribed',
  mismatch: 'catalog.shapeMismatch',
  unknown: 'catalog.shapeUnknown',
  ambiguous: 'catalog.shapeAmbiguous',
};

export function mappedMessageKey(
  map: Readonly<Record<string, string>>,
  value: string,
): string | undefined {
  return Object.hasOwn(map, value) ? map[value] : undefined;
}
