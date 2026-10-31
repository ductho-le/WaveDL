import modelSourceHashes from './model-source-hashes.json';
import modelMessagesVI from './locales/vi/models.json';
import type { ModelDescription } from '../lib/catalog-api';
import type { WorkstationLocale } from './core';

export type BundledModelId = keyof typeof modelSourceHashes;

export type ModelMessageKey =
  | `models.${BundledModelId}.description`
  | `models.${BundledModelId}.bestFor`
  | `models.${BundledModelId}.shortDescription`
  | `models.${BundledModelId}.pros.first`
  | `models.${BundledModelId}.pros.second`;

export type ModelMessageTranslator = (key: ModelMessageKey) => string;

/** Vietnamese model copy is bundled; English remains the daemon's authoritative text. */
export const bundledModelMessages = {
  vi: modelMessagesVI,
  sourceHashes: modelSourceHashes,
} as const;

export interface ModelPresentation {
  readonly shortDescription: string;
  readonly pros: readonly string[];
  readonly description: string;
  readonly bestFor: string;
  readonly untranslated: boolean;
}

function isBundledModelId(id: string): id is BundledModelId {
  return Object.hasOwn(modelSourceHashes, id);
}

/** Non-cryptographic staleness guard only; it is not a security boundary. */
export function hashModelSource(value: string): string {
  let first = 0xdeadbeef;
  let second = 0x41c6ce57;

  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    first = Math.imul(first ^ code, 0x85ebca77);
    second = Math.imul(second ^ code, 0xc2b2ae3d);
  }

  first =
    Math.imul(first ^ (first >>> 16), 0x85ebca77) ^ Math.imul(second ^ (second >>> 13), 0xc2b2ae3d);
  second =
    Math.imul(second ^ (second >>> 16), 0x85ebca77) ^ Math.imul(first ^ (first >>> 13), 0xc2b2ae3d);
  const hash = 0x1_0000_0000 * (0x1fffff & second) + (first >>> 0);

  return hash.toString(16).padStart(14, '0');
}

/** Unknown IDs and changed daemon metadata remain authoritative, verbatim plaintext. */
export function modelPresentation(
  model: ModelDescription,
  locale: WorkstationLocale,
  translate: ModelMessageTranslator,
): ModelPresentation {
  const original = {
    description: model.description,
    bestFor: model.bestFor,
    shortDescription: model.shortDescription ?? '',
    pros: model.pros?.slice(0, 2) ?? [],
  };

  if (locale === 'en') return { ...original, untranslated: false };

  if (!isBundledModelId(model.id)) return { ...original, untranslated: true };

  const source = modelSourceHashes[model.id];

  if (
    hashModelSource(model.description) !== source.description ||
    hashModelSource(model.bestFor) !== source.bestFor ||
    hashModelSource(original.shortDescription) !== source.shortDescription ||
    hashModelSource(original.pros[0] ?? '') !== source.pros.first ||
    hashModelSource(original.pros[1] ?? '') !== source.pros.second
  )
    return { ...original, untranslated: true };

  return {
    description: translate(`models.${model.id}.description`),
    bestFor: translate(`models.${model.id}.bestFor`),
    shortDescription: translate(`models.${model.id}.shortDescription`),
    pros: [translate(`models.${model.id}.pros.first`), translate(`models.${model.id}.pros.second`)],
    untranslated: false,
  };
}
