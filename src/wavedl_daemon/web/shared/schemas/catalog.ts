import { z } from 'zod';
import { jsonObjectSchema } from './json';

const text = (maximum = 8192) => z.string().max(maximum).refine((value) => !value.includes('\0'));
const nonEmptyText = (maximum = 8192) => text(maximum).refine((value) => value.trim().length > 0);
const integer = z.number().refine(Number.isSafeInteger).nonnegative();
const rootId = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const modelId = z.string().regex(/^[a-z0-9_]{1,128}$/);
const kind = z.enum(['file', 'directory']);

export const relativePathSchema = text(4096).refine((value) =>
  new TextEncoder().encode(value).length <= 4096 &&
  (value === '' || (value.split('/').length <= 128 && value.split('/').every((part) =>
    part !== '' && part !== '.' && part !== '..' && !/[\\:]/.test(part),
  ))),
);

export const datasetHostPathSchema = text().refine((value) => {
  let parts: string[];

  if (value.startsWith('/')) parts = value.slice(1).split('/');
  else if (/^[A-Za-z]:\\/.test(value)) parts = value.slice(3).split('\\');
  else if (value.startsWith('\\\\')) {
    parts = value.slice(2).split('\\');

    if (parts.length < 2) return false;
  } else return false;

  return parts.every((part, index) =>
    part !== '.' && part !== '..' && (part !== '' || index === parts.length - 1),
  );
});

export const browseLocationSchema = z.object({ root: rootId, path: relativePathSchema, offset: integer.max(10000) });

export const browseSelectionSchema = browseLocationSchema.omit({ offset: true });

export const browseRootsSchema = z.strictObject({
  roots: z.array(z.strictObject({ id: rootId })).max(32)
    .refine((items) => new Set(items.map((item) => item.id)).size === items.length),
});

const browseEntrySchema = z.strictObject({
  name: relativePathSchema.refine((value) => value !== '' && !value.includes('/')),
  kind,
});

export const browsePageSchema = z.strictObject({
  root: rootId,
  path: relativePathSchema,
  entries: z.array(browseEntrySchema).max(100),
  total: integer.max(10000),
  limit: z.literal(100),
  offset: integer.max(10000),
}).refine((value) =>
  value.entries.length === Math.min(value.limit, Math.max(0, value.total - value.offset)) &&
  new Set(value.entries.map((entry) => entry.name)).size === value.entries.length,
);

export const resolvedDatasetSchema = z.strictObject({
  root: rootId,
  relative_path: relativePathSchema,
  path: datasetHostPathSchema,
  kind,
}).refine((value) => value.relative_path !== '' || value.kind === 'directory');

const textList = (maximumItems: number, maximumLength: number) =>
  z.array(nonEmptyText(maximumLength)).max(maximumItems)
    .refine((items) => new Set(items).size === items.length);

const categorySchema = z.object({ short_description: text(), models: z.array(modelId).max(1000) });

export const catalogModelSchema = z.strictObject({
  id: modelId,
  display_name: nonEmptyText(256),
  architecture_family: z.enum(['CNN', 'Hybrid', 'Transformer', 'SSM', 'Encoder-Decoder']),
  author_institution: nonEmptyText(256),
  year_published: integer.min(1800).max(3000),
  params_m: z.number().finite().min(0).max(1_000_000),
  size_mb: z.number().finite().min(0).max(1_000_000),
  short_description: nonEmptyText(),
  description: nonEmptyText(),
  best_for: nonEmptyText(),
  pros: textList(32, 2048),
  cons: textList(32, 2048),
  compute_tier: z.enum(['Edge', 'Low', 'Medium', 'High']),
  tags: textList(32, 256),
  supported_dims: z.array(z.union([z.literal(1), z.literal(2), z.literal(3)])).min(1).max(3)
    .refine((items) => new Set(items).size === items.length),
  is_pretrained: z.boolean(),
});

export const modelCatalogSchema = z.strictObject({
  source: z.literal('bundled_metadata'),
  runtime_verified: z.literal(false),
  categories: jsonObjectSchema.transform((value) => Object.entries(value))
    .pipe(z.array(z.tuple([text(256), categorySchema])).max(128)),
  models: z.array(catalogModelSchema).max(1000),
});

export const customModelSelectionSchema = z.object({ model: z.string().regex(/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/) });
