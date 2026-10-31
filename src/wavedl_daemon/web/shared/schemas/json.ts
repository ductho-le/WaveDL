import { z } from 'zod';

export type Json = null | boolean | number | string | Json[] | JsonObject;

export type JsonObject = { [key: string]: Json };

export const jsonScalarSchema = z.union([z.null(), z.string(), z.boolean(), z.number().finite()]);

const containerFields = z.record(z.string(), z.unknown());

type JsonContainer = z.infer<typeof containerFields>;

// Keep own members, including __proto__. This container contract deliberately
// makes no claim about child types; only jsonObjectSchema validates those.
export const jsonContainerSchema = z.custom<JsonContainer>((value) =>
  containerFields.safeParse(value).success,
);

export const jsonObjectSchema: z.ZodType<JsonObject> = jsonContainerSchema.transform((value, ctx) => {
  const result: JsonObject = {};

  for (const [key, child] of Object.entries(value)) {
    const parsed = jsonSchema.safeParse(child);

    if (!parsed.success) {
      for (const issue of parsed.error.issues) ctx.addIssue({ ...issue, path: [key, ...issue.path] });

      return z.NEVER;
    }

    Object.defineProperty(result, key, {
      value: parsed.data, enumerable: true, writable: true, configurable: true,
    });
  }

  return result;
});

export const jsonSchema: z.ZodType<Json> = z.lazy(() =>
  z.union([
    jsonScalarSchema,
    z.array(jsonSchema),
    jsonObjectSchema,
  ]),
);
