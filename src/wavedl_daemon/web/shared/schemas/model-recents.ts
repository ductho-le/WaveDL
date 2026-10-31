import { z } from 'zod';

const modelId = z.string();

export const recentModelsSchema = z.array(z.unknown()).transform((items) => items.flatMap((item) => {
  const parsed = modelId.safeParse(item);

  return parsed.success ? [parsed.data] : [];
}).slice(0, 4));
