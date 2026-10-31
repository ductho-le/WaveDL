import { z } from 'zod';

export const nextPathSchema = z.string().min(1).max(2048);
