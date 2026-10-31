import { z } from 'zod';

const count = z.number().int().nonnegative();

/** `GET /api/host`: what the daemon's own machine and interpreter can run. Additive fields are ignored. */
export const hostProfileSchema = z.object({
  hostname: z.string().max(255),
  os: z.string().max(64),
  os_release: z.string().max(128),
  arch: z.string().max(64),
  cpu_count: count.nullable(),
  memory_bytes: count.nullable(),
  python: z.string().min(1).max(8192),
  python_version: z.string().max(64),
  // Null when the accelerator probe failed; not the same as "no accelerator".
  accelerators: z
    .object({
      torch: z.string().max(64).nullable(),
      cuda: z.array(z.object({ name: z.string().max(200), memory_bytes: count })).max(64),
      mps: z.boolean(),
    })
    .nullable(),
  backends: z.array(z.string().max(64)).max(16),
});
export type HostProfile = z.infer<typeof hostProfileSchema>;

export const runtimeSetups = ['automatic', 'manual'] as const;
export type RuntimeSetup = (typeof runtimeSetups)[number];
