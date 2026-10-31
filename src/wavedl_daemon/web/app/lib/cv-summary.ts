import { clientError } from '../i18n/client-messages';
import { ArtifactClient, type Artifact } from './artifact-client';
import type { ConnectionSession } from './connection-session';
import { parseJson, type Json } from './daemon-api';
import { cvSummarySchema } from '../../shared/schemas/cv-summary';
import type { z } from 'zod';

export const CV_SUMMARY_BUDGET = 1024 * 1024;

export type CvMetric = number | null;

export type CvSummary = z.output<typeof cvSummarySchema>;

export type CvFold = CvSummary['fold_results'][number];

/** Exact current Python producer schema; target labels are indices, not inferred names. */
export function parseCvSummary(input: Json): CvSummary {
  const parsed = cvSummarySchema.safeParse(input);

  if (!parsed.success) throw clientError('invalidCrossValidationSummary', {});

  return parsed.data;
}

export async function loadCvSummary(
  session: ConnectionSession,
  artifact: Artifact,
  signal?: AbortSignal,
): Promise<CvSummary> {
  const client = new ArtifactClient(session);
  client.assertSelection(artifact);

  if (artifact.path !== 'cv_summary.json' || artifact.size > CV_SUMMARY_BUDGET)
    throw clientError('cvSummaryTooLarge', {});
  const blob = await client.verifiedBlob(artifact, signal);
  signal?.throwIfAborted();
  const text = await blob.text();
  signal?.throwIfAborted();
  client.assertSelection(artifact);
  const summary = parseCvSummary(parseJson(text));
  signal?.throwIfAborted();
  client.assertSelection(artifact);

  return summary;
}
