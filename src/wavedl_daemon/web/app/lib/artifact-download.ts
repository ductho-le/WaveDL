import { clientError } from '../i18n/client-messages';
import type { Artifact } from './artifact-client';
import { isUuid } from './daemon-api';

const encoder = new TextEncoder();

function trimBytes(value: string, limit: number): string {
  let result = '';
  let bytes = 0;

  for (const character of value) {
    const size = encoder.encode(character).byteLength;

    if (bytes + size > limit) break;
    result += character;
    bytes += size;
  }

  return result;
}

/** Browser save label only; never use this name to address a daemon artifact. */
function basenameLabel(path: string): string {
  let name = (path.split('/').at(-1) ?? '')
    .normalize('NFKC')
    .replace(/[\p{Cc}\p{Cf}<>:"/\\|?*]/gu, '_')
    .replace(/[. ]+$/u, '')
    .replace(/^[. ]+/u, '');

  if (!name) name = 'artifact';

  if (/^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/iu.test(name)) name = `_${name}`;

  return name;
}

export async function artifactDownloadName(
  artifact: Artifact,
  inventory: readonly Artifact[] = [],
): Promise<string> {
  if (
    !isUuid(artifact.runId) ||
    !/^[0-9a-f]{64}$/u.test(artifact.sha256) ||
    !/^gen-\d{6,}$/.test(artifact.generation) ||
    artifact.generation.length > 64
  )
    throw clientError('artifactProvenanceTooLong', {});
  const prefix = `${artifact.runId}_${artifact.generation}_`;
  const budget = Math.min(180, 255 - encoder.encode(prefix).byteLength);

  const paths = [
    ...new Set([
      artifact.path,
      ...inventory
        .flatMap((other) =>
          other.runId === artifact.runId && other.generation === artifact.generation ? [other.path] : [],
        ),
    ]),
  ];

  async function candidate(path: string, disambiguate = false): Promise<string> {
    const name = basenameLabel(path);

    if (!disambiguate && encoder.encode(name).byteLength <= budget) return name;
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode(path)));
    const pathHash = Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
    const extension = name.match(/\.[A-Za-z0-9]{1,16}$/u)?.[0] ?? '';
    const stem = extension ? name.slice(0, -extension.length) : name;
    const suffix = `_${pathHash.slice(0, 24)}${extension}`;

    return trimBytes(stem, budget - encoder.encode(suffix).byteLength) + suffix;
  }

  const nativeKey = (name: string) => name.normalize('NFKC').toLocaleLowerCase('en-US');
  const initial = await Promise.all(paths.map((path) => candidate(path)));
  const counts = new Map<string, number>();

  for (const name of initial) counts.set(nativeKey(name), (counts.get(nativeKey(name)) ?? 0) + 1);

  const names = await Promise.all(
    paths.map((path, index) =>
      (counts.get(nativeKey(initial[index] ?? '')) ?? 0) > 1
        ? candidate(path, true)
        : initial[index],
    ),
  );

  const selected = names[0];

  // Also compare generated labels: a short sibling can deliberately equal a
  // truncated/hash-suffixed label. Chained aliases or digest collisions refuse
  // export rather than delegating filename collision handling to the browser.
  if (!selected || names.slice(1).some((name) => name && nativeKey(name) === nativeKey(selected)))
    throw clientError('artifactNamesTooLong', {});

  return prefix + selected;
}
