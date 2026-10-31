import type { ConnectionStatus } from './connection-session';
import { stripBootstrapUrl } from './daemon-api';
import { nextPathSchema } from '../../shared/schemas/auth-routing';
import type { LocationQueryValue } from 'vue-router';

const WORK_PATHS = ['/experiments', '/new', '/datasets', '/results'];

export function validatedNextPath(input: LocationQueryValue | LocationQueryValue[] | undefined): string {
  const parsed = nextPathSchema.safeParse(input);

  if (!parsed.success) return '/experiments';
  const value = parsed.data;

  if (!value.startsWith('/') || value.trim() !== value) return '/experiments';

  try {
    let decoded = value;

    // Check every encoding layer, but return the original query/path encoding.
    for (let depth = 0; depth < 8; depth++) {
      if ([...decoded].some((char) => char === '\\' || char <= '\u001f' || char === '\u007f') || decoded.startsWith('//'))
        return '/experiments';
      const url = new URL(decoded, 'https://wavedl.invalid');

      if (
        url.origin !== 'https://wavedl.invalid' ||
        url.pathname.startsWith('//') ||
        url.pathname.toLowerCase().startsWith('/login')
      )
        return '/experiments';
      const stripped = new URL(stripBootstrapUrl(url.href).url);

      if (stripped.search !== url.search) return '/experiments';
      const next = decodeURIComponent(decoded);

      if (next === decoded) {
        const original = new URL(value, url.origin);

        return `${original.pathname}${original.search}${original.hash}`;
      }

      decoded = next;
    }

    return '/experiments';
  } catch {
    return '/experiments';
  }
}

function isWorkstationPath(path: string): boolean {
  const pathname = path.split(/[?#]/, 1)[0] ?? '';

  return WORK_PATHS.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

export function authRedirect(
  path: string,
  status: ConnectionStatus,
  resumePending: boolean,
): string | null {
  if (!isWorkstationPath(path) || resumePending) return null;

  if (status !== 'disconnected' && status !== 'authentication') return null;

  return `/login?next=${encodeURIComponent(validatedNextPath(path))}`;
}
