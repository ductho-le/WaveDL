// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import { validatedNextPath, authRedirect } from '../../app/lib/auth-routing';

describe('validated login return path', () => {
  it('accepts the length boundary and rejects one extra character', () => {
    const prefix = '/datasets?filter=';
    const path = `${prefix}${'a'.repeat(2048 - prefix.length)}`;
    expect(validatedNextPath(path)).toBe(path);
    expect(validatedNextPath(`${path}a`)).toBe('/experiments');
  });
  it.each([
    '/experiments',
    '/datasets?sort=name#preview',
    '/datasets?label=two%20words',
    '/results',
    '/new',
    '/settings',
    '/',
  ])('accepts %s', (path) => {
    expect(validatedNextPath(path)).toBe(path);
  });
  it.each([
    undefined,
    null,
    [],
    ['/results'],
    '',
    'results',
    '//x',
    '/\\x',
    'https://x',
    'javascript:alert(1)',
    '/login',
    '/login?next=/results',
    '/login/extra',
    '/loginElse',
    '/LOGIN',
    '/%6cogin',
    '/%256cogin',
    '/%2fx',
    '/%252fx',
    '/%5cx',
    '/%255cx',
    '/datasets/../login',
    '/datasets/%2e%2e/login',
    '/datasets\n',
    '/datasets%0a',
    '/datasets/..//evil',
    '/%2e%2e//evil',
    '/login-extra',
    '/%4cOGIN',
    '/%25252flogin',
    '/%00',
    '/datasets%09',
    '/datasets%7f',
    'data:text/html,x',
    '/datasets?%62ootstrap=secret',
    '/datasets?authorization=secret',
    '/datasets?bootstrap=secret',
    '/datasets?access_token=secret',
    '/%',
    `/datasets?x=${'a'.repeat(2048)}`,
  ])('rejects unsafe or non-app next %j', (value) => {
    expect(validatedNextPath(value)).toBe('/experiments');
  });
});

describe('work page auth decision', () => {
  it('redirects disconnected work pages with a sanitized next', () => {
    expect(authRedirect('/datasets', 'disconnected', false)).toBe('/login?next=%2Fdatasets');
    expect(authRedirect('/results?sort=name', 'authentication', false)).toBe(
      '/login?next=%2Fresults%3Fsort%3Dname',
    );
  });
  it.each(['connecting', 'connected', 'offline', 'skew', 'error'])(
    'keeps %s on the current page',
    (status) => {
      expect(authRedirect('/datasets', status, false)).toBeNull();
    },
  );
  it('defers initial auth until mounted resume starts and exempts public pages', () => {
    expect(authRedirect('/datasets', 'disconnected', true)).toBeNull();
    expect(authRedirect('/login', 'disconnected', false)).toBeNull();
    expect(authRedirect('/settings', 'disconnected', false)).toBeNull();
    expect(authRedirect('/settings/storage', 'disconnected', false)).toBeNull();
  });
});
