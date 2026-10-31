import englishClientErrors from './locales/en/client-errors.json';
import vietnameseClientErrors from './locales/vi/client-errors.json';
import englishZodLocale from 'zod/v4/locales/en.js';
import vietnameseZodLocale from 'zod/v4/locales/vi.js';
import { z } from 'zod';
import type { MessageParams, WorkstationLocale } from './core';
import { clientDiagnosticSchema, customIssueParamsSchema } from '../../shared/schemas/client-diagnostic';

export type ClientErrorKey = keyof typeof englishClientErrors;

export type ClientErrorParams = MessageParams;

type ClientErrorCatalog = Record<ClientErrorKey, string>;

export interface ClientDiagnostic {
  readonly clientMessageKey: ClientErrorKey;
  readonly clientMessageParams?: ClientErrorParams;
}

export type ClientMessage = string | ClientDiagnostic;

let validationLocale: WorkstationLocale = 'en';

/** A client-owned diagnostic whose English Error.message remains API-compatible. */
export class ClientError extends Error implements ClientDiagnostic {
  constructor(
    readonly clientMessageKey: ClientErrorKey,
    message = englishClientErrors[clientMessageKey],
    readonly clientMessageParams: ClientErrorParams = {},
  ) {
    super(message);
    this.name = 'ClientError';
  }
}

/** Preserve RangeError semantics for bounded client-side collections. */
export class ClientRangeError extends RangeError implements ClientDiagnostic {
  constructor(
    readonly clientMessageKey: ClientErrorKey,
    message = englishClientErrors[clientMessageKey],
    readonly clientMessageParams: ClientErrorParams = {},
  ) {
    super(message);
    this.name = 'ClientRangeError';
  }
}

export function clientError(key: ClientErrorKey, params: ClientErrorParams = {}): ClientError {
  return new ClientError(key, interpolate(englishClientErrors[key], params), params);
}

export function clientRangeError(
  key: ClientErrorKey,
  params: ClientErrorParams = {},
): ClientRangeError {
  return new ClientRangeError(key, interpolate(englishClientErrors[key], params), params);
}

function interpolate(template: string, params: ClientErrorParams): string {
  return template.replace(/\{([A-Za-z][A-Za-z0-9_]*)\}/g, (_match, name: string) => {
    const value = params[name];

    return value === undefined ? `{${name}}` : String(value);
  });
}

export function isClientDiagnostic(value: ClientMessage | Error): value is ClientDiagnostic {
  return clientDiagnosticSchema.safeParse(value).success;
}

const localeMessages: Record<WorkstationLocale, ClientErrorCatalog> = {
  en: englishClientErrors,
  vi: vietnameseClientErrors,
};

function translatedClientError(
  locale: WorkstationLocale,
  key: ClientErrorKey,
  params: ClientErrorParams,
): string | undefined {
  const template = localeMessages[locale][key];

  if (template === undefined) return undefined;

  return interpolate(template, params);
}

/** Use Zod 4's bundled locale maps and stable keys for product-defined custom issues. */
export function configureValidationLocale(locale: WorkstationLocale): void {
  validationLocale = locale;
  z.config({
    localeError:
      locale === 'vi' ? vietnameseZodLocale().localeError : englishZodLocale().localeError,
    customError: (issue) => {
      if (issue.code !== 'custom' || !('params' in issue) || !issue.params) return undefined;
      const parsed = customIssueParamsSchema.safeParse(issue.params);

      if (!parsed.success) return undefined;

      return translatedClientError(validationLocale, parsed.data.clientMessageKey, parsed.data);
    },
  });
}

/** Resolve schema-owned overrides while retaining Zod's built-in map for other issues. */
export function validationMessage(key: ClientErrorKey, params: ClientErrorParams = {}): string {
  return interpolate(localeMessages[validationLocale][key], params);
}

/** Pure helper for non-component callers; raw or daemon-owned text is never reverse-matched. */
export function translateClientDiagnostic(
  locale: WorkstationLocale,
  diagnostic: ClientMessage | Error,
): string {
  const parsed = clientDiagnosticSchema.safeParse(diagnostic);

  if (!parsed.success) return diagnostic instanceof Error ? diagnostic.message : String(diagnostic);
  const template = localeMessages[locale][parsed.data.clientMessageKey];

  if (template === undefined) return parsed.data.clientMessageKey;
  const params = parsed.data.clientMessageParams ?? {};

  return interpolate(template, params);
}

configureValidationLocale('en');
