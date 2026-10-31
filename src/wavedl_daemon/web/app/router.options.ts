import type { RouterConfig } from 'nuxt/schema';
import { stringifyQuery } from 'vue-router';

export default {
  // Keep return paths encoded in login URLs (Router 5 otherwise prints literal slashes).
  stringifyQuery: (query) =>
    stringifyQuery(query).replace(
      /(^|&)next=([^&]*)/g,
      (_match: string, prefix: string, value: string) =>
        `${prefix}next=${value.replaceAll('/', '%2F')}`,
    ),
} satisfies RouterConfig;
