import { readFileSync, writeFileSync } from 'node:fs';
import tailwindcss from '@tailwindcss/vite';

// The browser must identify the Python build it was shipped with. Updating the
// package cannot silently retain a stale hard-coded client build identifier.
const versionSource = readFileSync(new URL('../../wavedl/__init__.py', import.meta.url), 'utf8');
const version = versionSource.match(/^__version__ = "([A-Za-z0-9.+_-]+)"$/m)?.[1];

if (!version) throw new Error('Cannot determine the WaveDL package build identity');

const identityPath = new URL('./app/lib/build-identity.ts', import.meta.url);
const identitySource = `/** Generated from the Python package version by nuxt.config.ts. */\nexport const CLIENT_BUILD_ID = '${version}';\n`;

if (readFileSync(identityPath, 'utf8') !== identitySource)
  writeFileSync(identityPath, identitySource);

// The Nuxt 4 default app/ source root contains the browser application. There is
// no Nitro API or backend.
export default defineNuxtConfig({
  ssr: false,
  modules: [
    '@nuxtjs/i18n',
    'shadcn-nuxt',
    '@vueuse/nuxt',
    '@pinia/nuxt',
    '@nuxt/fonts',
    'motion-v/nuxt',
  ],
  fonts: {
    defaults: { subsets: ['latin', 'latin-ext', 'vietnamese'] },
    families: [
      {
        name: 'Manrope',
        provider: 'google',
        weights: ['200 800'],
        styles: ['normal'],
        global: true,
      },
      {
        name: 'Source Serif 4',
        provider: 'google',
        weights: ['200 900'],
        styles: ['normal', 'italic'],
        global: true,
      },
    ],
  },
  shadcn: { prefix: '', componentDir: '@/components/ui' },
  i18n: {
    strategy: 'no_prefix',
    detectBrowserLanguage: false,
    locales: ['en', 'vi'],
    defaultLocale: 'en',
    vueI18n: './i18n.config.ts',
    bundle: { runtimeOnly: false, fullInstall: false },
  },
  hooks: {
    'nitro:config'(nitro) {
      // Both locales are static imports in vueI18n. The daemon has no Nitro
      // runtime, so do not generate the module's unused messages endpoint.
      nitro.handlers = nitro.handlers?.filter((handler) => !handler.route?.startsWith('/_i18n/'));
    },
  },
  // Component file names are unique, so register them unprefixed to match their
  // tag names. shadcn-nuxt claims components/ui first and registers its named exports.
  components: [{ path: '~/components', pathPrefix: false }],
  css: ['@/assets/css/tailwind.css'],
  devtools: { enabled: false },
  telemetry: false,
  debug: false,
  compatibilityDate: '2026-10-05',
  // Build skew is surfaced by daemon health. Keep Nuxt's automatic chunk-error
  // and outdated-build plugins disabled so reloads remain user initiated.
  experimental: {
    emitRouteChunkError: false,
    checkOutdatedBuildInterval: false,
  },
  app: {
    head: {
      charset: 'utf-8',
      viewport:
        'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover',
      title: 'WaveDL Workstation',
      htmlAttrs: { lang: 'en' },
      meta: [{ name: 'description', content: 'A local interface to the WaveDL daemon.' }],
      link: [{ rel: 'icon', href: '/favicon.svg', type: 'image/svg+xml' }],
    },
  },
  nitro: {
    compressPublicAssets: true,
    minify: true,
    preset: 'static',
    prerender: { routes: ['/'], crawlLinks: false },
  },
  typescript: {
    tsConfig: {
      include: ['../*.d.ts'],
      exclude: ['**/dist/**', '**/node_modules/**'],
    },
    strict: true,
  },
  vite: { plugins: [tailwindcss()] },
});
