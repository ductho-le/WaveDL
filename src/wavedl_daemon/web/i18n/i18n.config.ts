import { messages } from '../app/i18n/messages';

export default defineI18nConfig(() => ({
  legacy: false,
  globalInjection: false,
  locale: 'en',
  fallbackLocale: 'en',
  messages,
}));
