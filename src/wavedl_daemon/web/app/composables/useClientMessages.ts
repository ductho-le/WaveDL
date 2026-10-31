import { isClientDiagnostic, type ClientMessage } from '../i18n/client-messages';

/** Use only at client-diagnostic render boundaries, never for daemon logs/details. */
export function useClientMessages() {
  const { t } = useI18n({ useScope: 'global' });

  return {
    clientMessage: (message: ClientMessage | Error): string => {
      if (isClientDiagnostic(message))
        return t(`clientErrors.${message.clientMessageKey}`, message.clientMessageParams ?? {});

      return message instanceof Error ? message.message : message;
    },
  };
}
