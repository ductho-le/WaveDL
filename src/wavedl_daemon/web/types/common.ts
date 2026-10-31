

export interface BreadcrumbItemType {
  title: string;
  href: string;
}

export interface MessagePayload {
  message: string;
}

export interface StatusPayload {
  status: string;
  timestamp: number;
}

export interface DeletedPayload {
  deleted: true;
}

export type FormatDateInput = Date | number | string;

export interface FormatDateOptions extends Intl.DateTimeFormatOptions {
  fallback?: string;
}
