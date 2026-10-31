declare module '#app' {
  interface PageMeta {
    title?: string;
    breadcrumb?: string;
    fullWidth?: boolean;
    fillHeight?: boolean;
  }
}

declare module 'vue-router' {
  interface RouteMeta {
    title?: string;
    breadcrumb?: string;
    fullWidth?: boolean;
  }
}

export {};
