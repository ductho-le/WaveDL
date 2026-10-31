import type { Component } from 'vue';

/**
 * A navigation tree node, declared structurally rather than imported from
 * `@nuxt/content`.
 *
 * Helpers that only walk the tree need its shape, not the content module — so
 * declaring it here is what lets them stay in place when the module is not
 * installed. `ContentNavigationItem` is assignable to this.
 */
export interface NavigationItem {
  title: string;
  path: string;
  stem?: string;
  children?: NavigationItem[];
  page?: false;
  [key: string]: unknown;
}

/**
 * A badge declared on a navigation entry's frontmatter. `variant` is narrowed to
 * what the Badge component accepts; anything else is dropped when read.
 */
export interface NavigationBadge {
  value: string;
  variant?: 'default' | 'secondary' | 'destructive' | 'success' | 'warning' | 'outline';
  type?: string;
  size?: string;
}

export interface DefaultConfig {
  site: {
    name: string;
    description: string;
    ogImage: string;
    ogImageComponent: string;
    ogImageColor: 'dark' | 'light';
    umami: {
      enable: boolean;
      src: string;
      dataWebsiteId: string;
    };
  };
  theme: {
    customizable: boolean;
    color: Color;
    radius: number;
  };
  banner: {
    enable: boolean;
    showClose: boolean;
    content: string;
    to: string;
    target: Target;
    border: boolean;
  };
  header: {
    showLoadingIndicator: boolean;
    title: string;
    showTitle: boolean;
    border: boolean;
    logo: {
      light: string;
      dark: string;
    };
    showTitleInMobile: boolean;
    darkModeToggle: boolean;
    languageSwitcher: {
      enable: boolean;
      triggerType: 'icon' | 'text';
      dropdownType: 'select' | 'popover';
    };
    nav: {
      title: string;
      to: string;
      target: string;
      showLinkIcon: boolean;
      links: {
        title?: string;
        to?: string;
        target?: string;
        description?: string;
        icon?: string;
      }[];
    }[];
    links: {
      icon: string;
      to: string;
      target: string;
    }[];
  };
  aside: {
    useLevel: boolean;
    levelStyle: 'aside' | 'header';
    headerLevelNavAlign: 'start' | 'center' | 'end';
    collapse: boolean;
    collapseLevel: number;
    folderStyle: 'default' | 'tree' | 'group';
  };
  main: {
    breadCrumb: boolean;
    showTitle: boolean;
    codeCopyToast: boolean;
    codeCopyToastText: string;
    fieldRequiredText: string;
    padded: boolean;
    editLink: {
      enable: boolean;
      pattern: string;
      text: string;
      icon: string;
      placement: ('docsFooter' | 'toc')[];
    };
    issueLink: {
      enable: boolean;
      pattern: string;
      text: string;
      icon: string;
      placement: ('docsFooter' | 'toc')[];
    };
    backToTop: boolean;
    pm: ('npm' | 'pnpm' | 'bun' | 'yarn')[];
    imageZoom: boolean;
  };
  footer: {
    credits: string;
    border: boolean;
    links: {
      icon: string;
      title: string;
      to: string;
      target: string;
    }[];
  };
  toc: {
    enable: boolean;
    enableInMobile: boolean;
    enableInHomepage: boolean;
    progressBar: boolean;
    title: string;
    links: {
      title: string;
      icon: string;
      to: string;
      target: string;
      showLinkIcon: boolean;
    }[];
    iconLinks: {
      icon: string;
      to: string;
      target: string;
    }[];
    carbonAds: {
      enable: boolean;
      disableInDev: boolean;
      disableInMobile: boolean;
      fallback: boolean;
      fallbackMessage: string;
      code: string;
      placement: string;
      format: 'cover' | 'responsive';
    };
  };
  search: {
    enable: boolean;
    inAside: boolean;
    style: 'input' | 'button';
    placeholder: string;
    placeholderDetailed: string;
  };
  data?: {
    [key: string]: unknown;
  };
}

export type Color =
  | 'zinc'
  | 'slate'
  | 'stone'
  | 'gray'
  | 'neutral'
  | 'red'
  | 'rose'
  | 'orange'
  | 'green'
  | 'blue'
  | 'yellow'
  | 'violet';

export type Target = '_blank' | '_parent' | '_self' | '_top' | (string & object) | null | undefined;

export type FileTreeItemDiff = 'addition' | 'deletion' | 'none';
export interface FileTreeItem {
  title: string;
  icon?: Component | string;
  children?: FileTreeItem[];
  highlighted: boolean;
  diff: FileTreeItemDiff;
  path?: string;
}

export type InputTreeItem =
  | string
  | {
      [key: string]: InputTreeItem[];
    };
