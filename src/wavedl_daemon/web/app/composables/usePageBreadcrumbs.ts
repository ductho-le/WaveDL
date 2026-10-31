import type { BreadcrumbItemType } from '../types/breadcrumbs';

interface PageBreadcrumbOverride {
  path: string;
  ownerId: string;
  items: BreadcrumbItemType[];
}

const pageBreadcrumbStateKey = 'page-breadcrumbs';

function usePageBreadcrumbState() {
  return useState<PageBreadcrumbOverride | null>(pageBreadcrumbStateKey, () => null);
}

export function useCurrentPageBreadcrumbs() {
  return usePageBreadcrumbState();
}

export function usePageBreadcrumbs(items: MaybeRefOrGetter<BreadcrumbItemType[] | undefined>) {
  const route = useRoute();
  const pageBreadcrumbs = usePageBreadcrumbState();
  const ownerId = `page-breadcrumbs-${useId()}`;
  let stopRegistration: (() => void) | null = null;
  let stopRouteClear: (() => void) | null = null;

  function clearOwnedState(path?: string) {
    if (pageBreadcrumbs.value?.ownerId !== ownerId) {
      return;
    }

    if (path && pageBreadcrumbs.value.path !== path) {
      return;
    }

    if (pageBreadcrumbs.value) {
      pageBreadcrumbs.value = null;
    }
  }

  function normalizeBreadcrumbItems(resolvedItems: BreadcrumbItemType[] | undefined) {
    return (
      resolvedItems?.reduce<BreadcrumbItemType[]>((result, item) => {
        const title = item.title.trim();
        const href = item.href.trim();

        if (title.length > 0 && href.length > 0) {
          result.push({ title, href });
        }

        return result;
      }, []) ?? []
    );
  }

  function startRegistration() {
    stopRegistration = watch(
      () => toValue(items),
      (resolvedItems) => {
        const normalizedItems = normalizeBreadcrumbItems(resolvedItems);

        if (
          normalizedItems.length > 0 &&
          normalizedItems[normalizedItems.length - 1]?.href === route.path
        ) {
          pageBreadcrumbs.value = { path: route.path, ownerId, items: normalizedItems };

          return;
        }

        clearOwnedState();
      },
      { immediate: true, deep: true },
    );
  }

  function stopOwnedRegistration() {
    if (stopRegistration) {
      stopRegistration();
      stopRegistration = null;
    }
  }

  function startRouteClear() {
    stopRouteClear = watch(
      () => route.path,
      (_path, previousPath) => {
        clearOwnedState(previousPath);
      },
      { flush: 'post' },
    );
  }

  function stopWatchers() {
    stopOwnedRegistration();

    if (stopRouteClear) {
      stopRouteClear();
      stopRouteClear = null;
    }
  }

  if (import.meta.server) {
    return pageBreadcrumbs;
  }

  onMounted(() => {
    startRegistration();
    startRouteClear();
  });

  onActivated(() => {
    stopWatchers();
    startRegistration();
    startRouteClear();
  });
  onDeactivated(() => {
    clearOwnedState();
    stopWatchers();
  });

  onScopeDispose(() => {
    clearOwnedState();
    stopWatchers();
  });

  return pageBreadcrumbs;
}
