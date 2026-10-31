<script setup lang="ts">
import { ChevronLeft } from '@lucide/vue';

const { breadcrumbs, currentBreadcrumb, parentBreadcrumb, mobileOverflowBreadcrumbs } =
  useRouteBreadcrumbs();

const { t } = useI18n({ useScope: 'global' });
const collisionPadding = useSafeAreaCollisionPadding();
const middleBreadcrumbs = computed(() => breadcrumbs.value.slice(1, -1));
</script>

<template>
  <header class="relative flex shrink-0 items-center gap-2 px-3 py-2.5 md:gap-3 md:px-4">
    <div class="flex min-w-0 flex-1 items-center gap-2">
      <Breadcrumb>
        <BreadcrumbList class="flex-nowrap">
          <BreadcrumbItem
            v-if="breadcrumbs.length > 1 && breadcrumbs[0]"
            class="hidden md:inline-flex"
          >
            <BreadcrumbLink as-child>
              <NuxtLink :to="breadcrumbs[0].href">{{ breadcrumbs[0].title }}</NuxtLink>
            </BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator v-if="breadcrumbs.length > 1" class="hidden md:block" />

          <template v-if="parentBreadcrumb">
            <BreadcrumbItem class="md:hidden">
              <Button
                variant="ghost"
                size="sm"
                class="h-8 gap-1 px-2"
                @click="navigateTo(parentBreadcrumb.href)"
              >
                <ChevronLeft class="size-4" />
                <span class="text-xs font-medium">{{ t('shell.back') }}</span>
              </Button>
            </BreadcrumbItem>
            <BreadcrumbSeparator class="md:hidden" />
          </template>

          <template v-if="mobileOverflowBreadcrumbs.length > 0">
            <BreadcrumbItem class="md:hidden">
              <DropdownMenu>
                <DropdownMenuTrigger as-child>
                  <Button variant="ghost" size="icon" class="size-8">
                    <BreadcrumbEllipsis class="size-4" />
                    <span class="sr-only">{{ t('shell.openBreadcrumbs') }}</span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="start"
                  side="bottom"
                  :side-offset="8"
                  :collision-padding="collisionPadding"
                >
                  <DropdownMenuGroup>
                    <DropdownMenuItem
                      v-for="crumb in mobileOverflowBreadcrumbs"
                      :key="`mobile-breadcrumb-${crumb.href}`"
                      class="cursor-pointer"
                      @click="navigateTo(crumb.href)"
                    >
                      {{ crumb.title }}
                    </DropdownMenuItem>
                  </DropdownMenuGroup>
                </DropdownMenuContent>
              </DropdownMenu>
            </BreadcrumbItem>
            <BreadcrumbSeparator class="md:hidden" />
          </template>

          <template v-for="crumb in middleBreadcrumbs" :key="crumb.href">
            <BreadcrumbItem class="hidden md:inline-flex">
              <BreadcrumbLink as-child>
                <NuxtLink :to="crumb.href">{{ crumb.title }}</NuxtLink>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator class="hidden md:block" />
          </template>

          <BreadcrumbItem v-if="currentBreadcrumb" class="hidden md:inline-flex">
            <BreadcrumbPage>
              {{ currentBreadcrumb.title }}
            </BreadcrumbPage>
          </BreadcrumbItem>
          <BreadcrumbItem v-if="currentBreadcrumb" class="md:hidden">
            <BreadcrumbPage class="max-w-28 truncate">
              {{ currentBreadcrumb.title }}
            </BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>
    </div>

    <div class="ml-auto flex shrink-0 items-center gap-0.5 md:gap-2">
      <ShellPageActions />
      <slot name="title-controls" />
    </div>

    <div class="absolute inset-x-4 bottom-0 h-px bg-border/50" aria-hidden="true" />
  </header>
</template>
