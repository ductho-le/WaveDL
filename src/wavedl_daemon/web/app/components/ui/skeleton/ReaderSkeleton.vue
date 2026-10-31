<script setup lang="ts">
import { cn } from '@/lib/utils';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  readerSkeleton: cn(
    "relative overflow-hidden isolate",
  ),
  readerSkeleton__shine: cn(
    "absolute inset-0 block rounded-[inherit]",
    "[background:linear-gradient(_90deg,_transparent_0%,_color-mix(in_oklch,_var(--primary)_0%,_transparent)_18%,_color-mix(in_oklch,_var(--primary)_4%,_transparent)_32%,_color-mix(in_oklch,_var(--primary)_14%,_transparent)_50%,_color-mix(in_oklch,_var(--primary)_4%,_transparent)_68%,_color-mix(in_oklch,_var(--primary)_0%,_transparent)_82%,_transparent_100%_)]",
    "shadow-[0_0_14px_color-mix(in_oklch,_var(--primary)_5%,_transparent)]",
    "[transform:translate3d(-140%,_0,_0)_skewX(-12deg)] [will-change:transform,_opacity]",
    "[animation-name:reader-skeleton-shimmer] [animation-duration:var(--reader-skeleton-duration,_2.45s)]",
    "[animation-timing-function:cubic-bezier(0.4,_0,_0.2,_1)] [animation-iteration-count:infinite]",
    "motion-reduce:animate-none motion-reduce:opacity-[0]",
  ),
  readerSkeletonPrimary: cn(
    String.raw`[&_.reader-skeleton\_\_shine]:[--reader-skeleton-duration:2.35s]`,
  ),
  readerSkeletonSecondary: cn(
    String.raw`[&_.reader-skeleton\_\_shine]:[--reader-skeleton-duration:2.55s]`,
  ),
  readerSkeletonPill: cn(
    String.raw`[&_.reader-skeleton\_\_shine]:[--reader-skeleton-duration:2.7s]`,
    String.raw`[&_.reader-skeleton\_\_shine]:opacity-[0.7]`,
  ),
  readerSkeletonCover: cn(
    String.raw`[&_.reader-skeleton\_\_shine]:[--reader-skeleton-duration:2.85s]`,
  ),
};
</script>

<template>
  <div class="w-full max-w-3xl mx-auto px-4 py-8 space-y-8">
    <!-- Story title -->
    <div class="space-y-3">
      <div :class="cn('w-2/3 rounded', styles.readerSkeleton, styles.readerSkeletonPrimary)">
        <Skeleton class="h-8 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div :class="cn('w-1/3 rounded', styles.readerSkeleton, styles.readerSkeletonSecondary)">
        <Skeleton class="h-5 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
    </div>

    <!-- Meta row: author, date, tags -->
    <div class="flex flex-wrap items-center gap-3">
      <div :class="cn('w-20 rounded-full', styles.readerSkeleton, styles.readerSkeletonPill)">
        <Skeleton class="h-6 w-full rounded-full bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div :class="cn('w-24 rounded-full', styles.readerSkeleton, styles.readerSkeletonPill)">
        <Skeleton class="h-6 w-full rounded-full bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div :class="cn('w-16 rounded-full', styles.readerSkeleton, styles.readerSkeletonPill)">
        <Skeleton class="h-6 w-full rounded-full bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div :class="cn('w-20 rounded-full', styles.readerSkeleton, styles.readerSkeletonPill)">
        <Skeleton class="h-6 w-full rounded-full bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
    </div>

    <!-- Cover image placeholder -->
    <div :class="cn('rounded-lg', styles.readerSkeleton, styles.readerSkeletonCover)">
      <Skeleton class="h-56 w-full rounded-lg bg-muted/75 animate-none sm:h-72" />
      <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
    </div>

    <!-- Section heading -->
    <div class="w-1/2 rounded relative overflow-hidden isolate">
      <Skeleton class="h-6 w-full rounded bg-muted/75 animate-none" />
      <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
    </div>

    <!-- Paragraph 1 -->
    <div class="space-y-2">
      <div class="w-full rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div class="w-full rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div class="w-full rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div class="w-4/5 rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
    </div>

    <!-- Paragraph 2 -->
    <div class="space-y-2">
      <div class="w-full rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div class="w-full rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div class="w-5/6 rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div class="w-full rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
      <div class="w-3/4 rounded relative overflow-hidden isolate">
        <Skeleton class="h-4 w-full rounded bg-muted/75 animate-none" />
        <span :class="cn('reader-skeleton__shine', styles.readerSkeleton__shine)" />
      </div>
    </div>
  </div>
</template>
