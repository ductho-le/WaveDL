<script setup lang="ts">
import { cn } from '@/lib/utils';
import type { Artifact } from '../../lib/artifact-client';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  artifactGrid: cn(
    "[&_[data-slot='card']]:min-w-0 [&_[data-slot='card']]:max-w-full [&_[data-slot='badge']]:max-w-full",
    "[&_[data-slot='badge']]:whitespace-normal [&_[data-slot='badge']]:wrap-anywhere",
  ),
};

const { t, locale } = useI18n({ useScope: 'global' });

defineProps<{
  artifacts: readonly Artifact[];
  busy: boolean;
  imageUrl?: string | null;
  previewPath?: string | null;
}>();

defineEmits<{ download: [artifact: Artifact]; preview: [artifact: Artifact] }>();
</script>
<template>
  <section
    :aria-label="t('results.publications')"
    :class="cn('flex min-w-0 max-w-full flex-col gap-4', styles.artifactGrid)"
  >
    <Skeleton v-if="busy && !artifacts.length" class="h-32 w-full" />
    <div class="grid grid-cols-[repeat(auto-fit,minmax(min(100%,16rem),1fr))] gap-4">
      <Card
        v-for="artifact in artifacts"
        :key="`${artifact.generation}/${artifact.path}`"
        :data-artifact-generation="artifact.generation"
        :data-artifact-path="artifact.path"
        :data-artifact-attempt="artifact.attempt"
        :data-artifact-size="artifact.size"
      >
        <CardHeader
          ><CardTitle class="break-all">{{ artifact.path }}</CardTitle></CardHeader
        >
        <CardContent class="flex flex-col gap-4">
          <img
            v-if="imageUrl && previewPath === `${artifact.generation}/${artifact.path}`"
            :src="imageUrl"
            :alt="t('results.verifiedPlot')"
            class="max-h-56 w-full rounded-md object-contain"
          />
          <div class="flex flex-wrap gap-2">
            <Badge variant="outline"
              >{{ t('results.generationAttempt') }} {{ artifact.generation }} /
              {{ artifact.attempt }}</Badge
            >
            <Badge variant="secondary"
              >{{ artifact.size.toLocaleString(locale === 'vi' ? 'vi-VN' : 'en-US') }}
              {{ t('results.bytes') }}</Badge
            >
          </div>
          <div class="flex flex-wrap gap-2" role="group" :aria-label="t('results.actions')">
            <Button
              variant="outline"
              size="sm"
              :disabled="busy"
              @click="$emit('download', artifact)"
              >{{ t('results.download') }}</Button
            ><Button
              v-if="artifact.path.endsWith('.png')"
              variant="outline"
              size="sm"
              :disabled="busy"
              @click="$emit('preview', artifact)"
              >{{ t('results.viewPng') }}</Button
            >
          </div>
        </CardContent>
      </Card>
    </div>
    <Empty v-if="!artifacts.length && !busy"
      ><EmptyDescription>{{ t('results.noArtifacts') }}</EmptyDescription></Empty
    >
  </section>
</template>
