<script setup lang="ts">
defineProps<{ visible: boolean }>();

const { t } = useI18n({ useScope: 'global' });

function reload(): void {
  location.reload();
}
</script>

<template>
  <Alert v-if="visible" class="mx-auto mt-8 mb-0 max-w-165" role="status" aria-live="polite">
    <AlertTitle>{{ t('shell.updateAvailable') }}</AlertTitle>
    <AlertDescription>
      <Button type="button" size="sm" @click="reload">{{ t('shell.reload') }}</Button>
    </AlertDescription>
  </Alert>
</template>
