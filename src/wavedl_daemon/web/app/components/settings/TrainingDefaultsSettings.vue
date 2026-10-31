<script setup lang="ts">
import { useForm } from 'vee-validate';
import { z } from 'zod';
import { ChevronDown } from '@lucide/vue';
import {
  loadTrainingPreferences,
  saveTrainingPreferences,
  resetTrainingPreferences,
  defaultTrainingPreferences,
  trainingPreferenceFields,
  trainingPreferencesSchema,
} from '../../lib/preferences';

const props = defineProps<{ daemonId: string | null }>();
const message = shallowRef('');

const { handleSubmit, resetForm } = useForm({
  validationSchema: z.strictObject({ config: trainingPreferencesSchema }),
  initialValues: { config: defaultTrainingPreferences() },
});

watch(
  () => props.daemonId,
  (daemonId) => {
    resetForm({
      values: {
        config: daemonId ? loadTrainingPreferences(daemonId) : defaultTrainingPreferences(),
      },
    });
    message.value = '';
  },
  { immediate: true },
);

const submit = handleSubmit((value) => {
  if (!props.daemonId) return;
  message.value = saveTrainingPreferences(
    props.daemonId,
    trainingPreferencesSchema.parse(value.config),
  )
    ? 'shell.trainingDefaultsSaved'
    : 'shell.trainingDefaultsSaveFailed';
});

function resetDefaults() {
  if (!props.daemonId) return;
  const reset = resetTrainingPreferences(props.daemonId);

  if (reset) resetForm({ values: { config: defaultTrainingPreferences() } });
  message.value = reset ? 'shell.trainingDefaultsReset' : 'shell.defaultsCouldNotBeReset';
}

const { t } = useI18n({ useScope: 'global' });
</script>
<template>
  <section aria-labelledby="defaults-heading" class="flex flex-col gap-4">
    <h2 id="defaults-heading" class="font-medium">
      {{ t('shell.newRunDefaults') }}
    </h2>
    <div class="flex items-center justify-between gap-2">
      <p class="text-sm text-muted-foreground">
        {{ t('shell.trainingDefaultsSummary') }}
      </p>
      <ShellInfo :label="t('shell.newRunDefaults')" :text="t('shell.trainingDefaultsHelp')" />
    </div>
    <p v-if="!daemonId">
      {{ t('shell.trainingDefaultsDisconnected') }}
    </p>
    <form v-else class="flex flex-col gap-4" @submit.prevent="submit">
      <Collapsible v-slot="{ open }" class="flex flex-col gap-4" :unmount-on-hide="false">
        <CollapsibleTrigger as-child>
          <Button type="button" variant="outline" class="w-full justify-between">
            {{ t('shell.trainingAndOptionalWBDefaults') }}
            <ChevronDown data-icon="inline-end" aria-hidden="true" />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent :force-mount="true">
          <FieldGroup class="mt-4" :hidden="!open" :inert="!open" :aria-hidden="!open">
            <TrainingOptionField
              v-for="name in trainingPreferenceFields"
              :key="name"
              :name="name"
              :disabled="false"
            />
          </FieldGroup>
        </CollapsibleContent>
      </Collapsible>
      <div class="flex items-center justify-between gap-2">
        <p class="text-sm text-muted-foreground">{{ t('shell.loggingSummary') }}</p>
        <ShellInfo
          :label="t('shell.trainingAndOptionalWBDefaults')"
          :text="t('shell.loggingHelp')"
        />
      </div>
      <div class="flex flex-wrap gap-2">
        <Button type="submit">{{ t('shell.saveTrainingDefaults') }}</Button
        ><Button type="button" variant="outline" @click="resetDefaults">{{
          t('shell.resetTrainingDefaults')
        }}</Button>
      </div>
      <p role="status" class="text-sm text-muted-foreground">
        {{ message ? t(message) : '' }}
      </p>
    </form>
  </section>
</template>
