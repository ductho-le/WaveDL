<script setup lang="ts">
import { Field as VeeField, useForm } from 'vee-validate';
import {
  appearanceSchema,
  loadAppearance,
  saveAppearance,
  applyAppearance,
  type AppearancePreferences,
} from '../../lib/preferences';
import { appearanceControlMessageKeys } from '../../i18n/dynamic-keys';

const props = defineProps<{ appearance?: AppearancePreferences }>();
const emit = defineEmits<{ changed: [value: AppearancePreferences] }>();
const { clientMessage } = useClientMessages();
const message = shallowRef('');

const { handleSubmit, resetForm } = useForm({
  validationSchema: appearanceSchema,
  initialValues: props.appearance ?? loadAppearance(),
});

let media: MediaQueryList | null = null;
let applied = props.appearance ?? loadAppearance();

function applySystem() {
  applyAppearance(applied, document.documentElement, media?.matches ?? false);
}

function controlChange(name: string, value: string, change: (value: string | number) => void) {
  change(name === 'zoom' ? Number(value) : value);
}

onMounted(() => {
  media = window.matchMedia('(prefers-color-scheme: dark)');
  media.addEventListener('change', applySystem);
});

onUnmounted(() => media?.removeEventListener('change', applySystem));

const submit = handleSubmit((value) => {
  const parsed = appearanceSchema.parse(value);
  applied = parsed;
  const saved = saveAppearance(parsed);
  applyAppearance(parsed, document.documentElement, media?.matches ?? false);
  emit('changed', parsed);
  message.value = saved ? 'shell.browserPreferencesSaved' : 'shell.appearanceSessionOnly';
});

function reload() {
  resetForm({ values: applied });
  message.value = '';
}

const controls = [
  {
    name: 'theme',
    label: appearanceControlMessageKeys.appearance,
    options: [
      { value: 'system', label: appearanceControlMessageKeys.system },
      { value: 'light', label: appearanceControlMessageKeys.light },
      { value: 'dark', label: appearanceControlMessageKeys.dark },
    ],
  },
  {
    name: 'accent',
    label: appearanceControlMessageKeys.accent,
    options: [
      { value: 'neutral', label: appearanceControlMessageKeys.neutral },
      { value: 'blue', label: appearanceControlMessageKeys.blue },
      { value: 'violet', label: appearanceControlMessageKeys.violet },
      { value: 'green', label: appearanceControlMessageKeys.green },
    ],
  },
  {
    name: 'zoom',
    label: appearanceControlMessageKeys.interfaceScale,
    options: [
      { value: 90, label: '90%' },
      { value: 100, label: '100%' },
      { value: 110, label: '110%' },
      { value: 125, label: '125%' },
    ],
  },
];

const { t } = useI18n({ useScope: 'global' });
</script>
<template>
  <section aria-labelledby="appearance-heading" class="flex flex-col gap-4">
    <div class="flex items-center justify-between gap-2">
      <h2 id="appearance-heading" class="font-medium">
        {{ t('shell.appearanceAndGuidance') }}
      </h2>
      <ShellInfo :label="t('shell.appearanceAndGuidance')" :text="t('shell.appearanceHelp')" />
    </div>
    <p class="text-sm text-muted-foreground">{{ t('shell.appearanceSummary') }}</p>
    <form class="flex flex-col gap-4" @submit.prevent="submit">
      <FieldGroup>
        <VeeField
          v-for="control in controls"
          :key="control.name"
          v-slot="{ field, errors }"
          :name="control.name"
        >
          <Field :data-invalid="!!errors.length">
            <FieldLabel :for="`preference-${control.name}`">{{ t(control.label) }}</FieldLabel>
            <Select
              :model-value="String(field.value)"
              @update:model-value="
                controlChange(control.name, String($event ?? ''), field.onChange)
              "
            >
              <SelectTrigger
                :id="`preference-${control.name}`"
                :name="field.name"
                :aria-invalid="!!errors.length"
                class="w-full"
                @blur="field.onBlur"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem
                    v-for="option in control.options"
                    :key="option.value"
                    :value="String(option.value)"
                    :data-value="String(option.value)"
                  >
                    {{ option.label.includes('.') ? t(option.label) : option.label }}
                  </SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
            <FieldError v-if="errors.length" :errors="errors.map(clientMessage)" />
          </Field>
        </VeeField>
        <VeeField v-slot="{ field, errors }" name="language">
          <Field :data-invalid="!!errors.length">
            <FieldLabel for="preference-language">{{ t('shell.language') }}</FieldLabel>
            <Select
              :model-value="String(field.value)"
              @update:model-value="controlChange('language', String($event ?? ''), field.onChange)"
            >
              <SelectTrigger
                id="preference-language"
                :name="field.name"
                :aria-invalid="!!errors.length"
                class="w-full"
                @blur="field.onBlur"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem value="en" data-value="en">English</SelectItem>
                  <SelectItem value="vi" data-value="vi">Tiếng Việt</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
            <FieldDescription>{{ t('shell.languageHelp') }}</FieldDescription>
          </Field>
        </VeeField>
      </FieldGroup>
      <div class="flex flex-wrap gap-2">
        <Button type="submit">{{ t('shell.saveAppearance') }}</Button
        ><Button type="button" variant="outline" @click="reload">{{
          t('shell.discardChanges')
        }}</Button>
      </div>
      <p role="status" class="text-sm text-muted-foreground">
        {{ message ? t(message) : '' }}
      </p>
      <p class="text-sm text-muted-foreground">
        {{ t('shell.appearanceHelp') }}
      </p>
    </form>
  </section>
</template>
