<script setup lang="ts">
import { trainingFieldMessageKeys } from '../../i18n/dynamic-keys';
import { Field as VeeField } from 'vee-validate';
import type { Json } from '../../lib/daemon-api';
import { trainingFieldMetadata } from '../../../shared/schemas/training-request';

const { clientMessage } = useClientMessages();
const { t } = useI18n({ useScope: 'global' });
const props = defineProps<{ name: string; disabled: boolean }>();
const option = trainingFieldMetadata.find((item) => item.name === props.name);
const messageKeys = trainingFieldMessageKeys[props.name];
const idPrefix = useId();
const controlId = computed(() => `${idPrefix}-training-${props.name}`);
const label = computed(() => (messageKeys ? t(messageKeys.label) : props.name));
const description = computed(() => (messageKeys ? t(messageKeys.description) : props.name));

const defaultText = computed(() => {
  if (
    !option ||
    option.default === null ||
    (Array.isArray(option.default) && !option.default.length)
  )
    return t('project.none');

  return Array.isArray(option.default) ? option.default.join(', ') : String(option.default);
});

function listText(value: Json): string {
  return Array.isArray(value) ? value.join('\n') : '';
}

function listValue(value: string | number): (string | number)[] {
  if (!option || String(value).trim() === '') return [];

  return String(value)
    .split('\n')
    .map((item) =>
      option.kind === 'number' ? (item.trim() === '' ? Number.NaN : Number(item)) : item,
    );
}

function stringValue(value: string | number): string | null {
  return value === '' && option?.default === null ? null : String(value);
}
</script>
<template>
  <VeeField v-if="option" v-slot="{ field, errors }" :name="`config.${name}`" :keep-value="true">
    <Field :data-invalid="!!errors.length">
      <div
        v-if="option.kind !== 'boolean'"
        class="flex flex-wrap items-center justify-between gap-2"
      >
        <FieldLabel :for="controlId">{{ label }}</FieldLabel>
        <InfoChip
          v-if="description.length > 80"
          :label="t('project.fieldHelp')"
          :text="description"
        />
      </div>
      <Field v-if="option.kind === 'boolean'" orientation="horizontal" class="max-w-full">
        <Checkbox
          :id="controlId"
          :aria-label="label"
          :data-testid="`training-field-${name}`"
          :model-value="field.value"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange"
          @blur="field.onBlur"
        />
        <FieldLabel :for="controlId">{{ label }}</FieldLabel>
      </Field>
      <Select
        v-else-if="option.choices.length"
        :model-value="String(field.value)"
        :name="field.name"
        :disabled="disabled"
        @update:model-value="field.onChange"
      >
        <SelectTrigger
          :id="controlId"
          :data-value="String(field.value)"
          :aria-invalid="!!errors.length"
          @blur="field.onBlur"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem v-for="choice in option.choices" :key="choice" :value="choice">
              {{ choice }}
            </SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
      <Textarea
        v-else-if="option.multiple"
        :id="controlId"
        :data-testid="`training-field-${name}`"
        :model-value="listText(field.value)"
        :disabled="disabled"
        :aria-invalid="!!errors.length"
        @update:model-value="field.onChange(listValue($event))"
        @blur="field.onBlur"
      />
      <Input
        v-else-if="option.kind === 'number' || option.kind === 'integer'"
        :id="controlId"
        :data-testid="`training-field-${name}`"
        type="number"
        :step="option.kind === 'integer' ? 1 : 'any'"
        :model-value="field.value"
        :disabled="disabled"
        :aria-invalid="!!errors.length"
        @update:model-value="field.onChange($event === '' ? Number.NaN : Number($event))"
        @blur="field.onBlur"
      />
      <Input
        v-else
        :id="controlId"
        :data-testid="`training-field-${name}`"
        :model-value="field.value ?? ''"
        :disabled="disabled"
        :aria-invalid="!!errors.length"
        @update:model-value="field.onChange(stringValue($event))"
        @blur="field.onBlur"
      />
      <InfoChip
        v-if="option.kind === 'boolean' && description.length > 80"
        :label="t('project.fieldHelp')"
        :text="description"
      />
      <FieldDescription v-if="description.length <= 80">
        {{ description }}{{ option.multiple ? t('project.enterOneValuePerLine') : '' }}
      </FieldDescription>
      <div class="flex flex-wrap gap-2">
        <Badge variant="outline">{{ t('project.defaultValue', { value: defaultText }) }}</Badge>
        <Badge v-if="option.multiple" variant="secondary">
          {{ t('project.oneValuePerLine') }}
        </Badge>
      </div>
      <FieldError v-if="errors.length" :errors="errors.map(clientMessage)" />
    </Field>
  </VeeField>
</template>
