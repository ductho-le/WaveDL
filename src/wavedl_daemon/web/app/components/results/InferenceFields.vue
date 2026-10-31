<script setup lang="ts">
import { Field as VeeField } from 'vee-validate';
import { inferenceFieldMessageKeys } from '../../i18n/dynamic-keys';

const { t } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();

defineProps<{ disabled: boolean }>();

const numbers = [
  { name: 'batch_size', label: inferenceFieldMessageKeys.batchSize, min: 1, nullable: false },
  { name: 'workers', label: inferenceFieldMessageKeys.workers, min: 0, nullable: false },
  {
    name: 'input_channels',
    label: inferenceFieldMessageKeys.inputChannels,
    min: 1,
    nullable: true,
  },
  { name: 'out_size', label: inferenceFieldMessageKeys.outputSize, min: 1, nullable: true },
  { name: 'export_opset', label: inferenceFieldMessageKeys.opset, min: 11, nullable: false },
];

const texts = [
  { name: 'model', label: inferenceFieldMessageKeys.model },
  { name: 'input_key', label: inferenceFieldMessageKeys.inputKey },
  { name: 'output_key', label: inferenceFieldMessageKeys.outputKey },
];

const flags = [
  { name: 'save_predictions', label: inferenceFieldMessageKeys.predictions },
  { name: 'plot', label: inferenceFieldMessageKeys.plot },
  { name: 'verbose', label: inferenceFieldMessageKeys.verbose },
  { name: 'no_denorm', label: inferenceFieldMessageKeys.normalized },
];

function numberValue(value: string | number, nullable: boolean): number | null {
  return value === '' ? (nullable ? null : Number.NaN) : Number(value);
}

function listValue(input: string | number): string[] | null {
  const value = String(input).trim();

  return value === '' ? null : value.split(',').map((item) => item.trim());
}

function nullableText(input: string | number): string | null {
  return input === '' ? null : String(input);
}
</script>
<template>
  <FieldGroup>
    <VeeField
      v-for="item in numbers"
      :key="item.name"
      v-slot="{ field, errors }"
      :name="`config.${item.name}`"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel :for="`inference-${item.name}`">{{ t(item.label) }}</FieldLabel
        ><Input
          :id="`inference-${item.name}`"
          type="number"
          :min="item.min"
          :model-value="field.value ?? ''"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange(numberValue($event, item.nullable))" /><FieldError
          v-if="errors.length"
          :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <VeeField
      v-for="item in texts"
      :key="item.name"
      v-slot="{ field, errors }"
      :name="`config.${item.name}`"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel :for="`inference-${item.name}`">{{ t(item.label) }}</FieldLabel
        ><Input
          :id="`inference-${item.name}`"
          :model-value="field.value ?? ''"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange(nullableText($event))"
          @blur="field.onBlur" /><FieldError
          v-if="errors.length"
          :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <VeeField v-slot="{ field, errors }" name="config.format"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel for="inference-format">{{ t('results.datasetFormat') }}</FieldLabel
        ><Select
          :model-value="field.value"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange"
          ><SelectTrigger id="inference-format" :data-value="field.value"
            ><SelectValue /></SelectTrigger
          ><SelectContent
            ><SelectGroup>
              <SelectItem
                v-for="value in ['auto', 'npz', 'mat', 'hdf5']"
                :key="value"
                :value="value"
              >
                {{ value === 'auto' ? t('results.auto') : value }}
              </SelectItem></SelectGroup
            ></SelectContent
          ></Select
        ><FieldError v-if="errors.length" :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <VeeField v-slot="{ field, errors }" name="config.import_modules">
      <Field :data-invalid="!!errors.length" :data-disabled="disabled">
        <FieldLabel for="inference-import-modules">{{ t('results.importModules') }}</FieldLabel>
        <Input
          id="inference-import-modules"
          :model-value="field.value?.join(', ') ?? ''"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange(listValue($event) ?? [])"
          @blur="field.onBlur"
        />
        <InfoChip :label="t('results.trustedCode')" :text="t('results.importModulesHelp')" />
        <FieldError v-if="errors.length" :errors="errors.map(clientMessage)" />
      </Field>
    </VeeField>
    <VeeField v-slot="{ field, errors }" name="config.param_names"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel for="inference-parameters">{{ t('results.parameters') }}</FieldLabel
        ><Input
          id="inference-parameters"
          :model-value="field.value?.join(', ') ?? ''"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange(listValue($event))"
          @blur="field.onBlur" /><FieldError
          v-if="errors.length"
          :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <VeeField v-slot="{ field, errors }" name="config.plot_format"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel for="inference-plot-formats">{{ t('results.formats') }}</FieldLabel
        ><Input
          id="inference-plot-formats"
          :model-value="field.value?.join(', ') ?? ''"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange(listValue($event))"
          @blur="field.onBlur" /><InfoChip
          :label="t('results.verifiedPng')"
          :text="t('results.formatHelp')" /><FieldError
          v-if="errors.length"
          :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <VeeField
      v-for="item in flags"
      :key="item.name"
      v-slot="{ field, errors }"
      :name="`config.${item.name}`"
      ><Field orientation="horizontal" :data-invalid="!!errors.length"
        ><Checkbox
          :id="`inference-${item.name}`"
          :aria-label="t(item.label)"
          :model-value="field.value"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange($event === true)" /><FieldLabel
          :for="`inference-${item.name}`"
          >{{ t(item.label) }}</FieldLabel
        ><FieldError v-if="errors.length" :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
  </FieldGroup>
</template>
