<script setup lang="ts">
import { Field as VeeField } from 'vee-validate';
import {
  runtimeValueMessageKeys,
  runtimeValues,
  runtimeFieldMessageKeys,
} from '../../i18n/dynamic-keys';

const { clientMessage } = useClientMessages();
const { t } = useI18n({ useScope: 'global' });

defineProps<{ disabled: boolean }>();

const idPrefix = useId();

function controlId(name: string): string {
  return `${idPrefix}-runtime-${name.replaceAll('.', '-')}`;
}

function runtimeValueLabel(value: string): string {
  const entry = runtimeValues.find((item) => item === value);

  if (!entry) return value;
  const key = runtimeValueMessageKeys[entry];

  return key ? t(key) : value;
}

const textFields = [
  {
    name: 'runtime.python',
    label: runtimeFieldMessageKeys.pythonInterpreter,
    description: runtimeFieldMessageKeys.interpreterDescription,
  },
];

const choices = [
  { name: 'backend', label: runtimeFieldMessageKeys.backend, values: ['local', 'slurm'] },
  {
    name: 'runtime.mode',
    label: runtimeFieldMessageKeys.mode,
    values: ['existing', 'selected', 'bundled', 'container'],
  },
  {
    name: 'runtime.device',
    label: runtimeFieldMessageKeys.device,
    values: ['cpu', 'mps', 'cuda', 'auto'],
  },
];

const counts = [
  { name: 'topology.num_machines', label: runtimeFieldMessageKeys.machines, min: 1 },
  {
    name: 'topology.processes_per_machine',
    label: runtimeFieldMessageKeys.processesPerMachine,
    min: 1,
  },
  { name: 'topology.machine_rank', label: runtimeFieldMessageKeys.machineRank, min: 0 },
];
</script>
<template>
  <fieldset class="grid gap-5 sm:grid-cols-2">
    <legend class="mb-4 font-medium">{{ t('project.executionEnvironment') }}</legend>
    <VeeField
      v-for="item in textFields"
      :key="item.name"
      v-slot="{ field, errors }"
      :name="item.name"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel :for="controlId(item.name)">{{ t(item.label) }}</FieldLabel
        ><Input
          :id="controlId(item.name)"
          :name="field.name"
          :model-value="field.value"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange"
          @blur="field.onBlur"
          :aria-describedby="`${controlId(item.name)}-description`" /><FieldDescription
          :id="`${controlId(item.name)}-description`"
          >{{ t(item.description) }}</FieldDescription
        ><FieldError v-if="errors.length" :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <VeeField v-for="item in choices" :key="item.name" v-slot="{ field, errors }" :name="item.name"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel :for="controlId(item.name)">{{ t(item.label) }}</FieldLabel
        ><Select
          :model-value="String(field.value)"
          :name="field.name"
          :disabled="disabled"
          @update:model-value="field.onChange"
        >
          <SelectTrigger
            :id="controlId(item.name)"
            :data-value="String(field.value)"
            :aria-invalid="!!errors.length"
            @blur="field.onBlur"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectItem v-for="value in item.values" :key="value" :value="value">
                {{ runtimeValueLabel(value) }}
              </SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
        <FieldError v-if="errors.length" :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <VeeField v-for="item in counts" :key="item.name" v-slot="{ field, errors }" :name="item.name"
      ><Field :data-invalid="!!errors.length"
        ><FieldLabel :for="controlId(item.name)">{{ t(item.label) }}</FieldLabel
        ><Input
          :id="controlId(item.name)"
          type="number"
          step="1"
          :min="item.min"
          :model-value="field.value"
          :disabled="disabled"
          :aria-invalid="!!errors.length"
          @update:model-value="field.onChange($event === '' ? Number.NaN : Number($event))"
          @blur="field.onBlur" /><FieldError
          v-if="errors.length"
          :errors="errors.map(clientMessage)" /></Field
    ></VeeField>
    <div class="sm:col-span-2 flex flex-wrap gap-2" :aria-label="t('project.runtimeHelp')">
      <Badge variant="outline">{{ t('project.runtimeUnavailableModes') }}</Badge>
      <Badge variant="outline">{{ t('project.slurmPolicy') }}</Badge>
      <InfoChip :label="t('project.runtimeDetailsLabel')" :text="t('project.runtimeDetails')" />
    </div>
  </fieldset>
</template>
