<script setup lang="ts">
import { Field as VeeField } from 'vee-validate';
import { storageMapSchema, type StorageMapping } from '../../../shared/schemas/storage-mapping';

const props = defineProps<{
  mappings?: readonly StorageMapping[];
  backend: string;
  disabled: boolean;
}>();

const emit = defineEmits<{ update: [mappings: StorageMapping[]] }>();
const { t } = useI18n({ useScope: 'global' });
const id = useId();
const mappings = computed(() => props.mappings ?? []);
const validation = computed(() => storageMapSchema.safeParse(mappings.value));
const issues = computed(() => (validation.value.success ? [] : validation.value.error.issues));

const mapErrors = computed(() =>
  issues.value.some((issue) => issue.path.length === 0) ? [t('storage.mapError')] : [],
);

function rootErrors(index: number, key: keyof StorageMapping): string[] {
  return issues.value.some((issue) => issue.path[0] === index && issue.path[1] === key)
    ? [t('storage.rootError')]
    : [];
}

function rowErrors(index: number): string[] {
  return issues.value.some((issue) => issue.path[0] === index && issue.path.length === 1)
    ? [t('storage.rowError')]
    : [];
}

function add() {
  if (props.disabled || mappings.value.length >= 32) return;
  emit('update', [...mappings.value, { host_root: '', execution_root: '' }]);
}

function update(index: number, key: keyof StorageMapping, value: string | number) {
  if (props.disabled) return;
  emit(
    'update',
    mappings.value.map((mapping, position) =>
      position === index ? { ...mapping, [key]: String(value) } : { ...mapping },
    ),
  );
}

function remove(index: number) {
  if (props.disabled) return;
  emit(
    'update',
    mappings.value.flatMap((mapping, position) => (position !== index ? [{ ...mapping }] : [])),
  );
}
</script>

<template>
  <Collapsible :default-open="mappings.length > 0" :unmount-on-hide="false">
    <CollapsibleTrigger class="flex w-full items-center justify-between text-left font-medium">
      {{ t('storage.title') }}
    </CollapsibleTrigger>
    <CollapsibleContent class="collapsible-motion">
      <FieldGroup class="mt-4">
        <FieldDescription :id="`${id}-storage-help`">{{ t('storage.guidance') }}</FieldDescription>
        <FieldDescription>{{ t('storage.interpreter') }}</FieldDescription>
        <div class="flex flex-wrap gap-2">
          <Badge variant="outline">{{ t('storage.pathRewriteOnly') }}</Badge>
          <Badge variant="outline">{{ t('storage.noUploadOrMount') }}</Badge>
          <Badge variant="outline">{{ t('storage.accessNotVerified') }}</Badge>
          <Badge variant="outline">{{ t('storage.interpreterNotMapped') }}</Badge>
          <InfoChip
            :label="t('storage.mappingDetailsLabel')"
            :text="t('storage.guidanceDetails')"
            uncertain
          />
          <InfoChip
            :label="t('storage.interpreterDetailsLabel')"
            :text="t('storage.interpreterDetails')"
            uncertain
          />
        </div>
        <Alert v-if="backend === 'local' && mappings.length">
          <AlertTitle>{{ t('storage.title') }}</AlertTitle>
          <AlertDescription>{{ t('storage.local') }}</AlertDescription>
          <div class="mt-3 flex flex-wrap gap-2">
            <Badge variant="outline">{{ t('storage.switchDoesNotClear') }}</Badge>
          </div>
        </Alert>
        <FieldDescription v-if="!mappings.length">{{ t('storage.empty') }}</FieldDescription>
        <FieldSet v-for="(mapping, index) in mappings" :key="index">
          <FieldLegend variant="label">{{ t('storage.row', { number: index + 1 }) }}</FieldLegend>
          <FieldGroup>
            <VeeField
              v-slot="{ field }"
              :name="`storage_map[${index}].host_root`"
              :keep-value="true"
            >
              <Field
                :data-invalid="!!rootErrors(index, 'host_root').length"
                :data-disabled="disabled"
              >
                <FieldLabel :for="`${id}-storage-${index}-host`">{{
                  t('storage.host')
                }}</FieldLabel>
                <Input
                  :id="`${id}-storage-${index}-host`"
                  :name="`storage_map[${index}].host_root`"
                  :model-value="mapping.host_root"
                  :disabled="disabled"
                  autocomplete="off"
                  :aria-describedby="`${id}-storage-help ${id}-storage-${index}-host-error`"
                  :aria-invalid="!!rootErrors(index, 'host_root').length"
                  @update:model-value="
                    (value) => {
                      field.onChange(value);
                      update(index, 'host_root', value);
                    }
                  "
                  @blur="field.onBlur"
                />
                <FieldError
                  :id="`${id}-storage-${index}-host-error`"
                  :errors="rootErrors(index, 'host_root')"
                />
              </Field>
            </VeeField>
            <VeeField
              v-slot="{ field }"
              :name="`storage_map[${index}].execution_root`"
              :keep-value="true"
            >
              <Field
                :data-invalid="!!rootErrors(index, 'execution_root').length"
                :data-disabled="disabled"
              >
                <FieldLabel :for="`${id}-storage-${index}-execution`">{{
                  t('storage.execution')
                }}</FieldLabel>
                <Input
                  :id="`${id}-storage-${index}-execution`"
                  :name="`storage_map[${index}].execution_root`"
                  :model-value="mapping.execution_root"
                  :disabled="disabled"
                  autocomplete="off"
                  :aria-describedby="`${id}-storage-help ${id}-storage-${index}-execution-error`"
                  :aria-invalid="!!rootErrors(index, 'execution_root').length"
                  @update:model-value="
                    (value) => {
                      field.onChange(value);
                      update(index, 'execution_root', value);
                    }
                  "
                  @blur="field.onBlur"
                />
                <FieldError
                  :id="`${id}-storage-${index}-execution-error`"
                  :errors="rootErrors(index, 'execution_root')"
                />
              </Field>
            </VeeField>
            <FieldError :errors="rowErrors(index)" />
            <Button type="button" variant="outline" :disabled="disabled" @click="remove(index)">{{
              t('storage.remove', { number: index + 1 })
            }}</Button>
          </FieldGroup>
        </FieldSet>
        <FieldError :errors="mapErrors" />
        <FieldDescription>{{ t('storage.limit') }}</FieldDescription>
        <Button
          type="button"
          variant="outline"
          :disabled="disabled || mappings.length >= 32"
          @click="add"
          >{{ t('storage.add') }}</Button
        >
      </FieldGroup>
    </CollapsibleContent>
  </Collapsible>
</template>
