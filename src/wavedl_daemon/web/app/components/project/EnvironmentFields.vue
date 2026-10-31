<script setup lang="ts">
import { Field as VeeField } from 'vee-validate';
import { reservedRuntimeVariable } from '../../../shared/schemas/training-request';

const { clientMessage } = useClientMessages();
const { t } = useI18n({ useScope: 'global' });
const props = defineProps<{ environment: Record<string, string>; disabled: boolean }>();
const emit = defineEmits<{ update: [value: Record<string, string>] }>();
const name = shallowRef('');
const error = shallowRef('');

function add() {
  if (
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name.value) ||
    reservedRuntimeVariable(name.value) ||
    name.value in props.environment
  ) {
    error.value = 'project.environmentNameInvalid';

    return;
  }

  emit('update', { ...props.environment, [name.value]: '' });
  name.value = '';
  error.value = '';
}

function remove(key: string) {
  emit(
    'update',
    Object.fromEntries(Object.entries(props.environment).filter(([name]) => name !== key)),
  );
}
</script>
<template>
  <Collapsible :unmount-on-hide="false">
    <CollapsibleTrigger class="flex w-full items-center justify-between text-left font-medium">
      {{ t('project.explicitRuntimeEnvironment') }}
    </CollapsibleTrigger>
    <CollapsibleContent class="collapsible-motion">
      <div class="mt-4 flex flex-col gap-4">
        <p>{{ t('project.environmentHelp') }}</p>
        <div class="flex flex-wrap gap-2">
          <Badge variant="outline">{{ t('project.environmentSecretsStayOnHost') }}</Badge>
          <InfoChip
            :label="t('project.environmentDetailsLabel')"
            :text="t('project.environmentDetails')"
          />
        </div>
        <VeeField
          v-for="(_, key) in environment"
          :key="key"
          v-slot="{ field, errors }"
          :name="`runtime.environment.${key}`"
          :keep-value="true"
          ><Field :data-invalid="!!errors.length"
            ><FieldLabel :for="`env-${key}`">{{ key }}</FieldLabel
            ><Input
              :id="`env-${key}`"
              v-bind="field"
              :disabled="disabled"
              :aria-invalid="!!errors.length"
            /><FieldError v-if="errors.length" :errors="errors.map(clientMessage)" /><Button
              type="button"
              variant="outline"
              :disabled="disabled"
              @click="remove(key)"
              >{{ t('project.removeName', { name: key }) }}</Button
            ></Field
          ></VeeField
        >
        <Field
          ><FieldLabel for="new-env-name">{{ t('project.newEnvironmentVariable') }}</FieldLabel
          ><Input id="new-env-name" v-model="name" :disabled="disabled" :aria-invalid="!!error" />
          <p v-if="error" role="alert">{{ t(error) }}</p>
          <Button type="button" variant="outline" :disabled="disabled" @click="add">{{
            t('project.addVariable')
          }}</Button></Field
        >
      </div>
    </CollapsibleContent>
  </Collapsible>
</template>
