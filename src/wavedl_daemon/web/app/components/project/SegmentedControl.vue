<script setup lang="ts" generic="Value extends string">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import type { AcceptableValue } from 'reka-ui';
import { segmentedSelectionSchema } from '../../../shared/schemas/workstation-controls';

const props = withDefaults(
  defineProps<{
    modelValue: Value | undefined;
    options: readonly {
      readonly value: Value;
      readonly label: string;
      readonly disabled?: boolean;
    }[];
    label: string;
    id?: string;
    disabled?: boolean;
    testId?: string;
    size?: 'sm' | 'md';
  }>(),
  { id: undefined, disabled: false, testId: undefined, size: 'sm' },
);

const emit = defineEmits<{ 'update:modelValue': [value: Value] }>();
const thumb = `${useId()}-thumb`;

const selectionSchema = computed(() =>
  segmentedSelectionSchema(props.options.map((item) => item.value)),
);

function choose(value: AcceptableValue | AcceptableValue[]) {
  // A single toggle group emits '' when the active item is pressed again; keep it.
  const parsed = selectionSchema.value.safeParse(value);

  if (!parsed.success) return;
  const option = props.options.find((item) => item.value === parsed.data);

  if (option) emit('update:modelValue', option.value);
}

const projectStyles = {
  segmentedControl: cn(
    'gap-0.5 p-0.5 rounded-[0.65rem] [background:color-mix(in_srgb,_var(--foreground)_6.5%,_transparent)] max-w-full',
    "overflow-x-auto [scrollbar-width:none] [&[data-size='md']_.segmented-control-item]:h-[2.2rem]",
    "[&[data-size='md']_.segmented-control-item]:px-4 [&[data-size='md']_.segmented-control-item]:text-[0.85rem]",
  ),
  segmentedControlItem: cn(
    'relative h-[1.85rem] min-w-0 rounded-sm px-[0.8rem] text-muted-foreground text-[0.8rem] font-medium',
    "[transition:color_160ms_ease] [&:hover]:text-foreground [&[data-state='on']]:text-foreground",
  ),
  segmentedControlThumb: cn(
    'absolute [inset:0] rounded-[inherit] bg-surface shadow-[0_1px_2px_rgb(0_0_0_/_0.12),_0_0_0_0.5px_rgb(0_0_0_/_0.06)]',
  ),
};
</script>
<template>
  <LayoutGroup>
    <ToggleGroup
      :id="id"
      type="single"
      :spacing="1"
      :class="cn('segmented-control', projectStyles.segmentedControl)"
      :data-size="size"
      :model-value="modelValue"
      :disabled="disabled"
      :aria-label="label"
      :data-testid="testId"
      @update:model-value="choose"
    >
      <ToggleGroupItem
        v-for="option in options"
        :key="option.value"
        :value="option.value"
        :disabled="option.disabled"
        :data-value="option.value"
        :class="
          cn(
            'segmented-control-item data-[state=on]:bg-transparent hover:bg-transparent',
            projectStyles.segmentedControlItem,
          )
        "
      >
        <motion.span
          v-if="option.value === modelValue"
          :layout-id="thumb"
          :class="projectStyles.segmentedControlThumb"
          :transition="{ type: 'spring', stiffness: 560, damping: 38 }"
          aria-hidden="true"
        />
        <span class="relative">{{ option.label }}</span>
      </ToggleGroupItem>
    </ToggleGroup>
  </LayoutGroup>
</template>
