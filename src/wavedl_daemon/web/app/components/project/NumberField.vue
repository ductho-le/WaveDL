<script setup lang="ts">
import { cn } from '@/lib/utils';
import { Minus, Plus } from '@lucide/vue';
import { compactNumber } from '../../lib/training-layout';

/**
 * Locale-independent numeric entry: native number inputs render "0,001" in some
 * locales, which reads as one thousand. Steppers follow how the value is tuned.
 */
const props = withDefaults(
  defineProps<{
    id: string;
    label: string;
    modelValue: number | null | undefined;
    integer?: boolean;
    step?: 'linear' | 'double' | 'decade';
    min?: number;
    disabled?: boolean;
    invalid?: boolean;
    testId?: string;
    describedBy?: string;
  }>(),
  { integer: false, step: 'linear', min: undefined, testId: undefined, describedBy: undefined },
);

const emit = defineEmits<{ 'update:modelValue': [value: number]; blur: [event: FocusEvent] }>();
const { t } = useI18n({ useScope: 'global' });
const focused = shallowRef(false);
const text = shallowRef('');

function format(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '';

  return props.step === 'decade' ? compactNumber(value) : String(value);
}

function parse(value: string): number {
  const trimmed = value.trim().replace(/^\+/, '');

  if (!trimmed || !/^-?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(trimmed)) return Number.NaN;

  return Number(trimmed);
}

watch(
  () => props.modelValue,
  (value) => {
    // Never rewrite what the user is typing; reformat only external changes.
    if (!focused.value || !Object.is(parse(text.value), value)) text.value = format(value);
  },
  { immediate: true },
);

function input(event: Event) {
  if (!(event.target instanceof HTMLInputElement)) return;
  text.value = event.target.value;
  emit('update:modelValue', parse(text.value));
}

function clamp(value: number): number {
  const rounded = props.integer ? Math.round(value) : Number(value.toPrecision(12));

  return props.min === undefined ? rounded : Math.max(props.min, rounded);
}

function stepped(direction: 1 | -1): number {
  const current =
    props.modelValue !== null && props.modelValue !== undefined && Number.isFinite(props.modelValue)
      ? props.modelValue
      : (props.min ?? 0);

  if (props.step === 'double') return clamp(direction > 0 ? Math.max(1, current * 2) : current / 2);

  if (props.step === 'decade' && current > 0) {
    const exponent = Math.floor(Math.log10(current) + 1e-9);
    const mantissa = current / 10 ** exponent;

    if (direction > 0) return clamp(mantissa < 2.999 ? 3 * 10 ** exponent : 10 ** (exponent + 1));

    if (mantissa > 3.001) return clamp(3 * 10 ** exponent);

    return clamp(mantissa > 1.001 ? 10 ** exponent : 3 * 10 ** (exponent - 1));
  }

  return clamp(current + direction);
}

function nudge(direction: 1 | -1) {
  if (props.disabled) return;
  const value = stepped(direction);
  text.value = format(value);
  emit('update:modelValue', value);
}

function keydown(event: KeyboardEvent) {
  if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
  event.preventDefault();
  nudge(event.key === 'ArrowUp' ? 1 : -1);
}

function blur(event: FocusEvent) {
  focused.value = false;

  if (Number.isFinite(parse(text.value))) text.value = format(parse(text.value));
  emit('blur', event);
}

const projectStyles = {
  numberField: cn(
    'inline-flex h-[2.1rem] items-stretch rounded-[0.6rem]',
    '[background:color-mix(in_srgb,_var(--foreground)_5%,_transparent)] [transition:box-shadow_150ms_ease]',
    '[&:focus-within]:shadow-[0_0_0_2px_var(--primary)]',
    '[&[data-invalid]]:shadow-[0_0_0_1.5px_var(--destructive,_#bc2836)] [&[data-disabled]]:opacity-55',
  ),
  numberFieldInput: cn(
    "w-26 min-w-0 border-0 bg-transparent [font-family:ui-monospace,_'SF_Mono',_SFMono-Regular,_Menlo,_monospace]",
    'text-[0.82rem] tabular-nums text-center outline-none',
  ),
  numberFieldStep: cn(
    'grid w-[1.9rem] place-items-center rounded-[0.6rem] text-muted-foreground',
    '[transition:color_120ms_ease,_background-color_120ms_ease] [&:hover:not(:disabled)]:text-foreground',
    '[&:hover:not(:disabled)]:[background:color-mix(in_srgb,_var(--foreground)_6%,_transparent)]',
    '[&:active:not(:disabled)]:[transform:scale(0.92)] [&_svg]:size-[0.8rem]',
  ),
};
</script>
<template>
  <div
    :class="projectStyles.numberField"
    :data-invalid="invalid || undefined"
    :data-disabled="disabled || undefined"
  >
    <button
      type="button"
      :class="projectStyles.numberFieldStep"
      tabindex="-1"
      :disabled="disabled"
      :aria-label="t('builder.decrease', { label })"
      @click="nudge(-1)"
    >
      <Minus aria-hidden="true" />
    </button>
    <input
      :id="id"
      :class="projectStyles.numberFieldInput"
      type="text"
      :inputmode="integer ? 'numeric' : 'decimal'"
      autocomplete="off"
      spellcheck="false"
      :value="text"
      :disabled="disabled"
      :aria-invalid="invalid"
      :aria-describedby="describedBy"
      :data-testid="testId"
      @focus="focused = true"
      @input="input"
      @keydown="keydown"
      @blur="blur"
    />
    <button
      type="button"
      :class="projectStyles.numberFieldStep"
      tabindex="-1"
      :disabled="disabled"
      :aria-label="t('builder.increase', { label })"
      @click="nudge(1)"
    >
      <Plus aria-hidden="true" />
    </button>
  </div>
</template>
