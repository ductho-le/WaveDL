<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import type { Transition } from 'motion-v';
import { CornerDownLeft, FileUp, Plus } from '@lucide/vue';
import type { Button } from '@/components/ui/button';

export interface DraftSummary {
  readonly savedAt: number;
  readonly steps: readonly {
    readonly label: string;
    readonly value: string;
    readonly done: boolean;
  }[];
  readonly changes: readonly string[];
}

const props = defineProps<{ summary: DraftSummary; dragging: boolean }>();
const emit = defineEmits<{ continue: []; import: []; fresh: [] }>();
const { t, locale } = useI18n({ useScope: 'global' });
const continueButton = useTemplateRef<InstanceType<typeof Button>>('continueButton');
const now = shallowRef(Date.now());

const edited = computed(() => {
  const minutes = Math.round((props.summary.savedAt - now.value) / 60000);
  const format = new Intl.RelativeTimeFormat(locale.value, { numeric: 'auto' });

  if (Math.abs(minutes) < 60) return format.format(minutes, 'minute');

  if (Math.abs(minutes) < 60 * 24) return format.format(Math.round(minutes / 60), 'hour');

  return format.format(Math.round(minutes / 1440), 'day');
});

const spring: Transition = { type: 'spring', stiffness: 260, damping: 26 };

function keydown(event: KeyboardEvent) {
  const target = event.target;

  if (
    event.key !== 'Enter' ||
    event.defaultPrevented ||
    (target instanceof HTMLElement && target.closest('input, textarea, select, button, a'))
  )
    return;
  event.preventDefault();
  emit('continue');
}

function attach() {
  window.addEventListener('keydown', keydown);
  const element = continueButton.value?.$el;

  if (element instanceof HTMLElement) element.focus({ preventScroll: true });
}

function detach() {
  window.removeEventListener('keydown', keydown);
}

usePageActive(attach, detach);

const projectStyles = {
  startStage: cn(
    'flex justify-center p-[3rem_0_4rem]',
    '[background:radial-gradient(_60%_50%_at_50%_0%,_color-mix(in_srgb,_var(--primary)_8%,_transparent),_transparent_70%_)]',
  ),
  startGrid: cn(
    'grid w-full max-w-216 grid-cols-[minmax(0,_1.4fr)_minmax(0,_1fr)] gap-4 [@media(max-width:_44rem)]:grid-cols-1',
  ),
  startHero: cn(
    'relative overflow-hidden p-[1.6rem] rounded-[1.4rem] bg-surface',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_10%,_transparent),_0_24px_48px_-24px_rgb(0_0_0_/_0.3),_0_2px_6px_rgb(0_0_0_/_0.04)]',
    "[&::before]:[content:''] [&::before]:absolute [&::before]:[inset:0_0_auto] [&::before]:h-[3px]",
    '[&::before]:[background:linear-gradient(90deg,_#0a63d8,_#5aa9ff,_#35c46a)]',
  ),
  startEyebrow: cn('text-muted-foreground text-[0.7rem] font-[650] tracking-[0.08em] uppercase'),
  startChain: cn(
    'flex items-center m-[1.25rem_0_1rem] py-[0.9rem] px-4 rounded-[0.9rem]',
    '[background:color-mix(in_srgb,_var(--foreground)_3.5%,_transparent)]',
  ),
  startWire: cn(
    '[flex:0_0_1rem] h-0.5 my-0 mx-[0.4rem] rounded-[1px]',
    '[background:color-mix(in_srgb,_var(--foreground)_15%,_transparent)]',
  ),
  startStepLabel: cn(
    'flex items-center gap-[0.35rem] text-muted-foreground text-[0.7rem] [&_i]:size-[0.4rem] [&_i]:rounded-full',
    "[&_i]:[background:color-mix(in_srgb,_var(--foreground)_20%,_transparent)] [&_i[data-done='true']]:bg-success",
  ),
  startStepValue: cn(
    'block overflow-hidden text-[0.82rem] font-semibold text-ellipsis whitespace-nowrap',
  ),
  startChip: cn(
    'py-[0.2rem] px-[0.6rem] rounded-full text-primary text-[0.72rem] font-[550]',
    '[background:color-mix(in_srgb,_var(--primary)_11%,_transparent)]',
  ),
  startContinue: cn(
    'flex-1 h-[2.9rem] rounded-[0.9rem]',
    'shadow-[inset_0_1px_0_rgb(255_255_255_/_0.22),_0_8px_20px_-8px_color-mix(in_srgb,_var(--primary)_75%,_transparent)]',
  ),
  startKbd: cn(
    'grid w-8 h-[1.7rem] place-items-center rounded-[0.4rem] text-muted-foreground',
    '[background:color-mix(in_srgb,_var(--foreground)_5%,_transparent)]',
    'shadow-[inset_0_-1px_0_color-mix(in_srgb,_var(--foreground)_12%,_transparent)] [&_svg]:size-[0.85rem]',
  ),
  startDrop: cn(
    'flex flex-1 min-h-48 flex-col items-center justify-center gap-[0.3rem] p-5 rounded-[1.4rem]',
    '[border:1.5px_dashed_color-mix(in_srgb,_var(--foreground)_18%,_transparent)]',
    '[background:color-mix(in_srgb,_var(--surface)_55%,_transparent)] text-center',
    '[transition:border-color_180ms_ease,_background-color_180ms_ease,_transform_220ms_cubic-bezier(0.34,_1.56,_0.64,_1)]',
    '[&:hover]:[border-color:var(--primary)] [&:hover]:[background:color-mix(in_srgb,_var(--primary)_6%,_var(--surface))]',
    "[&[data-dragging='true']]:[border-color:var(--primary)]",
    "[&[data-dragging='true']]:[background:color-mix(in_srgb,_var(--primary)_6%,_var(--surface))]",
    "[&[data-dragging='true']]:[transform:scale(1.02)]",
  ),
  startDropIcon: cn(
    'grid size-11 mb-2 place-items-center rounded-[0.8rem] text-primary bg-surface',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_10%,_transparent),_0_4px_12px_-4px_rgb(0_0_0_/_0.15)]',
  ),
  startFresh: cn(
    'flex items-center gap-[0.8rem] py-[0.9rem] px-4 rounded-[1.1rem] bg-surface',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_10%,_transparent),_0_1px_2px_rgb(0_0_0_/_0.04)]',
    '[transition:transform_200ms_ease] [&:hover]:[transform:translateY(-1px)]',
  ),
  startFreshIcon: cn(
    'grid size-8 flex-none place-items-center rounded-[0.6rem]',
    '[background:color-mix(in_srgb,_var(--foreground)_6%,_transparent)]',
  ),
};
</script>
<template>
  <section :class="projectStyles.startStage" :aria-label="t('builder.startTitle')">
    <div :class="projectStyles.startGrid">
      <motion.article
        :class="projectStyles.startHero"
        :initial="{ opacity: 0, y: 18, scale: 0.98 }"
        :animate="{ opacity: 1, y: 0, scale: 1 }"
        :transition="spring"
      >
        <p :class="projectStyles.startEyebrow">{{ t('builder.draft') }}</p>
        <h2 class="mt-1.5 text-[1.6rem] font-semibold tracking-[-0.025em]">
          {{ t('builder.startTitle') }}
        </h2>
        <p class="mt-0.5 text-sm text-muted-foreground">
          {{ t('builder.editedAgo', { time: edited }) }}
        </p>

        <ol :class="projectStyles.startChain">
          <template v-for="(step, index) in summary.steps" :key="step.label">
            <li v-if="index > 0" :class="projectStyles.startWire" aria-hidden="true" />
            <li class="min-w-0 flex-1">
              <span :class="projectStyles.startStepLabel"
                ><i :data-done="step.done" aria-hidden="true" />{{ step.label }}</span
              >
              <span :class="projectStyles.startStepValue">{{ step.value }}</span>
            </li>
          </template>
        </ol>

        <div v-if="summary.changes.length" class="flex flex-wrap items-center gap-1.5">
          <span v-for="change in summary.changes" :key="change" :class="projectStyles.startChip">{{
            change
          }}</span>
          <span class="ml-0.5 text-xs text-muted-foreground">{{
            t('builder.changedFromRecipe')
          }}</span>
        </div>

        <div class="mt-6 flex items-center gap-3">
          <Button
            ref="continueButton"
            type="button"
            size="lg"
            :class="projectStyles.startContinue"
            data-testid="continue-draft"
            @click="emit('continue')"
          >
            {{ t('builder.continueDraft') }}
          </Button>
          <kbd :class="projectStyles.startKbd" :title="t('builder.pressReturn')"
            ><CornerDownLeft aria-hidden="true"
          /></kbd>
        </div>
      </motion.article>

      <div class="flex flex-col gap-3">
        <motion.button
          type="button"
          :class="projectStyles.startDrop"
          :data-dragging="dragging"
          :initial="{ opacity: 0, y: 18 }"
          :animate="{ opacity: 1, y: 0 }"
          :transition="{ ...spring, delay: 0.06 }"
          @click="emit('import')"
        >
          <span :class="projectStyles.startDropIcon"><FileUp aria-hidden="true" /></span>
          <span class="text-[0.95rem] font-medium">{{ t('builder.importConfig') }}</span>
          <span class="max-w-[26ch] text-xs text-muted-foreground">{{
            t('builder.dropHint')
          }}</span>
        </motion.button>
        <motion.button
          type="button"
          :class="projectStyles.startFresh"
          :initial="{ opacity: 0, y: 18 }"
          :animate="{ opacity: 1, y: 0 }"
          :transition="{ ...spring, delay: 0.12 }"
          data-testid="start-fresh"
          @click="emit('fresh')"
        >
          <span :class="projectStyles.startFreshIcon"><Plus aria-hidden="true" /></span>
          <span class="flex min-w-0 flex-col text-left">
            <span class="text-[0.92rem] font-medium">{{ t('builder.startFresh') }}</span>
            <span class="text-xs text-muted-foreground">{{ t('builder.startFreshHelp') }}</span>
          </span>
        </motion.button>
      </div>
    </div>
  </section>
</template>
