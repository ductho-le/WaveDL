<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import {
  ArrowRight,
  Check,
  ClipboardCheck,
  Cpu,
  Database,
  LoaderCircle,
  SlidersHorizontal,
  Sparkles,
} from '@lucide/vue';
import type { ProjectBlock } from '../../../shared/schemas/project-draft';

export interface PipelineBlock {
  readonly id: ProjectBlock;
  readonly label: string;
  readonly value: string;
  readonly detail: string;
  /** complete: ready; attention: usable but worth a look; pending: needs input. */
  readonly state: 'complete' | 'attention' | 'pending';
}

const props = defineProps<{
  blocks: readonly PipelineBlock[];
  active: ProjectBlock;
  panelId: string;
  reviewLabel: string;
  reviewDisabled: boolean;
  reviewBusy: boolean;
}>();

const emit = defineEmits<{ select: [block: ProjectBlock]; review: [] }>();
const { t } = useI18n({ useScope: 'global' });

const icons: Record<ProjectBlock, Component> = {
  dataset: Database,
  model: Sparkles,
  training: SlidersHorizontal,
  runtime: Cpu,
  plan: ClipboardCheck,
};

function tabId(block: ProjectBlock): string {
  return `${props.panelId}-tab-${block}`;
}

// On narrow screens the strip scrolls sideways; keep the active block in view.
watch(
  () => props.active,
  (block) =>
    void nextTick(() =>
      document
        .getElementById(tabId(block))
        ?.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' }),
    ),
);

/** Roving focus per the WAI-ARIA tabs pattern; selection follows focus. */
function move(event: KeyboardEvent, index: number) {
  const last = props.blocks.length - 1;

  const target =
    event.key === 'ArrowRight'
      ? index === last
        ? 0
        : index + 1
      : event.key === 'ArrowLeft'
        ? index === 0
          ? last
          : index - 1
        : event.key === 'Home'
          ? 0
          : event.key === 'End'
            ? last
            : null;

  const block = target === null ? undefined : props.blocks[target];

  if (target === null || !block) return;
  event.preventDefault();
  emit('select', block.id);
  void nextTick(() => document.getElementById(tabId(block.id))?.focus());
}

const projectStyles = {
  runPipeline: cn(
    'sticky top-3 z-20 flex items-center gap-3 p-2 rounded-[1.25rem]',
    '[background:color-mix(in_srgb,_var(--surface)_90%,_transparent)] [backdrop-filter:saturate(180%)_blur(24px)]',
    '[-webkit-backdrop-filter:saturate(180%)_blur(24px)]',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_10%,_transparent),_0_12px_32px_-18px_rgb(0_0_0_/_0.35)]',
    '[&::before]:absolute [&::before]:-z-1 [&::before]:top-[-0.75rem] [&::before]:right-[-1rem]',
    '[&::before]:bottom-[-1rem] [&::before]:left-[-1rem]',
    '[&::before]:[--run-pipeline-canvas:color-mix(in_srgb,_var(--card)_50%,_var(--background))]',
    "[&::before]:[background:linear-gradient(var(--run-pipeline-canvas)_70%,_transparent)] [&::before]:[content:'']",
    '[&::before]:pointer-events-none [@media(max-width:_52rem)]:flex-col [@media(max-width:_52rem)]:items-stretch',
  ),
  runPipelineTrack: cn(
    'flex flex-1 min-w-0 items-center p-[3px] m-[-3px] overflow-x-auto [scrollbar-width:none] snap-x snap-proximity',
    '[&::-webkit-scrollbar]:hidden',
  ),
  runPipelineBlock: cn(
    'relative flex flex-1 min-w-34 items-center gap-[0.65rem] py-[0.6rem] px-[0.7rem] rounded-[0.9rem] text-left',
    'snap-start [transition:background-color_160ms_ease]',
    "[&:hover:not([aria-selected='true'])]:[background:color-mix(in_srgb,_var(--foreground)_4%,_transparent)]",
    "[&:focus-visible]:[outline-offset:-2px] [&[aria-selected='true']_.run-pipeline-label]:text-primary",
    "[&[data-state='pending']_.run-pipeline-value]:text-muted-foreground",
    "[&[data-state='pending']_.run-pipeline-value]:font-medium [@media(max-width:_52rem)]:min-w-34",
  ),
  runPipelineSelection: cn(
    'absolute [inset:0] rounded-[inherit] bg-surface',
    'shadow-[0_0_0_1.5px_var(--primary),_0_6px_18px_-8px_color-mix(in_srgb,_var(--primary)_55%,_transparent)]',
  ),
  runPipelineIcon: cn(
    'relative grid flex-none size-[2.1rem] place-items-center rounded-[0.6rem] text-white',
    'shadow-[inset_0_1px_0_rgb(255_255_255_/_0.25)] [&_svg]:size-[1.05rem]',
    "[&[data-block='dataset']]:[background:linear-gradient(160deg,_#35c46a,_#1f9d55)]",
    "[&[data-block='model']]:[background:linear-gradient(160deg,_#4b9dff,_#0a63d8)]",
    "[&[data-block='training']]:[background:linear-gradient(160deg,_#ffad3b,_#f07b16)]",
    "[&[data-block='runtime']]:[background:linear-gradient(160deg,_#9d8cf6,_#6b55e0)]",
    "[&[data-block='plan']]:[background:linear-gradient(160deg,_#5f6672,_#3a3f48)]",
  ),
  runPipelineBadge: cn(
    'absolute right-[-0.3rem] bottom-[-0.3rem] grid size-4 place-items-center rounded-full',
    "[border:2px_solid_var(--surface)] bg-success [&[data-state='attention']]:bg-warning [&_svg]:size-[0.6rem]",
    '[&_svg]:stroke-[3.5]',
  ),
  runPipelineCopy: cn('relative flex min-w-0 flex-col leading-[1.25]'),
  runPipelineLabel: cn('text-muted-foreground text-[0.7rem] font-semibold tracking-[0.02em]'),
  runPipelineValue: cn(
    'overflow-hidden text-ellipsis whitespace-nowrap text-[0.9rem] font-semibold tracking-[-0.01em]',
  ),
  runPipelineLink: cn(
    "relative [flex:0_0_0.75rem] h-0.5 rounded-[1px] bg-border overflow-hidden [&::after]:[content:'']",
    '[&::after]:absolute [&::after]:[inset:0] [&::after]:bg-success [&::after]:[transform:scaleX(0)]',
    '[&::after]:origin-left [&::after]:[transition:transform_420ms_cubic-bezier(0.22,_1,_0.36,_1)]',
    "[&[data-complete='true']::after]:[transform:scaleX(1)]",
  ),
  runPipelineReview: cn(
    'flex-none h-11 rounded-[0.9rem] px-[1.1rem]',
    'shadow-[inset_0_1px_0_rgb(255_255_255_/_0.2),_0_6px_16px_-6px_color-mix(in_srgb,_var(--primary)_70%,_transparent)]',
  ),
};
</script>
<template>
  <div :class="projectStyles.runPipeline" data-testid="project-stepper">
    <LayoutGroup>
      <div
        role="tablist"
        :aria-label="t('builder.runSteps')"
        aria-orientation="horizontal"
        :class="projectStyles.runPipelineTrack"
      >
        <template v-for="(block, index) in blocks" :key="block.id">
          <span
            v-if="index > 0"
            :class="projectStyles.runPipelineLink"
            :data-complete="blocks[index - 1]?.state === 'complete'"
            aria-hidden="true"
          />
          <button
            :id="tabId(block.id)"
            type="button"
            role="tab"
            :class="projectStyles.runPipelineBlock"
            :aria-selected="active === block.id"
            :aria-controls="panelId"
            :aria-current="active === block.id ? 'step' : undefined"
            :tabindex="active === block.id ? 0 : -1"
            :data-state="block.state"
            :data-testid="`guide-step-${index}`"
            :title="`${block.value} · ${block.detail}`"
            @click="emit('select', block.id)"
            @keydown="move($event, index)"
          >
            <motion.span
              v-if="active === block.id"
              layout-id="run-pipeline-selection"
              :class="projectStyles.runPipelineSelection"
              :transition="{ type: 'spring', stiffness: 520, damping: 40 }"
              aria-hidden="true"
            />
            <span :class="projectStyles.runPipelineIcon" :data-block="block.id" aria-hidden="true">
              <component :is="icons[block.id]" />
              <motion.span
                v-if="block.state !== 'pending'"
                :key="block.state"
                :class="projectStyles.runPipelineBadge"
                :data-state="block.state"
                :initial="{ scale: 0.4, opacity: 0 }"
                :animate="{ scale: 1, opacity: 1 }"
                :transition="{ type: 'spring', stiffness: 600, damping: 24 }"
              >
                <Check v-if="block.state === 'complete'" />
              </motion.span>
            </span>
            <span :class="projectStyles.runPipelineCopy">
              <span :class="cn('run-pipeline-label', projectStyles.runPipelineLabel)">{{
                block.label
              }}</span>
              <span :class="cn('run-pipeline-value', projectStyles.runPipelineValue)">{{
                block.value
              }}</span>
              <span class="sr-only">{{ block.detail }}</span>
            </span>
          </button>
        </template>
      </div>
    </LayoutGroup>
    <Button
      type="button"
      :class="projectStyles.runPipelineReview"
      :disabled="reviewDisabled"
      @click="emit('review')"
    >
      <LoaderCircle v-if="reviewBusy" class="animate-spin" aria-hidden="true" />
      {{ reviewLabel }}
      <ArrowRight v-if="!reviewBusy" aria-hidden="true" />
    </Button>
  </div>
</template>
