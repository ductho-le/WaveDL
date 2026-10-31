<script setup lang="ts">
import { cn } from '@/lib/utils';
import { motion } from 'motion-v';
import {
  ArrowRight,
  Check,
  ChevronRight,
  Circle,
  LoaderCircle,
  Play,
  RefreshCw,
} from '@lucide/vue';
import type { ProjectBlock } from '../../../shared/schemas/project-draft';
import type { PlanView, RunView } from '../../lib/daemon-api';

export interface ReadinessItem {
  readonly block: ProjectBlock;
  readonly label: string;
  readonly detail: string;
  readonly done: boolean;
}

const props = defineProps<{
  plan: PlanView | null;
  acknowledged: RunView | null;
  readiness: readonly ReadinessItem[];
  canPreview: boolean;
  busy: boolean;
  connected: boolean;
  stateLabel: string;
}>();

const emit = defineEmits<{
  submit: [];
  status: [];
  open: [];
  stop: [];
  fresh: [];
  go: [block: ProjectBlock];
}>();

const { t } = useI18n({ useScope: 'global' });
const rawId = useId();
const { manifest, differences, context, rawPlan } = usePlanSummary(() => props.plan);

/** Only the exact plan the daemon acknowledged is locked; an edited draft plans anew. */
const submitted = computed(
  () => !!props.plan && props.acknowledged?.request_id === props.plan.request_id,
);

const projectStyles = {
  planSuccess: cn(
    'flex flex-col items-center gap-[0.6rem] p-[2.5rem_1.5rem_2rem] text-center',
    '[background:radial-gradient(60%_80%_at_50%_0%,_color-mix(in_srgb,_var(--success)_12%,_transparent),_transparent),_var(--surface)]',
  ),
  planSuccessMark: cn('size-17 mb-2 text-success'),
  planStat: cn(
    'py-[0.6rem] px-3 rounded-lg [background:color-mix(in_srgb,_var(--foreground)_4.5%,_transparent)]',
    '[&_dt]:text-muted-foreground [&_dt]:text-[0.7rem] [&_dd]:m-[0.15rem_0_0] [&_dd]:text-[0.92rem] [&_dd]:font-semibold',
    '[&_dd]:[overflow-wrap:anywhere]',
  ),
  planDifferences: cn(
    'flex flex-col p-[0_0.5rem_0.75rem] [&_li]:flex [&_li]:items-center [&_li]:gap-[0.6rem] [&_li]:py-[0.45rem]',
    '[&_li]:px-3 [&_li]:rounded-[0.6rem] [&_li]:text-[0.85rem]',
    '[&_li:nth-child(odd)]:[background:color-mix(in_srgb,_var(--foreground)_3%,_transparent)]',
  ),
  planDefault: cn(
    "max-w-[30%] overflow-hidden text-muted-foreground [font-family:ui-monospace,_'SF_Mono',_monospace] text-xs/[inherit]",
    'line-through text-ellipsis whitespace-nowrap',
  ),
  planValue: cn(
    "max-w-[40%] overflow-hidden [font-family:ui-monospace,_'SF_Mono',_monospace] text-[0.78rem] font-semibold",
    'text-ellipsis whitespace-nowrap',
  ),
  planRawTrigger: cn(
    'flex w-full items-center gap-[0.6rem] py-[0.8rem] px-5 text-[0.85rem] font-medium',
    "[&[data-state='open']_.plan-raw-chevron]:[transform:rotate(90deg)]",
  ),
  planRawChevron: cn(
    'size-4 text-muted-foreground [transition:transform_260ms_cubic-bezier(0.22,_1,_0.36,_1)]',
  ),
  planCode: cn(
    "max-h-72 overflow-auto py-[0.9rem] px-4 rounded-lg [font-family:ui-monospace,_'SF_Mono',_monospace] text-[0.72rem]",
    'whitespace-pre-wrap [background:color-mix(in_srgb,_var(--foreground)_5%,_transparent)]',
  ),
  planApprove: cn(
    'min-w-48 h-[2.9rem] rounded-[0.9rem]',
    'shadow-[inset_0_1px_0_rgb(255_255_255_/_0.2),_0_8px_20px_-10px_color-mix(in_srgb,_var(--primary)_80%,_transparent)]',
  ),
  planCheck: cn(
    '[&_+_.plan-check]:[border-top:0.5px_solid_color-mix(in_srgb,_var(--foreground)_12%,_transparent)]',
  ),
  planCheckMark: cn(
    "grid size-[1.4rem] flex-none place-items-center rounded-full text-muted-foreground [&[data-done='true']]:text-white",
    "[&[data-done='true']]:bg-success [&_svg]:size-[0.8rem] [&_svg]:stroke-[3] [&[data-done='false']_svg]:size-[1.2rem]",
    "[&[data-done='false']_svg]:stroke-[1.5]",
  ),
  planCta: cn('flex flex-wrap items-center gap-4 py-[1.1rem] px-5'),
};
</script>
<template>
  <InspectorPanel :title="t('builder.planTitle')" :description="t('builder.planDescription')">
    <!-- Acknowledged: the daemon accepted this exact plan. -->
    <section
      v-if="submitted && acknowledged"
      :class="cn('surface-group', projectStyles.planSuccess)"
      :aria-label="t('project.acknowledgedTrainingRun')"
    >
      <svg :class="projectStyles.planSuccessMark" viewBox="0 0 64 64" aria-hidden="true">
        <motion.circle
          cx="32"
          cy="32"
          r="29"
          fill="none"
          stroke="currentColor"
          stroke-width="3"
          :initial="{ pathLength: 0, opacity: 0.2 }"
          :animate="{ pathLength: 1, opacity: 1 }"
          :transition="{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }"
        />
        <motion.path
          d="M20 33.5 28.5 42 44.5 24"
          fill="none"
          stroke="currentColor"
          stroke-width="4"
          stroke-linecap="round"
          stroke-linejoin="round"
          :initial="{ pathLength: 0 }"
          :animate="{ pathLength: 1 }"
          :transition="{ duration: 0.4, delay: 0.45, ease: 'easeOut' }"
        />
      </svg>
      <h3 class="text-[1.4rem] font-semibold tracking-[-0.02em]">
        {{ t('project.daemonAcknowledgedTheRun') }}
      </h3>
      <p class="max-w-[52ch] text-sm text-muted-foreground">
        {{ t('project.submissionAckHelp', { run: acknowledged.run_id, state: stateLabel }) }}
      </p>
      <div class="mt-2 flex flex-wrap justify-center gap-2">
        <Button
          type="button"
          size="lg"
          class="rounded-xl"
          :disabled="!connected"
          @click="emit('open')"
        >
          {{ t('project.openAcknowledgedRun') }}<ArrowRight aria-hidden="true" />
        </Button>
        <Button type="button" size="lg" variant="ghost" class="rounded-xl" @click="emit('fresh')">
          {{ t('builder.startAnother') }}
        </Button>
      </div>
    </section>

    <!-- Planned: review what the daemon will run, then approve. -->
    <template v-else-if="plan">
      <section class="surface-group overflow-hidden" aria-labelledby="approved-plan-title">
        <header class="flex flex-wrap items-start justify-between gap-4 px-5 pt-5">
          <div class="min-w-0">
            <p class="flex items-center gap-1.5 text-xs font-semibold text-success">
              <Check class="size-3.5" aria-hidden="true" />{{ t('builder.planReady') }}
            </p>
            <h3 id="approved-plan-title" class="mt-1 text-[1.15rem] font-semibold">
              {{ t('project.persistedPlanForApproval') }}
            </h3>
            <p class="mt-1 font-mono text-[0.68rem] break-all text-muted-foreground">
              {{
                t('project.runRunRequestRequestPlanPlan', {
                  run: plan.run_id,
                  request: plan.request_id,
                  plan: plan.plan_id,
                })
              }}
            </p>
          </div>
        </header>
        <dl
          class="grid grid-cols-2 gap-2 px-5 pt-4 sm:grid-cols-4"
          :aria-label="t('project.planContext')"
        >
          <div v-for="item in context" :key="item.label" :class="projectStyles.planStat">
            <dt>{{ item.label }}</dt>
            <dd>{{ item.value }}</dd>
          </div>
        </dl>
        <div class="px-5 pt-5 pb-2">
          <h4 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {{ t('builder.changesFromDefaults') }}
          </h4>
        </div>
        <ul v-if="differences.length" :class="projectStyles.planDifferences">
          <li v-for="item in differences" :key="item.name">
            <span class="min-w-0 flex-1 truncate">{{ item.label }}</span>
            <span :class="projectStyles.planDefault">{{ item.defaultValue }}</span>
            <ArrowRight class="size-3.5 flex-none text-muted-foreground" aria-hidden="true" />
            <span :class="projectStyles.planValue">{{ item.value }}</span>
          </li>
        </ul>
        <p v-else class="px-5 pb-4 text-sm text-muted-foreground">
          {{ t('project.noPlanDifferences') }}
        </p>
        <Collapsible class="border-t border-[color-mix(in_srgb,var(--foreground)_10%,transparent)]">
          <CollapsibleTrigger
            :id="rawId"
            :class="projectStyles.planRawTrigger"
            data-testid="plan-raw-disclosure"
          >
            <ChevronRight
              :class="cn('plan-raw-chevron', projectStyles.planRawChevron)"
              aria-hidden="true"
            />
            {{ t('project.fullPersistedExecutionPlan') }}
          </CollapsibleTrigger>
          <CollapsibleContent class="collapsible-motion">
            <div class="flex flex-col gap-3 px-5 pb-5">
              <div
                v-if="manifest"
                :class="projectStyles.planCode"
                tabindex="0"
                :aria-label="t('project.approvedNormalizedManifest')"
                data-testid="approved-manifest"
              >
                {{ JSON.stringify(manifest, null, 2) }}
              </div>
              <pre
                :class="projectStyles.planCode"
                tabindex="0"
                :aria-label="t('project.persistedExecutionPlan')"
                >{{ rawPlan }}</pre>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </section>
      <div class="flex flex-wrap items-center gap-3">
        <Button
          type="button"
          size="lg"
          :class="projectStyles.planApprove"
          :disabled="busy || !connected || submitted"
          @click="emit('submit')"
        >
          <LoaderCircle v-if="busy" class="animate-spin" aria-hidden="true" />
          <Play v-else aria-hidden="true" />
          {{ t(submitted ? 'project.submissionAcknowledged' : 'project.approveAndStartThisPlan') }}
        </Button>
        <Button type="button" variant="ghost" :disabled="busy" @click="emit('status')">
          <RefreshCw aria-hidden="true" />{{ t('project.checkSubmissionStatus') }}
        </Button>
        <Button v-if="busy" type="button" variant="ghost" @click="emit('stop')">
          {{ t('project.stopWaiting') }}
        </Button>
      </div>
      <p class="text-xs text-muted-foreground">
        {{ t('project.planRetryHelp') }} {{ t('project.leaveDoesNotCancel') }}
      </p>
    </template>

    <!-- Not planned yet: a readiness checklist and the planning action. -->
    <template v-else>
      <div
        v-if="acknowledged"
        class="surface-group surface-row"
        :aria-label="t('project.acknowledgedTrainingRun')"
      >
        <span :class="projectStyles.planCheckMark" data-done="true" aria-hidden="true"
          ><Check
        /></span>
        <span class="min-w-0 flex-1">
          <span class="block text-[0.9rem] font-medium">{{
            t('project.daemonAcknowledgedTheRun')
          }}</span>
          <span class="block truncate font-mono text-xs text-muted-foreground">{{
            acknowledged.run_id
          }}</span>
        </span>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          class="text-primary"
          :disabled="!connected"
          @click="emit('open')"
        >
          {{ t('project.openAcknowledgedRun') }}
        </Button>
      </div>
      <ul class="surface-group overflow-hidden" :aria-label="t('builder.readiness')">
        <li
          v-for="item in readiness"
          :key="item.block"
          :class="cn('plan-check surface-row', projectStyles.planCheck)"
        >
          <span :class="projectStyles.planCheckMark" :data-done="item.done" aria-hidden="true">
            <Check v-if="item.done" />
            <Circle v-else />
          </span>
          <span class="min-w-0 flex-1">
            <span class="block text-[0.9rem] font-medium">{{ item.label }}</span>
            <span class="block truncate text-xs text-muted-foreground">{{ item.detail }}</span>
          </span>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            class="text-primary"
            @click="emit('go', item.block)"
          >
            {{ item.done ? t('builder.edit') : t('builder.fix') }}
          </Button>
        </li>
      </ul>
      <div :class="cn('surface-group', projectStyles.planCta)">
        <div class="min-w-0 flex-1">
          <p class="text-[1rem] font-semibold">
            {{ canPreview ? t('builder.readyToPlan') : t('builder.notReady') }}
          </p>
          <p class="text-xs text-muted-foreground">{{ t('project.noAutomaticStart') }}</p>
        </div>
        <Button v-if="busy" type="button" variant="ghost" @click="emit('stop')">
          {{ t('project.stopWaiting') }}
        </Button>
        <Button type="submit" size="lg" :class="projectStyles.planApprove" :disabled="!canPreview">
          <LoaderCircle v-if="busy" class="animate-spin" aria-hidden="true" />
          {{ t(busy ? 'project.waitingForDaemon' : 'builder.createPlan') }}
        </Button>
      </div>
    </template>
  </InspectorPanel>
</template>
