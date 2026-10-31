<script setup lang="ts">
import type { PlanView } from '../../lib/daemon-api';

const { t } = useI18n({ useScope: 'global' });
const props = defineProps<{ plan: PlanView; busy: boolean; acknowledged?: boolean }>();
const rawDisclosureId = useId();

defineEmits<{ submit: []; status: [] }>();

const { manifest, differences, context, rawPlan } = usePlanSummary(() => props.plan);
</script>

<template>
  <section aria-labelledby="approved-plan-title" class="flex flex-col gap-5 border-t pt-6">
    <header class="flex flex-col gap-3">
      <h2 id="approved-plan-title" class="text-xl font-medium">
        {{ t('project.persistedPlanForApproval') }}
      </h2>
      <p class="font-mono text-xs text-muted-foreground">
        {{
          t('project.runRunRequestRequestPlanPlan', {
            run: plan.run_id,
            request: plan.request_id,
            plan: plan.plan_id,
          })
        }}
      </p>
      <InfoChip
        :label="t('project.planScopeChip')"
        :text="t('project.planManifestDetailsFull')"
        uncertain
      />
      <p class="text-sm text-muted-foreground">{{ t('project.planManifestHelp') }}</p>
    </header>
    <div class="flex flex-wrap gap-2" :aria-label="t('project.planContext')">
      <Badge v-for="item in context" :key="item.label" variant="outline">
        {{ item.label }}: {{ item.value }}
      </Badge>
    </div>
    <div v-if="differences.length" class="grid gap-3 sm:grid-cols-2">
      <Card v-for="item in differences" :key="item.name" class="min-w-0">
        <CardHeader class="pb-2">
          <CardTitle class="break-words text-sm">{{ item.label }}</CardTitle>
        </CardHeader>
        <CardContent class="flex flex-col gap-2 text-sm">
          <p class="break-all font-medium">{{ item.value }}</p>
          <p class="text-xs text-muted-foreground">
            {{ t('project.defaultValue', { value: item.defaultValue }) }}
          </p>
        </CardContent>
      </Card>
    </div>
    <Empty v-else>
      <EmptyDescription>{{ t('project.noPlanDifferences') }}</EmptyDescription>
    </Empty>
    <Collapsible>
      <CollapsibleTrigger
        class="flex w-full items-center justify-between text-left font-medium"
        data-testid="plan-raw-disclosure"
        :id="rawDisclosureId"
      >
        {{ t('project.fullPersistedExecutionPlan') }}
      </CollapsibleTrigger>
      <CollapsibleContent class="pt-3">
        <div
          v-if="manifest"
          class="mb-4 max-h-64 overflow-auto rounded-lg bg-muted p-4 font-mono text-xs whitespace-pre-wrap"
          tabindex="0"
          :aria-label="t('project.approvedNormalizedManifest')"
          data-testid="approved-manifest"
        >
          {{ JSON.stringify(manifest, null, 2) }}
        </div>
        <pre
          class="max-h-96 overflow-auto rounded-lg bg-muted p-4 text-xs"
          tabindex="0"
          :aria-label="t('project.persistedExecutionPlan')"
          >{{ rawPlan }}</pre>
      </CollapsibleContent>
    </Collapsible>
    <div class="flex flex-wrap gap-3">
      <Button type="button" :disabled="busy || acknowledged" @click="$emit('submit')">
        {{ t(acknowledged ? 'project.submissionAcknowledged' : 'project.approveAndStartThisPlan') }}
      </Button>
      <Button type="button" variant="outline" :disabled="busy" @click="$emit('status')">
        {{ t('project.checkSubmissionStatus') }}
      </Button>
      <Badge variant="secondary">{{ t('project.planRetryHelp') }}</Badge>
      <Badge variant="outline">{{ t('project.leaveDoesNotCancel') }}</Badge>
    </div>
  </section>
</template>
