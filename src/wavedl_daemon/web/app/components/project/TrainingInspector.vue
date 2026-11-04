<script setup lang="ts">
import { cn } from '@/lib/utils';
import { useFormContext } from 'vee-validate';
import {
  Atom,
  ChartLine,
  ChevronRight,
  Code,
  CornerDownLeft,
  Database,
  Layers,
  RotateCcw,
  Save,
  Search,
  Settings2,
  Target,
  TrendingDown,
} from '@lucide/vue';
import {
  trainingFieldMetadata,
  type TrainingRequest,
} from '../../../shared/schemas/training-request';
import {
  trainingChoiceLabel,
  trainingFieldMessageKeys,
  trainingPresetMessageKeysFor,
  trainingSectionMessageKeys,
} from '../../i18n/dynamic-keys';
import {
  changedFromDefault,
  compactNumber,
  essentialTrainingFields,
  matchingPreset,
  trainingFieldVisible,
  trainingRecipes,
  trainingSections,
  type TrainingFieldName,
  type TrainingSectionId,
} from '../../lib/training-layout';

const props = defineProps<{ disabled: boolean; scope?: number }>();
const emit = defineEmits<{ preset: [id: string]; reset: [] }>();
const { t } = useI18n({ useScope: 'global' });
const { values, errors } = useFormContext<TrainingRequest>();
const searchId = useId();
const query = shallowRef('');
const open = shallowRef<ReadonlySet<TrainingSectionId>>(new Set());
const confirmReset = shallowRef(false);

// A retired connection scope cancels a pending reset confirmation.
watch(
  () => props.scope,
  () => (confirmReset.value = false),
);

const config = computed(() => values.config);
const preset = computed(() => matchingPreset(config.value));

const presetOptions = computed(() => [
  ...trainingRecipes.map((item) => {
    const keys = trainingPresetMessageKeysFor(item.id);

    return { value: item.id, label: keys ? t(keys.label) : item.id };
  }),
  ...(preset.value ? [] : [{ value: 'custom', label: t('builder.customRecipe') }]),
]);

const presetDescription = computed(() => {
  const keys = preset.value ? trainingPresetMessageKeysFor(preset.value) : null;

  return keys ? t(keys.description) : t('builder.customRecipeHelp');
});

function label(name: TrainingFieldName): string {
  const keys = trainingFieldMessageKeys[name];

  return keys ? t(keys.label) : name;
}

function formatted(name: TrainingFieldName): string {
  const value = config.value[name];

  if (value === null || value === undefined) return '—';

  if (value === true || value === false) return t(value ? 'builder.on' : 'builder.off');

  if (Array.isArray(value)) return value.length ? value.join(', ') : '—';
  const kind = trainingFieldMetadata.find((field) => field.name === name)?.kind;

  if (kind === 'integer' || kind === 'number') return compactNumber(Number(value));

  return trainingChoiceLabel(name, String(value), t);
}

const essentials = computed(() =>
  essentialTrainingFields.filter((name) => trainingFieldVisible(name, config.value)),
);

const sections = computed(() =>
  trainingSections.flatMap((section) => {
    const fields = section.fields.filter((name) => trainingFieldVisible(name, config.value));
    const changed = fields.filter((name) => changedFromDefault(name, config.value));

    return fields.length
      ? [
          {
            ...section,
            fields,
            changed,
            title: t(trainingSectionMessageKeys[section.id]),
            icon: sectionIcons[section.id],
          },
        ]
      : [];
  }),
);

const sectionIcons: Record<TrainingSectionId, Component> = {
  optimizer: Settings2,
  schedule: TrendingDown,
  loss: Target,
  data: Database,
  checkpoints: Save,
  validation: Layers,
  constraints: Atom,
  code: Code,
  logging: ChartLine,
};

// ---- Find a setting: an autocomplete that opens the setting's group and lands on it ----
const settingsRoot = useTemplateRef<HTMLElement>('settingsRoot');
const listId = `${searchId}-results`;
const searchFocused = shallowRef(false);
const activeResult = shallowRef(0);

interface SettingResult {
  readonly name: TrainingFieldName;
  readonly label: string;
  readonly group: string;
  readonly value: string;
  readonly hidden: boolean;
}

const results = computed<SettingResult[]>(() => {
  const needle = query.value.trim().toLocaleLowerCase();

  if (!needle) return [];

  // Settings the current choices ignore are still findable, but say why they are hidden.
  const candidates = [
    ...essentials.value.map((name) => ({ name, group: t('project.essentials'), hidden: false })),
    ...trainingSections.flatMap((section) =>
      section.fields.map((name) => ({
        name,
        group: t(trainingSectionMessageKeys[section.id]),
        hidden: !trainingFieldVisible(name, config.value),
      })),
    ),
  ];

  return candidates
    .flatMap(({ name, group, hidden }) => {
      const keys = trainingFieldMessageKeys[name];
      const title = label(name);
      const haystack = `${title} ${name} ${keys ? t(keys.description) : ''}`.toLocaleLowerCase();
      const lower = title.toLocaleLowerCase();

      const rank = lower.startsWith(needle)
        ? 0
        : lower.includes(needle)
          ? 1
          : haystack.includes(needle)
            ? 2
            : -1;

      const item = {
        name,
        label: title,
        group: hidden ? t('builder.notUsedNow', { group }) : group,
        value: formatted(name),
        hidden,
        rank: rank < 0 ? rank : rank + (hidden ? 3 : 0),
      };

      return item.rank >= 0 ? [item] : [];
    })
    .sort((a, b) => a.rank - b.rank)
    .slice(0, 8);
});

const resultsShown = computed(() => searchFocused.value && query.value.trim().length > 0);

watch(query, () => (activeResult.value = 0));

function resultId(index: number): string {
  return `${listId}-${index}`;
}

async function jump(name: TrainingFieldName) {
  if (!isActive.value) return;

  if (!results.value.some((item) => item.name === name && !item.hidden)) return;
  query.value = '';
  searchFocused.value = false;
  const section = sections.value.find((item) => item.fields.includes(name));

  if (section) toggle(section.id, true);
  await nextTick();
  const row = settingsRoot.value?.querySelector(`[data-field="${name}"]`);

  if (!(row instanceof HTMLElement)) return;

  // An opening group unhides its content a frame or two later; land once it is laid out.
  for (let frame = 0; frame < 10 && isActive.value && !row.getClientRects().length; frame++)
    await new Promise((resolve) => requestAnimationFrame(resolve));

  if (!isActive.value) return;
  const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  row.scrollIntoView({ block: 'center', behavior: still ? 'auto' : 'smooth' });

  const control = row.querySelector(
    'input, textarea, [role="switch"], [role="combobox"], [role="radio"]',
  );

  if (control instanceof HTMLElement) control.focus({ preventScroll: true });

  if (!still)
    row.animate?.(
      [
        { backgroundColor: 'color-mix(in srgb, var(--primary) 16%, transparent)' },
        { backgroundColor: 'transparent' },
      ],
      { duration: 1600, easing: 'ease-out' },
    );
}

const { isActive } = usePageActive();

function searchKeydown(event: KeyboardEvent) {
  const count = results.value.length;

  if (event.key === 'ArrowDown' && count) {
    event.preventDefault();
    searchFocused.value = true;
    activeResult.value = (activeResult.value + 1) % count;
  } else if (event.key === 'ArrowUp' && count) {
    event.preventDefault();
    activeResult.value = (activeResult.value - 1 + count) % count;
  } else if (event.key === 'Enter') {
    event.preventDefault();
    const result = results.value[activeResult.value];

    if (result) void jump(result.name);
  } else if (event.key === 'Escape') {
    if (query.value) event.preventDefault();
    query.value = '';
  }
}

const hiddenCount = computed(
  () =>
    trainingSections.flatMap((section) => section.fields).length -
    sections.value.reduce((total, section) => total + section.fields.length, 0),
);

/** A section holding a validation error stays open so the error is never hidden. */
function isOpen(section: { id: TrainingSectionId; fields: readonly TrainingFieldName[] }) {
  return (
    open.value.has(section.id) ||
    section.fields.some((name) => Object.hasOwn(errors.value, `config.${name}`))
  );
}

function toggle(id: TrainingSectionId, value: boolean) {
  const next = new Set(open.value);

  if (value) next.add(id);
  else next.delete(id);
  open.value = next;
}

function choosePreset(id: string) {
  if (id !== 'custom') emit('preset', id);
}

const projectStyles = {
  trainingFind: cn('relative w-[min(20rem,_100%)]'),
  trainingFindIcon: cn(
    'absolute top-[50%] left-3 size-4 text-muted-foreground pointer-events-none [transform:translateY(-50%)]',
  ),
  trainingFindPopover: cn(
    'absolute top-[calc(100%_+_0.4rem)] right-0 z-30 w-[min(24rem,_calc(100vw_-_2rem))] max-h-88 overflow-y-auto',
    'p-[0.35rem] rounded-[0.9rem] bg-popover',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_14%,_transparent),_0_18px_40px_-16px_rgb(0_0_0_/_0.35)]',
    '[animation:training-find-in_180ms_cubic-bezier(0.22,_1,_0.36,_1)]',
  ),
  trainingFindOption: cn(
    'flex items-center gap-3 py-2 px-[0.65rem] rounded-[0.6rem] cursor-default',
    "[&[aria-disabled='true']]:opacity-55",
    "[&[aria-selected='true']]:[background:color-mix(in_srgb,_var(--primary)_12%,_transparent)]",
  ),
  trainingFindValue: cn(
    "max-w-[40%] overflow-hidden text-muted-foreground [font-family:ui-monospace,_'SF_Mono',_monospace] text-[0.72rem]",
    'text-ellipsis whitespace-nowrap',
  ),
  trainingFindEnter: cn('size-[0.85rem] flex-none text-primary'),
  trainingSectionIcon: cn(
    'grid size-7 flex-none place-items-center rounded-sm text-primary',
    '[background:color-mix(in_srgb,_var(--primary)_10%,_transparent)] [&_svg]:size-[0.95rem]',
  ),
  trainingSectionTrigger: cn(
    'flex w-full items-center gap-3 py-[0.8rem] px-4 [transition:background-color_150ms_ease]',
    '[&:hover]:[background:color-mix(in_srgb,_var(--foreground)_3%,_transparent)] [&:focus-visible]:[outline-offset:-3px]',
    "[&[data-state='open']_.training-section-chevron]:[transform:rotate(90deg)]",
  ),
  trainingSectionChevron: cn(
    'size-4 flex-none text-muted-foreground [transition:transform_260ms_cubic-bezier(0.22,_1,_0.36,_1)]',
  ),
  trainingSectionCount: cn(
    'flex-none py-[0.15rem] px-[0.55rem] rounded-full text-primary text-[0.72rem] font-semibold',
    '[background:color-mix(in_srgb,_var(--primary)_12%,_transparent)]',
  ),
};
</script>
<template>
  <InspectorPanel
    :title="t('builder.trainingTitle')"
    :description="t('builder.trainingDescription')"
  >
    <template #actions>
      <div :class="projectStyles.trainingFind">
        <Search :class="projectStyles.trainingFindIcon" aria-hidden="true" />
        <Input
          :id="searchId"
          v-model="query"
          type="search"
          role="combobox"
          class="h-9 w-full rounded-xl bg-surface pl-9"
          autocomplete="off"
          aria-autocomplete="list"
          :aria-controls="listId"
          :aria-expanded="resultsShown"
          :aria-activedescendant="
            resultsShown && results.length ? resultId(activeResult) : undefined
          "
          :placeholder="t('builder.findSetting')"
          :aria-label="t('project.searchAdvancedLabel')"
          data-testid="training-advanced-search"
          @focus="searchFocused = true"
          @blur="searchFocused = false"
          @keydown="searchKeydown"
        />
        <div v-if="resultsShown" :class="projectStyles.trainingFindPopover">
          <ul
            v-if="results.length"
            :id="listId"
            role="listbox"
            :aria-label="t('builder.settingResults')"
          >
            <li
              v-for="(result, index) in results"
              :id="resultId(index)"
              :key="result.name"
              role="option"
              :class="projectStyles.trainingFindOption"
              :aria-selected="index === activeResult"
              :aria-disabled="result.hidden || undefined"
              @mousedown.prevent
              @mouseenter="activeResult = index"
              @click="jump(result.name)"
            >
              <span class="min-w-0 flex-1">
                <span class="block truncate text-[0.85rem] font-medium">{{ result.label }}</span>
                <span class="block truncate text-[0.7rem] text-muted-foreground">{{
                  result.group
                }}</span>
              </span>
              <span :class="projectStyles.trainingFindValue">{{ result.value }}</span>
              <CornerDownLeft
                v-if="index === activeResult"
                :class="projectStyles.trainingFindEnter"
                aria-hidden="true"
              />
            </li>
          </ul>
          <p
            v-else
            :id="listId"
            role="status"
            class="px-3 py-4 text-center text-xs text-muted-foreground"
          >
            {{ t('project.noAdvancedMatches') }}
          </p>
        </div>
      </div>
    </template>

    <div class="surface-group flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-4">
      <div class="min-w-52 flex-1">
        <p class="text-[0.9rem] font-medium">{{ t('builder.recipe') }}</p>
        <p class="text-xs text-muted-foreground">{{ presetDescription }}</p>
      </div>
      <SegmentedControl
        :model-value="preset ?? 'custom'"
        :options="presetOptions"
        :label="t('builder.recipe')"
        :disabled="disabled"
        size="md"
        test-id="training-preset"
        @update:model-value="choosePreset"
      />
    </div>

    <div ref="settingsRoot" class="flex flex-col gap-6">
      <section class="flex flex-col gap-2" data-testid="training-tab-essentials">
        <h3 class="px-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          {{ t('project.essentials') }}
        </h3>
        <div class="surface-group overflow-hidden">
          <TrainingSettingRow
            v-for="name in essentials"
            :key="name"
            :name="name"
            :disabled="disabled"
          />
        </div>
      </section>

      <section class="flex flex-col gap-2" data-testid="training-tab-advanced">
        <h3 class="px-1 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          {{ t('builder.moreSettings') }}
        </h3>
        <div class="flex flex-col gap-2.5">
          <Collapsible
            v-for="section in sections"
            :key="section.id"
            class="surface-group overflow-hidden"
            :open="isOpen(section)"
            :unmount-on-hide="false"
            @update:open="toggle(section.id, $event)"
          >
            <CollapsibleTrigger :class="projectStyles.trainingSectionTrigger">
              <ChevronRight
                :class="cn('training-section-chevron', projectStyles.trainingSectionChevron)"
                aria-hidden="true"
              />
              <span :class="projectStyles.trainingSectionIcon"
                ><component :is="section.icon" aria-hidden="true"
              /></span>
              <span class="min-w-0 flex-1 truncate text-left text-[0.92rem] font-medium">
                {{ section.title }}
              </span>
              <span v-if="section.changed.length" :class="projectStyles.trainingSectionCount">
                {{ t('project.nonDefaultFields', { count: section.changed.length }) }}
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent class="collapsible-motion">
              <div class="border-t border-[color-mix(in_srgb,var(--foreground)_10%,transparent)]">
                <TrainingSettingRow
                  v-for="name in section.fields"
                  :key="name"
                  :name="name"
                  :disabled="disabled"
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
        <p v-if="hiddenCount" class="px-1 text-xs text-muted-foreground">
          {{ t('builder.hiddenSettings', { count: hiddenCount }) }}
        </p>
      </section>
    </div>

    <div class="flex flex-wrap items-center gap-3">
      <Button
        v-if="!confirmReset"
        type="button"
        variant="ghost"
        size="sm"
        class="text-muted-foreground"
        :disabled="disabled"
        @click="confirmReset = true"
      >
        <RotateCcw aria-hidden="true" />{{ t('project.resetTrainingDefaults') }}
      </Button>
      <div
        v-else
        role="group"
        :aria-label="t('project.confirmReset')"
        class="flex flex-wrap items-center gap-3 rounded-xl bg-surface px-4 py-2.5 text-sm shadow-sm"
      >
        <span class="text-muted-foreground">{{ t('project.resetHelp') }}</span>
        <Button
          type="button"
          size="sm"
          variant="destructive"
          @click="
            emit('reset');
            confirmReset = false;
          "
          >{{ t('project.confirmReset') }}</Button
        >
        <Button type="button" size="sm" variant="ghost" @click="confirmReset = false">{{
          t('project.keepDraft')
        }}</Button>
      </div>
    </div>
  </InspectorPanel>
</template>
