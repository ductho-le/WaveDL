<script setup lang="ts">
import { cn } from '@/lib/utils';
import { Field as VeeField, useForm } from 'vee-validate';
import { motion } from 'motion-v';
import {
  ArrowRight,
  Check,
  Eye,
  EyeOff,
  HelpCircle,
  KeyRound,
  RefreshCw,
  Terminal,
  TriangleAlert,
  Unplug,
} from '@lucide/vue';
import type { ConnectionState } from '../lib/connection-session';
import {
  daemonCredentialSchema,
  type DaemonCredentialValues,
} from '#shared/schemas/daemon-connection';

// Utility bundles keep state and descendant styling beside their component.
const styles = {
  connectBeacon: cn(
    '[--beacon:var(--primary)] relative grid size-20 place-items-center',
    "[&[data-tone='idle']]:[--beacon:var(--muted-foreground)]",
    "[&[data-tone='ready']]:[--beacon:var(--success)] [&[data-tone='warn']]:[--beacon:var(--warning)]",
  ),
  connectBeaconTile: cn(
    'relative grid size-full place-items-center rounded-[1.5rem] text-(--beacon)',
    '[background:linear-gradient(_160deg,_color-mix(in_srgb,_var(--beacon)_14%,_var(--surface)),_var(--surface)_70%_)]',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_12%,_transparent),_inset_0_1px_0_rgb(255_255_255_/_0.5),_0_18px_36px_-18px_color-mix(in_srgb,_var(--beacon)_55%,_transparent),_0_2px_6px_rgb(0_0_0_/_0.05)]',
    '[transition:color_400ms_ease,_box-shadow_400ms_ease]',
  ),
  connectBeaconBadge: cn(
    'absolute right-[-0.3rem] bottom-[-0.3rem] grid size-6 place-items-center rounded-full text-white',
    'bg-(--beacon) shadow-[0_0_0_3px_var(--background)]',
  ),
  connectInput: cn(
    'h-12 [padding-inline:2.6rem_3rem] rounded-[0.9rem] bg-surface dark:bg-surface text-[0.95rem] md:text-[0.95rem]',
    'shadow-[0_0_0_0.5px_color-mix(in_srgb,_var(--foreground)_6%,_transparent),_0_1px_2px_rgb(0_0_0_/_0.04)]',
  ),
  connectSubmit: cn(
    'h-12 rounded-[0.9rem] text-[0.95rem]',
    'shadow-[inset_0_1px_0_rgb(255_255_255_/_0.22),_0_8px_20px_-8px_color-mix(in_srgb,_var(--primary)_75%,_transparent)]',
  ),
  connectRow: cn(
    'flex min-h-[2.9rem] items-center justify-between gap-4 p-[0.65rem_1rem] text-sm',
    '[&+&]:[border-top:0.5px_solid_color-mix(in_srgb,_var(--foreground)_10%,_transparent)]!',
    '[&_dt]:shrink-0 [&_dt]:text-muted-foreground',
  ),
};

const props = withDefaults(defineProps<{ state: ConnectionState; keepSignedIn?: boolean }>(), {
  keepSignedIn: false,
});

const emit = defineEmits<{
  connect: [token: string, keepSignedIn: boolean];
  reconnect: [token: string, keepSignedIn: boolean];
  'update:keepSignedIn': [keepSignedIn: boolean];
  detach: [];
  refresh: [];
}>();

const { handleSubmit, resetForm } = useForm<DaemonCredentialValues>({
  initialValues: { token: '' },
  validationSchema: daemonCredentialSchema,
});

const { t } = useI18n({ useScope: 'global' });
const { clientMessage } = useClientMessages();
const reducedMotion = useReducedMotion();
const keepSignedIn = computed(() => props.keepSignedIn);
const revealed = shallowRef(false);
const shakes = shallowRef(0);

const busy = computed(
  () => props.state.status === 'connecting' || props.state.status === 'offline',
);

const connected = computed(() => props.state.status === 'connected');

const tone = computed<'idle' | 'busy' | 'ready' | 'warn'>(() => {
  if (connected.value) return 'ready';

  if (props.state.status === 'connecting') return 'busy';

  return props.state.message ? 'warn' : 'idle';
});

const heading = computed(() =>
  connected.value ? t('shell.connectedToYourDaemon') : t('shell.connectToYourDaemon'),
);

const subtitle = computed(() =>
  connected.value ? t('shell.connectedSubtitle') : t('shell.executionBelongsToTheDaemon'),
);

const daemonId = computed(() => props.state.health?.daemon_id ?? '');

const shortDaemonId = computed(() =>
  daemonId.value.length > 16
    ? `${daemonId.value.slice(0, 8)}…${daemonId.value.slice(-4)}`
    : daemonId.value,
);

// Rejected keys surface under the field; everything else gets a notice above it.
const rejection = computed(() =>
  props.state.status === 'authentication' && props.state.message ? props.state.message : null,
);

const notice = computed(() => (rejection.value ? null : props.state.message));
const spring = { type: 'spring', stiffness: 260, damping: 26 } as const;

const submit = handleSubmit((values) => {
  const token = values.token;
  resetForm({ values: { token: '' } });
  revealed.value = false;

  if (props.state.revision > 0) emit('reconnect', token, keepSignedIn.value);
  else emit('connect', token, keepSignedIn.value);
});

watch(
  () => props.state.revision,
  () => resetForm({ values: { token: '' } }),
);

watch(rejection, (message) => {
  if (message) shakes.value += 1;
});
</script>

<template>
  <MotionConfig reduced-motion="user">
    <section
      class="mx-auto flex w-full flex-1 items-center justify-center px-4 py-10 sm:px-8 [background:radial-gradient(_closest-side_at_50%_30%,_color-mix(in_srgb,_var(--primary)_9%,_transparent),_transparent_)]"
      aria-labelledby="connection-title"
    >
      <div class="flex w-full max-w-[25rem] flex-col items-center" data-testid="connection-card">
        <motion.div
          :class="cn(styles.connectBeacon)"
          :data-tone="tone"
          :initial="{ opacity: 0, scale: 0.8 }"
          :animate="{ opacity: 1, scale: 1 }"
          :transition="spring"
          aria-hidden="true"
        >
          <template v-if="!reducedMotion && (tone === 'busy' || tone === 'ready')">
            <motion.span
              v-for="ring in 2"
              :key="`${tone}-${ring}`"
              class="absolute inset-0 rounded-[1.5rem] [border:1.5px_solid_color-mix(in_srgb,_var(--beacon)_55%,_transparent)]!"
              :initial="{ opacity: 0.5, scale: 1 }"
              :animate="{ opacity: 0, scale: tone === 'busy' ? 1.9 : 1.6 }"
              :transition="{
                duration: tone === 'busy' ? 1.6 : 2.8,
                delay: ring * (tone === 'busy' ? 0.5 : 1.2),
                repeat: Infinity,
                ease: 'easeOut',
              }"
            />
          </template>
          <span :class="cn(styles.connectBeaconTile)">
            <img src="/favicon.svg" alt="" class="size-10 rounded-[0.75rem]" />
          </span>
          <motion.span
            v-if="tone === 'ready' || tone === 'warn'"
            :key="tone"
            :class="cn(styles.connectBeaconBadge)"
            :initial="{ opacity: 0, scale: 0.4 }"
            :animate="{ opacity: 1, scale: 1 }"
            :transition="{ ...spring, delay: 0.15 }"
          >
            <Check v-if="tone === 'ready'" class="size-3" />
            <TriangleAlert v-else class="size-3" />
          </motion.span>
        </motion.div>

        <motion.div
          :key="connected ? 'connected' : 'signin'"
          class="mt-6 flex w-full flex-col items-center"
          :initial="{ opacity: 0, y: 12 }"
          :animate="{ opacity: 1, y: 0 }"
          :transition="{ ...spring, delay: 0.05 }"
        >
          <h1
            id="connection-title"
            class="text-center text-[1.65rem] font-semibold tracking-[-0.025em]"
          >
            {{ heading }}
          </h1>
          <p class="mt-1.5 text-center text-sm text-muted-foreground">{{ subtitle }}</p>

          <motion.div
            v-if="notice"
            class="mt-6 w-full"
            :initial="{ opacity: 0, y: -6 }"
            :animate="{ opacity: 1, y: 0 }"
            :transition="spring"
          >
            <Alert role="alert" :variant="state.status === 'skew' ? 'warning' : 'default'">
              <TriangleAlert aria-hidden="true" />
              <AlertTitle v-if="state.status === 'skew'">
                {{ t('shell.connectionIdentityChanged') }}
              </AlertTitle>
              <AlertDescription>{{ clientMessage(notice) }}</AlertDescription>
            </Alert>
          </motion.div>

          <div v-if="connected && state.health" class="mt-7 flex w-full flex-col gap-4">
            <TooltipProvider>
              <dl
                class="overflow-hidden rounded-[1rem] bg-surface batch-r-surface-shadow"
                data-testid="connection-live-chip"
                :aria-label="t('shell.connectedDaemonIdentity', { daemonId })"
              >
                <div :class="cn('connect-row', styles.connectRow)">
                  <dt>{{ t('shell.connectionStateLabel') }}</dt>
                  <dd class="flex items-center gap-2 font-medium">
                    <span
                      class="size-2 rounded-full bg-success shadow-[0_0_0_3px_color-mix(in_srgb,_var(--success)_20%,_transparent)]"
                      aria-hidden="true"
                    />
                    {{ t('shell.connected') }}
                  </dd>
                </div>
                <div :class="cn('connect-row', styles.connectRow)">
                  <dt>{{ t('shell.connectionIdLabel') }}</dt>
                  <dd class="min-w-0">
                    <Tooltip>
                      <TooltipTrigger as-child>
                        <span
                          tabindex="0"
                          aria-describedby="connection-full-id"
                          class="block truncate rounded-sm font-mono text-xs text-muted-foreground outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                          >{{ shortDaemonId }}</span
                        >
                      </TooltipTrigger>
                      <TooltipContent class="max-w-xs break-all">{{
                        t('shell.fullDaemonIdentity', { daemonId })
                      }}</TooltipContent>
                    </Tooltip>
                  </dd>
                </div>
                <div :class="cn('connect-row', styles.connectRow)">
                  <dt>{{ t('shell.connectionVersionLabel') }}</dt>
                  <dd class="text-muted-foreground tabular-nums">{{ state.health.build_id }}</dd>
                </div>
              </dl>
            </TooltipProvider>
            <span id="connection-full-id" class="sr-only">{{
              t('shell.fullDaemonIdentity', { daemonId })
            }}</span>
            <div class="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="secondary"
                class="h-[2.6rem] rounded-[0.8rem]"
                @click="emit('refresh')"
              >
                <RefreshCw data-icon="inline-start" aria-hidden="true" />
                {{ t('shell.verifyConnection') }}
              </Button>
              <Button
                type="button"
                variant="secondary"
                class="h-[2.6rem] rounded-[0.8rem]"
                @click="emit('detach')"
              >
                <Unplug data-icon="inline-start" aria-hidden="true" />
                {{ t('shell.disconnect') }}
              </Button>
            </div>
          </div>

          <form v-else class="mt-7 flex w-full flex-col gap-3" @submit="submit">
            <VeeField v-slot="{ field, errors }" name="token">
              <Field :data-invalid="!!errors.length || !!rejection" :data-disabled="busy">
                <FieldLabel for="daemon-bearer" class="sr-only">
                  {{ t('shell.daemonMasterCredential') }}
                </FieldLabel>
                <div
                  :key="shakes"
                  class="relative flex items-center [&[data-shake='true']]:[animation:connect-shake_420ms_cubic-bezier(0.36,_0.07,_0.19,_0.97)]"
                  :data-shake="shakes > 0 && !reducedMotion"
                >
                  <Input
                    id="daemon-bearer"
                    v-bind="field"
                    :type="revealed ? 'text' : 'password'"
                    autocomplete="off"
                    autocapitalize="off"
                    spellcheck="false"
                    placeholder="your-access-key"
                    :aria-invalid="!!errors.length || !!rejection"
                    :disabled="busy"
                    aria-describedby="daemon-bearer-help"
                    :class="cn(styles.connectInput)"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    class="absolute right-[0.45rem] rounded-[0.6rem] text-muted-foreground"
                    :disabled="busy"
                    :aria-pressed="revealed"
                    :aria-label="revealed ? t('shell.hideAccessKey') : t('shell.showAccessKey')"
                    @click="revealed = !revealed"
                  >
                    <EyeOff v-if="revealed" aria-hidden="true" />
                    <Eye v-else aria-hidden="true" />
                  </Button>
                </div>
                <FieldError v-if="errors.length" :errors="errors.slice(0, 1).map(clientMessage)" />
                <FieldError v-else-if="rejection" :errors="[clientMessage(rejection)]" />
              </Field>
            </VeeField>
            <Button type="submit" size="lg" :class="cn(styles.connectSubmit)" :disabled="busy">
              <template v-if="state.status === 'connecting'">
                <Spinner size="sm" class="text-current" :label="t('shell.connecting2')" />
                {{ t('shell.connecting2') }}
              </template>
              <template v-else>
                {{ t('shell.connectSecurely') }}
                <ArrowRight data-icon="inline-end" aria-hidden="true" />
              </template>
            </Button>
            <Button
              v-if="state.revision > 0"
              type="button"
              variant="ghost"
              class="text-muted-foreground h-[2.6rem] rounded-[0.8rem]"
              :disabled="state.status === 'connecting'"
              @click="emit('detach')"
            >
              {{ t('shell.disconnectAndClearScope') }}
            </Button>
          </form>

          <div class="mt-4 w-full overflow-hidden rounded-[1rem] bg-surface batch-r-surface-shadow">
            <div :class="cn('connect-row', styles.connectRow)">
              <label for="keep-signed-in" class="flex min-w-0 flex-col gap-0.5">
                <span class="text-sm font-medium">{{
                  t('shell.keepMeSignedInOnThisBrowser')
                }}</span>
                <span id="keep-signed-in-help" class="text-xs text-muted-foreground">
                  {{ t('shell.sessionStorageHelp') }}
                </span>
              </label>
              <Switch
                id="keep-signed-in"
                :model-value="keepSignedIn"
                :disabled="busy"
                aria-describedby="keep-signed-in-help"
                @update:model-value="emit('update:keepSignedIn', $event)"
              />
            </div>
          </div>

          <p
            v-if="!connected"
            id="daemon-bearer-help"
            class="mt-6 flex items-start justify-center gap-2 text-center text-xs text-muted-foreground"
          >
            <Terminal class="mt-px size-3.5 shrink-0" aria-hidden="true" />
            <span>{{ t('shell.credentialHelp') }}</span>
          </p>

          <Popover v-if="state.revision > 0">
            <PopoverTrigger as-child>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                class="mt-3 rounded-full text-muted-foreground"
              >
                <HelpCircle data-icon="inline-start" aria-hidden="true" />
                {{ t('shell.disconnectHelpLabel') }}
              </Button>
            </PopoverTrigger>
            <PopoverContent class="max-w-xs text-sm" data-testid="connection-disconnect-help">{{
              t('shell.disconnectHelp')
            }}</PopoverContent>
          </Popover>
        </motion.div>
      </div>
    </section>
  </MotionConfig>
</template>
