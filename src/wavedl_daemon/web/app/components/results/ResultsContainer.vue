<script setup lang="ts">
import type { ConnectionSession } from '../../lib/connection-session';
import { ArtifactClient, pngPreviewBlob, type Artifact } from '../../lib/artifact-client';
import { artifactDownloadName } from '../../lib/artifact-download';
import type { RunMonitorSnapshot } from '../../lib/run-monitor';

const { isActive } = usePageActive();
const { t } = useI18n({ useScope: 'global' });
const shellLayout = useShellLayout();

const props = defineProps<{
  session: ConnectionSession;
  runId: string | null;
  monitor?: RunMonitorSnapshot | null;
}>();

const session = computed(() => toRaw(props.session));
const connected = shallowRef(false);
const scopeGeneration = shallowRef(0);
const emit = defineEmits<{ runSelected: [runId: string] }>();
const artifacts = shallowRef<readonly Artifact[]>([]);
const busy = shallowRef(false);
const message = shallowRef('');
const downloadStatus = shallowRef('');
const imageUrl = shallowRef<string | null>(null);
const previewPath = shallowRef<string | null>(null);
const panelTarget = shallowRef<HTMLElement | null>(null);
const parkedArtifacts = shallowRef<HTMLElement | null>(null);

onMounted(() => {
  parkedArtifacts.value = document.createElement('div');
  panelTarget.value = document.getElementById('shell-panel-tab-artifacts');
});

let scope = new AbortController();
let epoch = 0;

function clearImage() {
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value);
  imageUrl.value = null;
  previewPath.value = null;
}

function reset() {
  scope.abort();
  scope = new AbortController();
  epoch++;
  scopeGeneration.value++;
  artifacts.value = [];
  busy.value = false;
  message.value = '';
  downloadStatus.value = '';
  clearImage();
}

watch(
  session,
  (owner, _prior, onCleanup) => {
    reset();
    let active = true;
    let revision = owner.snapshot().revision;

    const unsubscribe = owner.subscribe((state) => {
      if (!active || owner !== session.value) return;
      connected.value = state.status === 'connected';

      if (state.status !== 'connected' || revision !== state.revision) reset();
      revision = state.revision;
    });

    onCleanup(() => {
      active = false;
      unsubscribe();
    });
  },
  { immediate: true, flush: 'sync' },
);

watch(() => props.runId, reset, { flush: 'sync' });

async function refresh() {
  if (!props.runId || busy.value) return;
  const current = epoch;
  busy.value = true;
  message.value = '';

  try {
    const items = await new ArtifactClient(session.value).list(props.runId, scope.signal);

    if (current === epoch) artifacts.value = items;
  } catch {
    if (current === epoch) message.value = 'results.artifactsFailed';
  } finally {
    if (current === epoch) busy.value = false;
  }
}

async function consume(artifact: Artifact, preview = false) {
  if (busy.value) return;
  const current = epoch;
  busy.value = true;
  message.value = '';

  try {
    const blob = await new ArtifactClient(session.value).verifiedBlob(artifact, scope.signal);

    if (current !== epoch) return;

    if (preview) {
      const image = await pngPreviewBlob(blob);

      if (current !== epoch) return;
      clearImage();
      imageUrl.value = URL.createObjectURL(image);
      previewPath.value = `${artifact.generation}/${artifact.path}`;
    } else {
      const name = await artifactDownloadName(artifact, artifacts.value);

      if (current !== epoch) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = name;
      link.click();
      downloadStatus.value = 'results.downloadRequested';
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  } catch {
    if (current === epoch)
      message.value =
        artifact.size > 64 * 1024 * 1024 ? 'results.budget' : 'results.verificationFailed';
  } finally {
    if (current === epoch) busy.value = false;
  }
}

onBeforeUnmount(() => {
  reset();
});
</script>
<template>
  <section class="flex min-w-0 flex-col gap-6" aria-labelledby="results-title">
    <div class="flex flex-wrap items-center justify-between gap-4">
      <div>
        <h2 id="results-title" class="text-xl font-semibold">
          {{ t('results.publishedResults') }}
        </h2>
        <InfoChip :label="t('results.exactIdentity')" :text="t('results.resultsHelp')" />
      </div>
      <div class="flex flex-wrap gap-2">
        <Button
          data-testid="results-refresh"
          variant="outline"
          :disabled="!runId || busy || !session.api"
          @click="refresh"
          >{{ busy ? t('results.working') : t('results.refresh') }}</Button
        ><Button
          v-if="artifacts.length"
          variant="outline"
          @click="shellLayout.setActiveTab('artifacts')"
          >{{ t('results.openArtifacts') }}</Button
        >
      </div>
    </div>
    <Alert v-if="message"
      ><AlertTitle>{{ t('results.artifactStatus') }}</AlertTitle
      ><AlertDescription>{{ t(message) }}</AlertDescription></Alert
    >
    <p v-if="downloadStatus" role="status" class="text-sm text-muted-foreground">
      {{ t(downloadStatus) }}
    </p>
    <p v-if="!runId" class="text-muted-foreground">{{ t('results.selectExperiment') }}</p>
    <EvaluationResultsPanel
      v-if="connected && monitor && monitor.run?.run_id === runId"
      :run="monitor.run"
      :events="monitor.events"
      :freshness="monitor.freshness"
    />
    <ArtifactList
      :artifacts="artifacts"
      :busy="busy"
      :image-url="imageUrl"
      :preview-path="previewPath"
      @download="consume"
      @preview="consume($event, true)"
    />
    <Teleport
      v-if="panelTarget && parkedArtifacts && artifacts.length"
      :to="isActive ? panelTarget : parkedArtifacts"
      defer
    >
      <ul class="flex flex-col gap-3" :aria-label="t('results.artifact')">
        <li
          v-for="artifact in artifacts"
          :key="`${artifact.generation}/${artifact.path}`"
          class="break-all rounded-lg border p-3 text-sm"
        >
          {{ artifact.path }}
          <p class="text-xs text-muted-foreground">
            {{ artifact.generation }} / {{ artifact.attempt }}
          </p>
          <Button variant="outline" size="sm" :disabled="busy" @click="consume(artifact)">{{
            t('results.download')
          }}</Button>
        </li>
      </ul>
    </Teleport>
    <CvSummaryPanel
      v-if="connected"
      :key="scopeGeneration"
      :session="session"
      :artifacts="artifacts"
    />
    <ResultOperationForm
      v-if="runId && artifacts.length"
      :key="`${runId}:${scopeGeneration}`"
      :session="session"
      :artifacts="artifacts"
      @run-selected="emit('runSelected', $event)"
    />
    <div v-if="imageUrl" class="flex flex-col gap-3">
      <Button variant="outline" @click="clearImage">{{ t('results.closePlot') }}</Button>
    </div>
  </section>
</template>
