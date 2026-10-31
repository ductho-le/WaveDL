<script setup lang="ts">
const { t } = useI18n({ useScope: 'global' });

const props = defineProps<{
  matrix: readonly (readonly (number | null)[])[];
  title: string;
  correlation?: boolean;
}>();

const canvas = useTemplateRef<HTMLCanvasElement>('canvas');
const zoom = shallowRef(1);
const offsetX = shallowRef(0);
const offsetY = shallowRef(0);
const hoveredCell = shallowRef<{ row: number; column: number; value: number | null } | null>(null);

const hovered = computed(() =>
  hoveredCell.value
    ? t('datasets.heatmapHover', {
        row: hoveredCell.value.row,
        column: hoveredCell.value.column,
        value: hoveredCell.value.value ?? t('datasets.missingLower'),
      })
    : '',
);

const {
  exportMessage,
  exporting,
  exportPng: requestPng,
  resetExport,
} = usePreviewExport('dataset-heatmap.png');

let origin: { x: number; y: number; offsetX: number; offsetY: number } | null = null;

function draw() {
  resetExport();
  const target = canvas.value;
  const context = target?.getContext('2d');

  if (!context || !target) return;
  const finite = props.matrix.flat().filter((value): value is number => value !== null);
  const magnitude = props.correlation ? 1 : Math.max(1, ...finite.map(Math.abs));

  const min = props.correlation ? -1 : Math.min(...finite.map((value) => value / magnitude)),
    max = props.correlation ? 1 : Math.max(...finite.map((value) => value / magnitude));

  context.clearRect(0, 0, target.width, target.height);
  const columns = props.matrix[0]?.length ?? 1;
  const cellWidth = (target.width / columns) * zoom.value;
  const cellHeight = (target.height / Math.max(1, props.matrix.length)) * zoom.value;
  props.matrix.forEach((row, y) =>
    row.forEach((value, x) => {
      const intensity =
        value === null ? 0 : max === min ? 0.5 : (value / magnitude - min) / (max - min);

      context.fillStyle = value === null ? '#d1d1d6' : `hsl(${240 - intensity * 240} 65% 55%)`;
      context.fillRect(
        offsetX.value + x * cellWidth,
        offsetY.value + y * cellHeight,
        cellWidth + 0.5,
        cellHeight + 0.5,
      );
    }),
  );
}

function reset() {
  zoom.value = 1;
  offsetX.value = 0;
  offsetY.value = 0;
  hoveredCell.value = null;
  draw();
}

function wheel(event: WheelEvent) {
  event.preventDefault();
  zoom.value = Math.max(1, Math.min(16, zoom.value * (event.deltaY < 0 ? 1.2 : 1 / 1.2)));
  draw();
}

function start(event: PointerEvent) {
  origin = { x: event.clientX, y: event.clientY, offsetX: offsetX.value, offsetY: offsetY.value };
  canvas.value?.setPointerCapture(event.pointerId);
}

function move(event: PointerEvent) {
  const target = canvas.value;

  if (!target) return;

  if (origin) {
    const bounds = target.getBoundingClientRect();
    offsetX.value = origin.offsetX + ((event.clientX - origin.x) * target.width) / bounds.width;
    offsetY.value = origin.offsetY + ((event.clientY - origin.y) * target.height) / bounds.height;
    draw();
  }

  const rect = target.getBoundingClientRect();

  const x = Math.floor(
    (((event.clientX - rect.left) * target.width) / rect.width - offsetX.value) /
      ((target.width / (props.matrix[0]?.length ?? 1)) * zoom.value),
  );

  const y = Math.floor(
    (((event.clientY - rect.top) * target.height) / rect.height - offsetY.value) /
      ((target.height / Math.max(1, props.matrix.length)) * zoom.value),
  );

  const value = props.matrix[y]?.[x];
  hoveredCell.value = value === undefined ? null : { row: y, column: x, value };
}

function end() {
  origin = null;
}

function key(event: KeyboardEvent) {
  const movements = {
    ArrowLeft: [-20, 0],
    ArrowRight: [20, 0],
    ArrowUp: [0, -20],
    ArrowDown: [0, 20],
  } satisfies Record<string, [number, number]>;

  const movement = Object.entries(movements).find(([name]) => name === event.key)?.[1];

  if (movement) {
    event.preventDefault();
    offsetX.value += movement[0];
    offsetY.value += movement[1];
    draw();
  } else if (event.key === '+' || event.key === '-' || event.key === '0') {
    event.preventDefault();

    if (event.key === '0') reset();
    else {
      zoom.value = Math.max(1, Math.min(16, zoom.value * (event.key === '+' ? 1.2 : 1 / 1.2)));
      draw();
    }
  }
}

function exportPng() {
  requestPng(canvas.value);
}

watch(
  () => [props.matrix, props.correlation],
  () => {
    reset();
  },
);

watch(() => props.title, resetExport);

onMounted(() => {
  draw();
});

onBeforeUnmount(() => {
  origin = null;
});
</script>

<template>
  <figure class="min-w-0 m-0 [&_canvas]:w-full [&_canvas]:h-80 [&_canvas]:touch-none">
    <figcaption>{{ title }}</figcaption>
    <p v-if="correlation" class="text-muted-foreground text-xs">{{ t('datasets.colorScale') }}</p>
    <canvas
      ref="canvas"
      width="800"
      height="320"
      :aria-label="title"
      role="img"
      tabindex="0"
      @keydown="key"
      @wheel="wheel"
      @pointerdown="start"
      @pointermove="move"
      @pointerup="end"
      @pointercancel="end"
      @lostpointercapture="end"
    />
    <p v-if="hovered" class="text-muted-foreground text-xs">{{ hovered }}</p>
    <InfoChip v-else :label="t('datasets.heatmapDetails')" :text="t('datasets.heatmapHelp')" />
    <p v-if="exportMessage" role="status" aria-live="polite">
      <InfoChip
        v-if="exportMessage === 'datasets.pngExportRequested'"
        :label="t('datasets.pngRequested')"
        :text="t(exportMessage)"
        uncertain
      />
      <template v-else>{{ t(exportMessage) }}</template>
    </p>
    <div class="flex gap-2">
      <Button variant="ghost" size="sm" @click="reset">{{ t('datasets.resetView') }}</Button>
      <Button variant="ghost" size="sm" :disabled="exporting" @click="exportPng">{{
        t('datasets.exportPng')
      }}</Button>
    </div>
  </figure>
</template>
