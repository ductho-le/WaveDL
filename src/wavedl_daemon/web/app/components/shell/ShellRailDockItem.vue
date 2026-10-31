<script setup lang="ts">
import { motion } from 'motion-v';
import type { MotionValue, SpringOptions } from 'motion-v';
import type { ShellNavigationDescriptor } from '../../shell/sections';

const { t } = useI18n({ useScope: 'global' });
const emit = defineEmits<{ select: [item: ShellNavigationDescriptor] }>();

const props = defineProps<{
  item: ShellNavigationDescriptor;
  active: boolean;
  /** Viewport-space cursor Y for the whole dock. `Infinity` parks every item at rest. */
  pointerY: MotionValue<number>;
  baseSize: number;
  magnifiedSize: number;
  falloff: number;
  spring: SpringOptions;
}>();

/**
 * A resting `size-9` control carries a `size-4` glyph, so the icon keeps 4/9 of
 * the button as the button grows. Magnifying the box on its own just leaves a
 * small icon swimming in a large circle.
 */
const ICON_RATIO = 4 / 9;

const root = useTemplateRef<HTMLElement>('root');

/**
 * Measured live instead of from a resting layout snapshot: magnified neighbours
 * push this item along the rail, so a cached centre would drift out from under
 * the cursor as the wave builds and the bulge would trail the pointer.
 */
const cursorOffset = useTransform(props.pointerY, (y) => {
  const rect = root.value?.getBoundingClientRect();

  if (!rect) return Number.POSITIVE_INFINITY;

  return y - rect.y - rect.height / 2;
});

const sizeTarget = useTransform(
  cursorOffset,
  [-props.falloff, 0, props.falloff],
  [props.baseSize, props.magnifiedSize, props.baseSize],
);

const size = useSpring(sizeTarget, props.spring);

/*
 * px strings rather than bare numbers: motion-v writes some style updates
 * straight onto `el.style`, where Vue's unitless-number handling is not there to
 * add the unit for us and the declaration is dropped silently.
 */
const sizePx = useTransform(size, (value) => `${value}px`);
const iconSizePx = useTransform(size, (value) => `${value * ICON_RATIO}px`);
</script>

<template>
  <!--
    The measured box is a plain element: a template ref on `motion.div` resolves
    to the component instance, not the node whose rect the falloff needs.
  -->
  <div ref="root" class="shrink-0">
    <motion.div class="flex items-center justify-center" :style="{ width: sizePx, height: sizePx }">
      <Tooltip>
        <TooltipTrigger as-child>
          <Button
            :variant="props.active ? 'default' : 'ghost'"
            size="icon"
            :aria-label="t(props.item.labelKey)"
            :aria-current="props.active ? 'page' : undefined"
            :data-testid="`nav-${props.item.id}`"
            @click="emit('select', props.item)"
            class="shell-rail-control size-full rounded-full"
          >
            <span
              :aria-label="t(props.item.labelKey)"
              :aria-current="props.active ? 'page' : undefined"
            >
              <motion.span
                class="flex items-center justify-center"
                :style="{ width: iconSizePx, height: iconSizePx }"
              >
                <component :is="props.item.icon" v-if="props.item.icon" class="size-full" />
              </motion.span>
            </span>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="right" :side-offset="10">{{ t(props.item.labelKey) }}</TooltipContent>
      </Tooltip>
    </motion.div>
  </div>
</template>
