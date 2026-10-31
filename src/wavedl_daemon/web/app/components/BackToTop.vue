<template>
  <Transition
    enter-active-class="transition-all duration-300 ease-out"
    leave-active-class="transition-all duration-300 ease-in"
    enter-from-class="opacity-0 translate-y-8 scale-75"
    leave-to-class="opacity-0 translate-y-8 scale-75"
  >
    <div
      v-if="showBackToTop"
      class="absolute bottom-5 right-5 z-20 flex items-center justify-center"
    >
      <button
        class="group relative flex size-12 items-center justify-center rounded-full bg-muted/10 backdrop-blur-md shadow-lg transition-all hover:scale-110 hover:shadow-xl focus:outline-none border border-border/50 cursor-pointer"
        :aria-label="t('shell.scrollToTop')"
        @click="scrollToTop"
      >
        <!-- Progress Ring SVG -->
        <svg class="absolute inset-0 size-full -rotate-90 transform" viewBox="0 0 36 36">
          <!-- Track -->
          <path
            class="text-muted/20"
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="currentColor"
            stroke-width="3"
          />
          <!-- Progress -->
          <path
            class="text-primary transition-all duration-100 ease-out"
            :stroke-dasharray="`${scrollProgress}, 100`"
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="currentColor"
            stroke-width="3"
            stroke-linecap="round"
          />
        </svg>

        <ArrowUp
          class="size-5 text-foreground transition-transform duration-300 group-hover:-translate-y-0.5"
        />
      </button>
    </div>
  </Transition>
</template>

<script lang="ts" setup>
import { ArrowUp } from '@lucide/vue';

const { t } = useI18n({ useScope: 'global' });
const reducedMotion = useReducedMotion();
const { scrollTarget } = useShellScroll();
const showBackToTop = ref(false);
const scrollProgress = ref(0);
let ticking = false;

const scrollToTop = () => {
  const target = scrollTarget.value;

  if (!target) return;
  scrollShellTo(target, { top: 0, behavior: reducedMotion.value ? 'instant' : 'smooth' });
};

const updateProgress = () => {
  const target = scrollTarget.value;

  if (!target) {
    showBackToTop.value = false;
    scrollProgress.value = 0;
    ticking = false;

    return;
  }

  const scrollTop = getShellScrollTop(target);
  const docHeight = getShellScrollMax(target);

  const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
  scrollProgress.value = Math.min(100, Math.max(0, progress));

  showBackToTop.value = scrollTop > 300;
  ticking = false;
};

const handleScroll = () => {
  if (!ticking) {
    window.requestAnimationFrame(updateProgress);
    ticking = true;
  }
};

useEventListener(scrollTarget, 'scroll', handleScroll, { passive: true });

onMounted(() => {
  updateProgress();
});
</script>
