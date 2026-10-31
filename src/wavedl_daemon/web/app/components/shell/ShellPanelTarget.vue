<script setup lang="ts">
import { shellPanelTabTargetId, type ShellPanelTabDefinition } from '../../shell/constants';

const props = defineProps<{ tab: ShellPanelTabDefinition }>();
const { t } = useI18n({ useScope: 'global' });
const target = useTemplateRef<HTMLElement>('target');
const empty = shallowRef(true);
let observer: MutationObserver | null = null;

function refresh() {
  empty.value =
    !target.value ||
    !Array.from(target.value.childNodes).some(
      (node) => node.nodeType === 1 || (node.nodeType === 3 && !!node.textContent?.trim()),
    );
}

onMounted(() => {
  refresh();
  observer = new MutationObserver(refresh);

  if (target.value) observer.observe(target.value, { childList: true, characterData: true });
});

onScopeDispose(() => observer?.disconnect());
</script>

<template>
  <div ref="target" :id="shellPanelTabTargetId(props.tab.id)" class="shell-panel-target">
    <slot />
  </div>
  <Empty v-if="empty" :data-testid="`shell-panel-empty-${props.tab.id}`" class="min-h-48">
    <EmptyHeader>
      <EmptyMedia variant="icon"><component :is="props.tab.icon" aria-hidden="true" /></EmptyMedia>
      <EmptyDescription>{{
        props.tab.id === 'logs' ? t('shell.companionLogsEmpty') : t('shell.companionArtifactsEmpty')
      }}</EmptyDescription>
    </EmptyHeader>
  </Empty>
</template>
