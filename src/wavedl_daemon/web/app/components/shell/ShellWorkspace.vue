<template>
  <main tabindex="-1" class="flex min-h-0 min-w-0 flex-1 flex-col">
    <section class="shell-panel relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      <slot name="canvas" />
    </section>
    <slot />
  </main>
</template>
