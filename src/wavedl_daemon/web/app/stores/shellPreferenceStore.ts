import { loadShellDockSlots, saveShellDockSlots } from '../lib/preferences';
import { DEFAULT_SHELL_DOCK_SLOTS, normalizeShellDockSlots, type ShellDockSlots } from '../constants/shellDock';

export const useShellPreferencesStore = defineStore('shellPreferences', () => {
  const storedDockSlots = shallowRef(loadShellDockSlots());
  const dockSlots = computed(() => storedDockSlots.value);
  const saveFailed = shallowRef(false);

  function updateSlots(next: ShellDockSlots): void {
    storedDockSlots.value = normalizeShellDockSlots(next);
    saveFailed.value = !saveShellDockSlots(storedDockSlots.value);
  }

  function reset(): void {
    updateSlots(DEFAULT_SHELL_DOCK_SLOTS);
  }

  return { dockSlots, saveFailed, updateSlots, reset };
});
