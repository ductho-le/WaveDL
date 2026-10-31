import type { ConnectionSession } from '../lib/connection-session';
import { AutoPlotController } from '../lib/autoplot-controller';
import type { PlanView, RunView } from '../lib/daemon-api';

/** Follow only a newly approved training workflow while this page owns it. */
export function useAutomaticPlots(session: ConnectionSession) {
  const controller = new AutoPlotController(session);
  const stop = () => controller.stop();
  onMounted(() => window.addEventListener('pagehide', stop));
  onScopeDispose(() => {
    controller.dispose();

    if (import.meta.client) window.removeEventListener('pagehide', stop);
  });

  return {
    state: shallowReadonly(controller.state),
    capture: (run: RunView, plan: PlanView) => controller.capture(run, plan),
    retry: () => controller.retry(),
    stop,
  };
}
