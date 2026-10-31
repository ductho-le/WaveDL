import { object, type JsonObject, type RunView } from './daemon-api';
import { toRaw } from 'vue';
import type { RunEvent } from './event-stream';

/** Only the exact current runtime's structured evaluation events are results. */
export function evaluationResults(view: RunView, events: readonly RunEvent[]): JsonObject | null {
  if (
    view.operation !== 'test' ||
    view.attempt === null ||
    view.uncertainty ||
    view.evidence?.native_pending === true
  )
    return null;
  let result: JsonObject | null = null;
  let sequence = -1;

  for (const event of events) {
    if (
      event.run_id !== view.run_id ||
      event.attempt !== view.attempt ||
      event.type !== 'metric' ||
      event.payload.name !== 'evaluation' ||
      event.seq <= sequence ||
      !object(event.payload.metrics)
    )
      continue;
    sequence = event.seq;
    result = structuredClone(toRaw(event.payload.metrics));
  }

  return result;
}
