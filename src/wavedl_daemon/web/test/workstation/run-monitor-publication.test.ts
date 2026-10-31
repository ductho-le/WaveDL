import { afterEach, expect, it, vi } from 'vitest';
import { RunMonitor, type RunMonitorSnapshot, type RunTransport } from '../../app/lib/run-monitor';
import type { JsonObject, RunView } from '../../app/lib/daemon-api';

const id = '11111111-1111-1111-1111-111111111111';

function view(patch: Partial<RunView> = {}): RunView {
  return {
    plan_id: id,
    run_id: id,
    request_id: id,
    backend: 'local',
    operation: 'train',
    attempt: 1,
    state: 'running',
    uncertainty: false,
    submission_requested: true,
    cancellation_requested: false,
    evidence: null,
    ...patch,
  };
}

function frame(
  seq: number,
  type = 'metric',
  attempt = 1,
  payload: JsonObject = {
    name: 'loss',
    value: seq,
    step: seq,
    metrics: { epoch: seq, total_epochs: 100000, train_loss: seq, val_loss: seq },
  },
) {
  return `id: ${attempt}:${seq}\ndata: ${JSON.stringify({ protocol: 'wavedl-jsonl', version: 1, run_id: id, attempt, seq, type, ts: '2026-10-07T00:00:00Z', payload })}\n\n`;
}

async function flush() {
  for (let i = 0; i < 40; i++) await Promise.resolve();
}

afterEach(() => vi.useRealTimers());

it('same-authority refresh retains buffered live events and pending scope clears them', async () => {
  vi.useFakeTimers();
  let current = view();
  let channel: ReadableStreamDefaultController<Uint8Array> | undefined;

  const transport: RunTransport = {
    getRun: async () => current,
    cancelRun: async () => view(),
    openEvents: async (_id, _cursor, signal) =>
      new Response(
        new ReadableStream({
          start(controller) {
            channel = controller;
            signal.addEventListener('abort', () => controller.close(), { once: true });
          },
        }),
      ),
  };

  const monitor = new RunMonitor(id, transport, 100);
  const task = monitor.start();
  await flush();
  channel?.enqueue(new TextEncoder().encode(frame(1) + frame(2)));
  await flush();
  await vi.advanceTimersByTimeAsync(100);
  expect(monitor.snapshot().liveEvents.map((e) => e.seq)).toEqual([1, 2]);
  channel?.enqueue(new TextEncoder().encode(frame(3)));
  await flush();
  current = view({
    evidence: {
      native_pending: true,
      native_identity: {
        run_id: id,
        request_id: id,
        cluster: 'test',
        job_id: '123',
        owner: 'tester',
        restart_count: 1,
      },
    },
  });
  await vi.advanceTimersByTimeAsync(100);
  expect(monitor.snapshot().metrics).toEqual([]);
  expect(monitor.snapshot().liveEvents).toEqual([]);
  expect(monitor.snapshot().summary.latest).toBeNull();
  await vi.advanceTimersByTimeAsync(250);
  expect(monitor.snapshot().metrics).toEqual([]);
  expect(monitor.snapshot().liveEvents).toEqual([]);
  monitor.stop();
  await task;
  expect(vi.getTimerCount()).toBe(0);
});

it('trailing publication stays bounded after wall-clock rollback', async () => {
  vi.useFakeTimers();
  let channel: ReadableStreamDefaultController<Uint8Array> | undefined;

  const transport: RunTransport = {
    getRun: async () => view(),
    cancelRun: async () => view(),
    openEvents: async (_id, _cursor, signal) =>
      new Response(
        new ReadableStream({
          start(controller) {
            channel = controller;
            signal.addEventListener('abort', () => controller.close(), { once: true });
          },
        }),
      ),
  };

  const monitor = new RunMonitor(id, transport, 10000);
  const seen: RunMonitorSnapshot[] = [];
  monitor.subscribe((s) => seen.push(s));
  const task = monitor.start();
  await flush();
  channel?.enqueue(new TextEncoder().encode(frame(1)));
  await flush();
  vi.setSystemTime(Date.now() - 5000);
  channel?.enqueue(new TextEncoder().encode(frame(2)));
  await flush();
  const prior = seen.length;
  await vi.advanceTimersByTimeAsync(249);
  expect(seen).toHaveLength(prior);
  await vi.advanceTimersByTimeAsync(1);
  expect(seen.at(-1)?.metrics.map((r) => r.step)).toEqual([1, 2]);
  monitor.stop();
  await task;
  expect(vi.getTimerCount()).toBe(0);
});

it('commits all 100k frames and exact final cursor while coalescing subscriber work', async () => {
  vi.useFakeTimers();

  const source = new TextEncoder().encode(
    Array.from({ length: 100000 }, (_, i) => frame(i + 1)).join(''),
  );

  let calls = 0;

  const transport: RunTransport = {
    getRun: async () => view({ state: calls++ ? 'completed' : 'running' }),
    openEvents: async () => new Response(source),
    cancelRun: async () => view(),
  };

  const monitor = new RunMonitor(id, transport, 2000);
  let published = 0;
  let last: RunMonitorSnapshot | null = null;
  monitor.subscribe((value) => {
    published++;
    last = value;
    // Model expensive UI publication crossing the former elapsed-time gate.
    // A dense parser chunk still must not publish on every following event.
    vi.setSystemTime(Date.now() + 1000);
  });
  await monitor.start();
  expect(monitor.position()).toEqual({ attempt: 1, seq: 100000 });
  expect(monitor.snapshot().metrics).toHaveLength(100000);
  expect(monitor.snapshot().summary.progress?.epoch).toBe(100000);
  expect(monitor.snapshot().events).toHaveLength(1000);
  expect(monitor.snapshot().events.at(-1)?.seq).toBe(100000);
  expect(monitor.snapshot().truncated).toBe(true);
  expect(published).toBeLessThan(12);
  expect(last).toBe(monitor.snapshot());
  expect(vi.getTimerCount()).toBe(0);
}, 15000);

it('publishes supersession immediately and never revives a queued metric after stop', async () => {
  vi.useFakeTimers();
  let channel: ReadableStreamDefaultController<Uint8Array> | undefined;

  const transport: RunTransport = {
    getRun: async () => view(),
    cancelRun: async () => view(),
    openEvents: async (_id, _cursor, signal) =>
      new Response(
        new ReadableStream({
          start(controller) {
            channel = controller;
            signal.addEventListener('abort', () => controller.close(), { once: true });
          },
        }),
      ),
  };

  const monitor = new RunMonitor(id, transport);
  const seen: RunMonitorSnapshot[] = [];
  monitor.subscribe((s) => seen.push(s));
  const task = monitor.start();
  await flush();
  const send = (raw: string) => channel?.enqueue(new TextEncoder().encode(raw));
  send(frame(1));
  await flush();
  const first = seen.length;
  send(frame(2));
  await flush();
  expect(seen).toHaveLength(first);
  expect(monitor.position()).toEqual({ attempt: 1, seq: 2 });
  expect(monitor.snapshot().metrics).toHaveLength(2);
  send(frame(0, 'supersede', 2, { source_attempt: 1, step: 1, generation: 'gen-000001' }));
  await flush();
  expect(seen).toHaveLength(first + 1);
  expect(seen.at(-1)?.events.at(-1)?.type).toBe('supersede');
  // Supersession moves lineage to attempt 2 before lifecycle confirms it;
  // the attempt-1 scope must not show that replacement as current progress.
  expect(seen.at(-1)?.metrics).toEqual([]);
  send(frame(1, 'metric', 2, { name: 'loss', value: 9, step: 2 }));
  await flush();
  monitor.stop();
  await task;
  const stopped = seen.length;
  await vi.advanceTimersByTimeAsync(1000);
  expect(seen).toHaveLength(stopped);
  expect(monitor.snapshot().freshness).toBe('stale');
  expect(monitor.snapshot().metrics).toHaveLength(0);
  expect(vi.getTimerCount()).toBe(0);
});
