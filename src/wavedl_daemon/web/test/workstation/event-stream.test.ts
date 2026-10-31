import { describe, expect, it } from 'vitest';
import {
  CursorStore,
  EventStreamParser,
  INITIAL_CURSOR,
  streamRun,
  validCursor,
  type RunEvent,
  type Cursor,
} from '../../app/lib/event-stream';
import { DaemonApi } from '../../app/lib/daemon-api';
import type { FixtureEventOverrides } from './fixture-schemas';

const run = '11111111-1111-1111-1111-111111111111';
const encoder = new TextEncoder();

function event(attempt = 1, seq = 0, overrides: FixtureEventOverrides = {}) {
  return {
    protocol: 'wavedl-jsonl',
    version: 1,
    run_id: run,
    attempt,
    seq,
    type: 'log',
    ts: '2026-10-05T00:00:00Z',
    payload: { text: 'wave 🌊' },
    ...overrides,
  };
}

function frame(attempt = 1, seq = 0, overrides: FixtureEventOverrides = {}) {
  return `id: ${attempt}:${seq}\ndata: ${JSON.stringify(event(attempt, seq, overrides))}\n\n`;
}

describe('attempt aware fetch stream', () => {
  it('decodes additive cursor fields without weakening the zero-attempt sentinel', () => {
    expect(validCursor({ attempt: 0, seq: -1, extra: true })).toBe(true);
    expect(validCursor({ attempt: 1, seq: -1 })).toBe(true);
    expect(validCursor({ attempt: Number.MAX_SAFE_INTEGER, seq: Number.MAX_SAFE_INTEGER })).toBe(true);

    for (const value of [
      { attempt: 0, seq: 0 },
      { attempt: 1, seq: null },
      { attempt: 1.5, seq: 0 },
      { attempt: 1, seq: Number.MAX_SAFE_INTEGER + 1 },
    ]) {
      expect(validCursor(value)).toBe(false);
    }
  });

  it('does not persist a callback revoked while its commit was pending', async () => {
    const abort = new AbortController();
    const saved: Cursor[] = [];

    const parser = new EventStreamParser(
      run,
      INITIAL_CURSOR,
      async () => {
        await Promise.resolve();
        abort.abort();
      },
      (cursor) => {
        saved.push(cursor);
      },
      undefined,
      abort.signal,
    );

    await expect(parser.push(encoder.encode(frame()))).rejects.toThrow();
    expect(saved).toEqual([]);
    expect(parser.cursor).toEqual(INITIAL_CURSOR);
  });
  it('rejects duplicate JSON ownership and ambiguous persisted cursor members', async () => {
    const parser = new EventStreamParser(run, INITIAL_CURSOR, () => undefined);
    const text = frame().replace('"attempt":1', '"attempt":2,"attempt":1');
    await expect(parser.push(encoder.encode(text))).rejects.toMatchObject({ code: 'protocol' });

    const store = new CursorStore('daemon-a', run, {
      getItem: () => '{"attempt":2,"attempt":1,"seq":0}',
      setItem: () => undefined,
    });

    expect(store.load()).toEqual(INITIAL_CURSOR);
  });
  it('decodes every UTF-8/newline split, including CRLF', async () => {
    const bytes = encoder.encode(frame().replaceAll('\n', '\r\n'));

    for (let split = 0; split <= bytes.length; split++) {
      const received: RunEvent[] = [];

      const parser = new EventStreamParser(run, INITIAL_CURSOR, (value) => {
        received.push(value);
      });

      await parser.push(bytes.slice(0, split));
      await parser.push(bytes.slice(split));
      await parser.finish();
      expect(received).toHaveLength(1);
      expect(received[0]?.payload.text).toBe('wave 🌊');
      expect(parser.cursor).toEqual({ attempt: 1, seq: 0 });
    }
  });
  it('deduplicates replay and accepts requeue seq zero', async () => {
    const received: RunEvent[] = [];

    const parser = new EventStreamParser(run, INITIAL_CURSOR, (value) => {
      received.push(value);
    });

    await parser.push(encoder.encode(frame() + frame() + frame(1, 1) + frame(2, 0) + frame(1, 1)));
    expect(received.map((value) => [value.attempt, value.seq])).toEqual([
      [1, 0],
      [1, 1],
      [2, 0],
    ]);
    expect(parser.cursor).toEqual({ attempt: 2, seq: 0 });
  });
  it('commits callback before cursor persistence', async () => {
    const order: string[] = [];

    const parser = new EventStreamParser(
      run,
      INITIAL_CURSOR,
      async () => {
        order.push('commit');
      },
      () => {
        order.push('cursor');
      },
    );

    await parser.push(encoder.encode(frame()));
    expect(order).toEqual(['commit', 'cursor']);

    const rejected = new EventStreamParser(run, INITIAL_CURSOR, () => {
      throw new Error('cannot commit');
    });

    await expect(rejected.push(encoder.encode(frame()))).rejects.toThrow('cannot commit');
    expect(rejected.cursor).toEqual(INITIAL_CURSOR);
  });
  it('controls and comments never advance the cursor', async () => {
    const controls: string[] = [];

    const parser = new EventStreamParser(
      run,
      { attempt: 1, seq: 4 },
      () => {
        throw new Error('not runtime');
      },
      undefined,
      (value) => {
        controls.push(value.type);
      },
    );

    await parser.push(
      encoder.encode(
        ': heartbeat\n\nevent: overflow\ndata: {"resumePosition":{"attempt":1,"seq":4}}\n\nevent: error\ndata: {"resumePosition":{"attempt":1,"seq":4}}\n\n',
      ),
    );
    expect(controls).toEqual(['overflow', 'error']);
    expect(parser.cursor).toEqual({ attempt: 1, seq: 4 });
  });
  it.each([
    frame(1, 0, { run_id: '22222222-2222-2222-2222-222222222222' }),
    frame(1, 0, { version: 2 }),
    frame(1, 0, { seq: 9007199254740992 }),
    frame(1, 0).replace('id: 1:0', 'id: 1:1'),
    frame(1, 0, { payload: [] }),
    frame(1, 0, { ts: 'not a timestamp' }),
  ])('rejects ownership/protocol/id/gap errors', async (text) => {
    const parser = new EventStreamParser(run, INITIAL_CURSOR, () => undefined);
    await expect(parser.push(encoder.encode(text))).rejects.toMatchObject({ code: 'protocol' });
    expect(parser.cursor).toEqual(INITIAL_CURSOR);
  });
  it('accepts sparse durable sequence numbers and attempts', async () => {
    const received: RunEvent[] = [];

    const parser = new EventStreamParser(run, INITIAL_CURSOR, (value) => {
      received.push(value);
    });

    await parser.push(encoder.encode(frame(1, 5) + frame(1, 19) + frame(4, 3)));
    expect(received.map((value) => [value.attempt, value.seq])).toEqual([
      [1, 5],
      [1, 19],
      [4, 3],
    ]);
    expect(parser.cursor).toEqual({ attempt: 4, seq: 3 });
  });
  it('accepts large finite metrics but never unsafe integer cursor fields', async () => {
    const received: RunEvent[] = [];

    const parser = new EventStreamParser(run, INITIAL_CURSOR, (value) => {
      received.push(value);
    });

    await parser.push(encoder.encode(frame(1, 0, { payload: { metric: 1e100 } })));
    expect(received[0]?.payload.metric).toBe(1e100);
    await expect(
      parser.push(encoder.encode(frame(1, 1, { seq: 9007199254740992 }))),
    ).rejects.toMatchObject({ code: 'protocol' });
  });
  it('refuses controls advertising uncommitted progress', async () => {
    const parser = new EventStreamParser(run, { attempt: 1, seq: 4 }, () => undefined);
    await expect(
      parser.push(
        encoder.encode('event: overflow\ndata: {"resumePosition":{"attempt":2,"seq":0}}\n\n'),
      ),
    ).rejects.toMatchObject({ code: 'protocol' });
    expect(parser.cursor).toEqual({ attempt: 1, seq: 4 });
  });
  it('rejects oversized lines and frames, malformed UTF8 and incomplete JSON', async () => {
    for (const bytes of [
      encoder.encode('x'.repeat(1024 * 1024 + 1)),
      encoder.encode(
        `data: ${'x'.repeat(700000)}\ndata: ${'x'.repeat(700000)}\ndata: ${'x'.repeat(700000)}\n\n`,
      ),
      new Uint8Array([255]),
      encoder.encode('id: 1:0\ndata: {\n\n'),
    ]) {
      await expect(
        new EventStreamParser(run, INITIAL_CURSOR, () => undefined).push(bytes),
      ).rejects.toThrow();
    }
  });
  it('does not dispatch incomplete final frame', async () => {
    const received: RunEvent[] = [];

    const parser = new EventStreamParser(run, INITIAL_CURSOR, (value) => {
      received.push(value);
    });

    await parser.push(encoder.encode(frame().slice(0, -1)));
    await parser.finish();
    expect(received).toEqual([]);
  });
  it('persists only exact cursor coordinates per daemon and run', () => {
    const map = new Map<string, string>();

    const storage = {
      getItem: (key: string) => map.get(key) ?? null,
      setItem: (key: string, value: string) => {
        map.set(key, value);
      },
    };

    const a = new CursorStore('daemon-a', run, storage);
    a.save({ attempt: 2, seq: 0 });
    expect(a.load()).toEqual({ attempt: 2, seq: 0 });
    expect(new CursorStore('daemon-b', run, storage).load()).toEqual(INITIAL_CURSOR);
    expect([...map.values()]).toEqual(['{"attempt":2,"seq":0}']);
    storage.setItem(a.key, '{"attempt":9007199254740992,"seq":0}');
    expect(a.load()).toEqual(INITIAL_CURSOR);
  });
  it('reads a real fetch body with committed cursor and releases on callback failure', async () => {
    let cancelled = false;
    let pulls = 0;

    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (pulls++ === 0) controller.enqueue(encoder.encode(frame()));
      },
      cancel() {
        cancelled = true;
      },
    });

    const api = new DaemonApi('http://127.0.0.1:8765', {}, async (url) => {
      if (String(url).endsWith('/health'))
        return new Response(
          JSON.stringify({
            daemon_id: 'daemon-a',
            build_id: 'build-a',
            protocol_min: 1,
            protocol_max: 1,
          }),
          { headers: { 'Content-Type': 'application/json' } },
        );

      return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
    });

    api.setToken('token');

    const store = new CursorStore('daemon-a', run, {
      getItem: () => null,
      setItem: () => {
        throw new Error('should not persist');
      },
    });

    await expect(
      streamRun(api, run, store, () => {
        throw new Error('commit failed');
      }),
    ).rejects.toThrow('commit failed');
    expect(cancelled).toBe(true);
  });
  it('fails closed when a real fetch body reports stream overflow without a recovery handler', async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode('event: overflow\ndata: {"resumePosition":{"attempt":0,"seq":-1}}\n\n'),
        );
        controller.close();
      },
    });

    const api = new DaemonApi('http://127.0.0.1:8765', {}, async (url) => {
      if (String(url).endsWith('/health'))
        return new Response(
          JSON.stringify({
            daemon_id: 'daemon-a',
            build_id: 'build-a',
            protocol_min: 1,
            protocol_max: 1,
          }),
          { headers: { 'Content-Type': 'application/json' } },
        );

      return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
    });

    api.setToken('token');
    const saved: string[] = [];

    const storage = {
      getItem: () => null,
      setItem: (_key: string, value: string) => saved.push(value),
    };

    await expect(
      streamRun(api, run, new CursorStore('daemon-a', run, storage), () => undefined),
    ).rejects.toMatchObject({ code: 'protocol', message: 'Stream requires replay' });
    expect(saved).toEqual([]);
  });
  it('caps multibyte lines by UTF8 bytes, not character count', async () => {
    await expect(
      new EventStreamParser(run, INITIAL_CURSOR, () => undefined).push(
        encoder.encode('🌊'.repeat(270000)),
      ),
    ).rejects.toThrow('line exceeds limit');
  });
  it('refuses cursor stores scoped to a different daemon or run', async () => {
    const urls: string[] = [];

    const api = new DaemonApi('http://127.0.0.1:8765', {}, async (url) => {
      urls.push(String(url));

      return new Response(
        JSON.stringify({
          daemon_id: 'daemon-a',
          build_id: 'build-a',
          protocol_min: 1,
          protocol_max: 1,
        }),
        { headers: { 'Content-Type': 'application/json' } },
      );
    });

    api.setToken('token');
    const storage = { getItem: () => null, setItem: () => undefined };
    await expect(
      streamRun(api, run, new CursorStore('daemon-b', run, storage), () => undefined),
    ).rejects.toMatchObject({ code: 'identity' });
    expect(urls).toHaveLength(1);
    expect(urls[0]).toContain('/health');
    await expect(
      streamRun(
        api,
        run,
        new CursorStore('daemon-a', '22222222-2222-2222-2222-222222222222', storage),
        () => undefined,
      ),
    ).rejects.toMatchObject({ code: 'identity' });
    expect(urls).toHaveLength(1);
  });
});
