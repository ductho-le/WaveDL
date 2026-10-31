import {
  DaemonApi,
  DaemonApiError,
  object,
  parseJson,
  type Json,
} from './daemon-api';

import { cursorSchema, runEventSchema } from '../../shared/schemas/event-stream';
import type { z } from 'zod';

export type Cursor = z.infer<typeof cursorSchema>;

export type RunEvent = z.infer<typeof runEventSchema>;

export interface StreamControl {
  type: 'overflow' | 'error';
  resumePosition: Cursor;
}

export interface CursorStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export const INITIAL_CURSOR: Readonly<Cursor> = Object.freeze({ attempt: 0, seq: -1 });
const MAX_LINE = 1024 * 1024;
const MAX_FRAME = 2 * 1024 * 1024;

export function validCursor(value: Json | Cursor | undefined): value is Cursor {
  return cursorSchema.safeParse(value).success;
}

interface CursorRead {
  cursor: Cursor;
  status: 'saved' | 'empty' | 'invalid' | 'unavailable';
}

export class CursorStore {
  readonly key: string;
  constructor(
    readonly daemonId: string,
    readonly runId: string,
    private readonly storage: CursorStorage,
  ) {
    if (
      !/^[\x21-\x7e]{1,256}$/.test(daemonId) ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(runId)
    ) {
      throw DaemonApiError.fromKey('identity', 'invalidCursorStorageScope');
    }

    this.key = `wavedl:cursor:v1:${encodeURIComponent(daemonId)}:${encodeURIComponent(runId)}`;
  }
  load(): Cursor {
    return this.read().cursor;
  }
  read(): CursorRead {
    let raw: string | null;

    try {
      raw = this.storage.getItem(this.key);
    } catch {
      return { cursor: { ...INITIAL_CURSOR }, status: 'unavailable' };
    }

    try {
      const value = raw === null || raw.length > 128 ? null : parseJson(raw);

      return validCursor(value)
        ? { cursor: { attempt: value.attempt, seq: value.seq }, status: 'saved' }
        : { cursor: { ...INITIAL_CURSOR }, status: raw === null ? 'empty' : 'invalid' };
    } catch {
      return { cursor: { ...INITIAL_CURSOR }, status: 'invalid' };
    }
  }
  save(cursor: Cursor): void {
    if (!validCursor(cursor)) throw DaemonApiError.fromKey('response', 'invalidCursor');
    this.storage.setItem(this.key, JSON.stringify({ attempt: cursor.attempt, seq: cursor.seq }));
  }
}

function envelope(data: string, runId: string, id: string): RunEvent {
  let value;

  try {
    value = parseJson(data);
  } catch {
    throw DaemonApiError.fromKey('protocol', 'malformedOrUnsafeRunEventJson');
  }

  const parsed = runEventSchema.safeParse(value);

  if (!parsed.success || parsed.data.run_id !== runId || id !== `${parsed.data.attempt}:${parsed.data.seq}`) {
    throw DaemonApiError.fromKey('protocol', 'invalidEventScope');
  }

  const event = parsed.data;

  if (new Date(event.ts).toISOString().slice(0, 19) !== event.ts.slice(0, 19)) {
    throw DaemonApiError.fromKey('protocol', 'invalidEventTimestamp');
  }

  // parseJson recursively validated payload's JSON domain before narrowing.
  const payload = event.payload;

  return {
    protocol: 'wavedl-jsonl',
    version: 1,
    run_id: runId,
    attempt: event.attempt,
    seq: event.seq,
    type: event.type,
    ts: event.ts,
    payload,
  };
}

/** Sequential async callbacks provide backpressure; commit precedes cursor advance. */
export class EventStreamParser {
  #decoder = new TextDecoder('utf-8', { fatal: true });
  #pending = '';
  #lineBytes = 0;
  #data: string[] = [];
  #id = '';
  #type = '';
  #frameBytes = 0;
  #cursor: Cursor;
  constructor(
    private readonly runId: string,
    cursor: Cursor,
    private readonly commit: (event: RunEvent) => void | Promise<void>,
    private readonly persist: (cursor: Cursor) => void = () => undefined,
    private readonly control: (event: StreamControl) => void | Promise<void> = () => undefined,
    private readonly signal?: AbortSignal,
  ) {
    if (
      !validCursor(cursor) ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(runId)
    )
      throw DaemonApiError.fromKey('response', 'invalidCursorOrRunIdentity');
    this.#cursor = { ...cursor };
  }
  get cursor(): Cursor {
    return { ...this.#cursor };
  }
  async push(chunk: Uint8Array): Promise<void> {
    for (let offset = 0; offset < chunk.length; offset += 65536) {
      let text: string;

      try {
        text = this.#decoder.decode(chunk.subarray(offset, offset + 65536), { stream: true });
      } catch {
        throw DaemonApiError.fromKey('protocol', 'invalidUtf8Stream');
      }

      await this.#consume(text);
    }
  }
  async finish(): Promise<void> {
    let text: string;

    try {
      text = this.#decoder.decode();
    } catch {
      throw DaemonApiError.fromKey('protocol', 'truncatedUtf8Stream');
    }

    await this.#consume(text);
    // An incomplete final frame is discarded and replayed on reconnect.
    this.#pending = '';
    this.#lineBytes = 0;
    this.#reset();
  }
  async #consume(text: string): Promise<void> {
    // Slice before appending, so a large network chunk cannot create an unbounded line.
    let start = 0;

    for (;;) {
      const newline = text.indexOf('\n', start);
      const part = text.slice(start, newline < 0 ? text.length : newline);
      this.#lineBytes += new TextEncoder().encode(part).byteLength;

      if (this.#lineBytes > MAX_LINE)
        throw DaemonApiError.fromKey('protocol', 'streamLineExceedsLimit');
      this.#pending += part;

      if (newline < 0) return;
      const line = this.#pending.endsWith('\r') ? this.#pending.slice(0, -1) : this.#pending;
      this.#pending = '';
      this.#lineBytes = 0;
      await this.#line(line);
      start = newline + 1;
    }
  }
  async #line(line: string): Promise<void> {
    this.#frameBytes += new TextEncoder().encode(line).byteLength + 1;

    if (this.#frameBytes > MAX_FRAME)
      throw DaemonApiError.fromKey('protocol', 'streamFrameExceedsLimit');

    if (!line) {
      await this.#dispatch();
      this.#reset();

      return;
    }

    if (line.startsWith(':')) return;
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);

    if (value.startsWith(' ')) value = value.slice(1);

    if (field === 'data') this.#data.push(value);
    else if (field === 'id') {
      if (this.#id || value.includes('\0'))
        throw DaemonApiError.fromKey('protocol', 'ambiguousEventId');
      this.#id = value;
    } else if (field === 'event') {
      if (this.#type) throw DaemonApiError.fromKey('protocol', 'ambiguousEventType');
      this.#type = value;
    }
  }
  #reset(): void {
    this.#data = [];
    this.#id = '';
    this.#type = '';
    this.#frameBytes = 0;
  }
  async #dispatch(): Promise<void> {
    this.signal?.throwIfAborted();

    if (!this.#data.length) return;
    const data = this.#data.join('\n');

    if (this.#type === 'overflow' || this.#type === 'error') {
      const value = parseJson(data);

      if (this.#id || !object(value))
        throw DaemonApiError.fromKey('protocol', 'invalidStreamControl');
      const resume = value.resumePosition;

      if (!validCursor(resume)) throw DaemonApiError.fromKey('protocol', 'invalidStreamControl');

      if (
        resume.attempt > this.#cursor.attempt ||
        (resume.attempt === this.#cursor.attempt && resume.seq > this.#cursor.seq)
      ) {
        throw DaemonApiError.fromKey('protocol', 'uncommittedStreamProgress');
      }

      await this.control({
        type: this.#type,
        resumePosition: { attempt: resume.attempt, seq: resume.seq },
      });

      return; // Server controls NEVER advance a runtime cursor.
    }

    if (this.#type && this.#type !== 'message')
      throw DaemonApiError.fromKey('protocol', 'unsupportedStreamEvent');
    const event = envelope(data, this.runId, this.#id);
    const current = this.#cursor;

    if (
      event.attempt < current.attempt ||
      (event.attempt === current.attempt && event.seq <= current.seq)
    )
      return;
    // Durable journals can be sparse. Only ordering, not adjacency, is promised.
    await this.commit(event);
    // A monitor may be revoked while its async UI commit is pending. Never
    // persist unseen progress for a disconnected or superseded consumer.
    this.signal?.throwIfAborted();
    const next = { attempt: event.attempt, seq: event.seq };
    this.persist(next);
    this.#cursor = next;
  }
}

export async function streamRun(
  api: DaemonApi,
  runId: string,
  store: CursorStore,
  commit: (event: RunEvent) => void | Promise<void>,
  options: { signal?: AbortSignal; control?: (event: StreamControl) => void | Promise<void> } = {},
): Promise<Cursor> {
  if (runId !== store.runId)
    throw DaemonApiError.fromKey('identity', 'eventCursorBelongsToAnotherRun');
  const cursor = store.load();
  const response = await api.openEvents(runId, cursor, options.signal, store.daemonId);

  if (!response.body) throw DaemonApiError.fromKey('response', 'missingEventStream');
  const reader = response.body.getReader();

  const parser = new EventStreamParser(
    runId,
    cursor,
    commit,
    (next) => store.save(next),
    options.control ??
      (() => {
        // A control frame means the server could not deliver a complete,
        // replayable stream.  Never resolve as if the stream completed cleanly
        // when the caller has not explicitly supplied recovery handling.
        throw DaemonApiError.fromKey('protocol', 'streamRequiresReplay');
      }),
    options.signal,
  );

  try {
    for (;;) {
      options.signal?.throwIfAborted();
      const { value, done } = await reader.read();

      if (done) break;
      await parser.push(value);
    }

    options.signal?.throwIfAborted();
    await parser.finish();

    return parser.cursor;
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
