# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded journal reads and authenticated-push ingestion on the owner thread."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import islice
from pathlib import Path

from wavedl.core.journal_reader import JournalReader
from wavedl.core.protocol import Envelope, ProtocolError
from wavedl.core.safe_files import open_directory
from wavedl_daemon.backends.contracts import Handle
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.negotiation import (
    NegotiationStatus,
    canonical_event,
    checked_hello,
)
from wavedl_daemon.storage.event_log import EventLogStore, record_bytes
from wavedl_daemon.storage.executions import native_runtime_pending


def registered_handle_matches(record, handle):
    """Only a durable execution snapshot supplies managed journal lineage."""
    return isinstance(handle, Handle) and (
        handle.run_id,
        handle.request_id,
        handle.owner,
        handle.backend,
        handle.run_dir,
        handle.attempt,
    ) == (
        record.run_id,
        record.request_id,
        record.owner,
        record.backend,
        record.run_dir,
        record.attempt,
    )


def standalone_adoption(record):
    """A stored explicit adoption is distinct from managed startup evidence."""
    return registered_handle_matches(record, getattr(record, "handle", None)) and (
        record.handle.backend == "local"
        and record.handle.raw_json == '{"adopted":true}'
    )


def archived_execution_matches(current, archived):
    return (
        archived.plan_id == current.plan_id
        and (
            archived.run_id,
            archived.request_id,
            archived.owner,
            archived.backend,
            archived.run_dir,
        )
        == (
            current.run_id,
            current.request_id,
            current.owner,
            current.backend,
            current.run_dir,
        )
        and type(archived.attempt) is int
        and 1 <= archived.attempt < current.attempt
        and registered_handle_matches(archived, archived.handle)
    )


class Ingestor:
    """Workers read files; negotiation and SQLite writes stay on the owner thread."""

    def __init__(
        self,
        registry,
        controller,
        broker,
        pool,
        *,
        max_pages=4,
        page_size=256,
        read_bytes=256 * 1024,
        max_readers=256,
        max_bookmarks=4096,
        event_log=None,
    ):
        bounds = (
            (max_pages, 16),
            (page_size, 1024),
            (read_bytes, 1024 * 1024),
            (max_readers, 4096),
            (max_bookmarks, 65536),
        )
        if any(
            type(value) is not int or not 1 <= value <= upper for value, upper in bounds
        ):
            raise ValueError("Invalid bounded ingestion configuration")
        self.registry, self.controller = registry, controller
        self.broker, self.pool = broker, pool
        self.max_pages, self.page_size = max_pages, page_size
        self.read_bytes, self.max_readers = read_bytes, max_readers
        self.max_bookmarks = max_bookmarks
        self.event_log = event_log
        self._readers = {}
        self._bookmarks = {}
        self._resume_keys = set()
        self._busy = set()

    def push(self, plan_id, run_id, attempt, events, *, journal=False):
        execution_store = getattr(self.controller, "execution_store", None)
        if execution_store is not None:
            record = execution_store.get(plan_id)
            if record is None:
                raise KeyError(plan_id)
            if record.attempt is None:
                raise ProtocolError("Submission has no bound runtime attempt")
            if (record.run_id, record.attempt) != (run_id, attempt):
                raise ProtocolError("Runtime push is not for the active attempt")
            # During a native requeue the previous runtime remains the last
            # started attempt, but is no longer an active writer.  Its wire
            # output is diagnostic history only; accepting it here could
            # mutate active provenance while the new binding is pending.
            if native_runtime_pending(record):
                raise ProtocolError("Runtime push is not for the active attempt")
        maximum = self.max_pages * self.page_size
        batch = list(islice(events, maximum + 1))
        if len(batch) > maximum:
            raise ProtocolError("Event batch exceeds record bound")
        previous = -1
        for event in batch:
            if not isinstance(event, Envelope):
                raise ProtocolError("Runtime event must be an Envelope")
            if (event.run_id, event.attempt) != (run_id, attempt):
                raise ProtocolError("Event identity differs from requested attempt")
            if event.attempt > 2**63 - 1 or event.seq > 2**63 - 1:
                raise ProtocolError("Event cursor exceeds storage integer bound")
            if event.seq <= previous:
                raise ProtocolError("Event sequence must increase")
            record_bytes(event)
            previous = event.seq
        cursor = self.registry.cursor(run_id, attempt)
        older = [event for event in batch if event.seq <= cursor]
        if older:
            store = self.event_log or EventLogStore(self.registry.path)
            try:
                for event in older:
                    saved = store.get(run_id, attempt, event.seq)
                    if saved is None and journal:
                        continue
                    if saved is None or canonical_event(saved) != canonical_event(
                        event
                    ):
                        raise ProtocolError("Conflicting duplicate runtime event")
            finally:
                if self.event_log is None:
                    store.close()
        fresh = [event for event in batch if event.seq > cursor]
        if not fresh and not journal:
            return cursor
        accepted = batch if journal else fresh
        if not accepted:
            return cursor
        # Validate live-stream continuity before handing events to the owner
        # controller.  Negotiation commits immutable runtime provenance; doing
        # it before rejecting a sequence gap would leave a hello/provenance
        # side effect for a batch that is guaranteed to fail and be replayed
        # from the journal.
        if not journal:
            expected = cursor + 1
            for event in fresh:
                if event.seq != expected:
                    raise ProtocolError("Push gap requires journal replay before retry")
                expected += 1
        result = self.controller.accept_runtime_events(plan_id, accepted)
        if result.status != NegotiationStatus.ACCEPTED:
            raise ProtocolError(result.reason or "Runtime negotiation failed")
        for event in accepted:
            canonical_event(event)
        applied = self.registry.apply_events(run_id, attempt, accepted)
        # Notification failure cannot undo a committed authoritative row.
        self.broker.notify(run_id)
        return applied

    def reject_runtime_events(self, plan_id, run_id, attempt, reason):
        """Malformed first wire output is protocol evidence, never training data."""
        execution_store = getattr(self.controller, "execution_store", None)
        if execution_store is not None:
            record = execution_store.get(plan_id)
            if record is None:
                raise KeyError(plan_id)
            if record.attempt is None:
                raise ProtocolError("Submission has no bound runtime attempt")
            if (record.run_id, record.attempt) != (run_id, attempt):
                raise ProtocolError("Runtime rejection is not for active attempt")
            if native_runtime_pending(record):
                raise ProtocolError("Runtime rejection is not for active attempt")
        provenance_store = getattr(self.controller, "provenance_store", None)
        already_accepted = (
            provenance_store is not None
            and provenance_store.get(plan_id, attempt) is not None
        )
        if self.registry.cursor(run_id, attempt) == -1 and not already_accepted:
            self.controller.reject_runtime_events(plan_id, reason)

    def _historical_identity(self, plan_id, run_id, attempt, path):
        record = self.controller.execution_store.get(plan_id)
        if record is None:
            raise KeyError(plan_id)
        native_pending = native_runtime_pending(record)
        same_pending_attempt = (
            native_pending and type(record.attempt) is int and attempt == record.attempt
        )
        if (
            record.attempt is None
            or record.run_id != run_id
            or not (same_pending_attempt or 1 <= attempt < record.attempt)
        ):
            raise ProtocolError("Historical journal is not an older registered attempt")
        expected = Path(record.run_dir) / "attempts" / str(attempt) / "events.jsonl"
        if Path(path).absolute() != expected.absolute():
            raise ProtocolError("Historical journal differs from registered run source")
        if standalone_adoption(record):
            # Standalone predecessors have no daemon activation snapshots. The
            # explicit adoption authorizes their independent full verification
            # on the filesystem worker, never a managed-run fallback.
            return True
        if not registered_handle_matches(record, getattr(record, "handle", None)):
            raise ProtocolError("Historical journal lacks registered execution lineage")
        if same_pending_attempt:
            return False
        store = self.controller.execution_store
        # Immutable history is ordered by attempt. Indexed offset probes bound
        # each identity check to at most 63 single-record owner-thread reads,
        # even when a run has more history than one monitoring page can retain.
        lower, upper = 0, store.history_count(plan_id)
        if type(upper) is not int or not 0 <= upper <= 2**63 - 1:
            raise ProtocolError("Invalid registered execution history count")
        for _ in range(63):
            if lower >= upper:
                break
            middle = (lower + upper) // 2
            page = store.history(plan_id, limit=1, offset=middle)
            if len(page) != 1:
                raise ProtocolError("Registered execution history changed")
            archived = page[0]
            if archived.attempt == attempt:
                if archived_execution_matches(record, archived):
                    return False
                break
            if archived.attempt < attempt:
                lower = middle + 1
            else:
                upper = middle
        raise ProtocolError("Historical journal lacks activated archived lineage")

    def _apply_historical(self, run_id, attempt, events):
        # Historical telemetry is never active execution or runtime provenance.
        previous = -1
        for event in events:
            if (event.run_id, event.attempt) != (
                run_id,
                attempt,
            ) or event.seq <= previous:
                raise ProtocolError("Invalid historical journal identity or order")
            record_bytes(event)
            canonical_event(event)
            previous = event.seq
        position = self.registry.apply_events(run_id, attempt, events)
        if events:
            self.broker.notify(run_id)
        return position

    async def poll_once(self, plan_id, run_id, attempt, path, *, historical=False):
        execution_store = getattr(self.controller, "execution_store", None)
        record = None
        if execution_store is not None:
            record = execution_store.get(plan_id)
            if record is None:
                raise KeyError(plan_id)
            if record.attempt is None:
                raise ProtocolError("Submission has no bound runtime attempt")
            if not historical:
                handle = getattr(record, "handle", None)
                if not registered_handle_matches(record, handle):
                    raise ProtocolError(
                        "Active journal lacks registered execution lineage"
                    )
                expected = (
                    Path(record.run_dir) / "attempts" / str(attempt) / "events.jsonl"
                )
                if (
                    record.run_id != run_id
                    or record.attempt != attempt
                    or Path(path).absolute() != expected.absolute()
                ):
                    raise ProtocolError(
                        "Active journal differs from registered run source"
                    )
        adopted_history = False
        if historical:
            adopted_history = self._historical_identity(plan_id, run_id, attempt, path)
        key = (run_id, attempt)
        if key in self._busy:
            return self.registry.cursor(run_id, attempt)
        reader = self._readers.get(key)
        if reader is None:
            if (
                key not in self._resume_keys
                and len(self._resume_keys) >= self.max_bookmarks
            ):
                raise BudgetExhausted("Journal resume bookmark capacity exhausted")
            if len(self._readers) >= self.max_readers:
                # Compact committed bookmarks retain forward progress without
                # retaining every reader's pending and ready raw payload.
                idle = next(
                    (item for item in self._readers if item not in self._busy), None
                )
                if idle is None:
                    raise BudgetExhausted("Journal reader capacity exhausted")
                self._readers.pop(idle)
            reader = JournalReader(
                path,
                run_id=run_id,
                attempt=attempt,
                read_bytes=self.read_bytes,
                verify_private_owner=historical,
            )
            bookmark = self._bookmarks.get(key)
            if bookmark is not None and bookmark.private_owner == historical:
                reader.restore(bookmark)
            self._readers[key] = reader
        elif historical and not reader.verify_private_owner:
            # A reader which previously followed an active attempt starts again
            # at byte zero to independently certify its historical first hello.
            reader = JournalReader(
                path,
                run_id=run_id,
                attempt=attempt,
                read_bytes=self.read_bytes,
                verify_private_owner=True,
            )
            self._readers[key] = reader
            self._bookmarks.pop(key, None)
        self._resume_keys.add(key)
        self._busy.add(key)

        def read(token):
            events = []
            with ExitStack() as anchors:
                root = (
                    anchors.enter_context(
                        open_directory(Path(path).parent.parent.parent)
                    )
                    if historical
                    else None
                )
                proof = None
                if adopted_history:
                    from wavedl_daemon.runs.adoption import verify_standalone_run

                    proof = anchors.enter_context(
                        verify_standalone_run(
                            Path(path).parent.parent.parent, attempt=attempt
                        )
                    )
                if proof is not None and (
                    proof.layout.run_id != run_id or proof.attempt != attempt
                ):
                    raise ProtocolError("Historical standalone proof identity differs")
                if root is not None:
                    root.verify_private_owner()
                    root.guard()
                for _ in range(self.max_pages):
                    token.check()
                    try:
                        page = reader.read(max_records=self.page_size)
                    except FileNotFoundError:
                        break
                    if (
                        historical
                        and page
                        and not getattr(reader, "historical_hello", False)
                    ):
                        checked_hello(page[0])
                        if proof is not None and canonical_event(
                            page[0]
                        ) != canonical_event(proof.hello):
                            raise ProtocolError("Historical standalone hello changed")
                        reader.historical_hello = True
                    events.extend(page)
                    if not page:
                        break
                if root is not None:
                    root.guard()
                if proof is not None:
                    proof.guard()
            token.check()
            return events, reader.snapshot()

        handle = None
        try:
            handle = self.pool.submit(read)
            events, bookmark = await handle.wait()
            if historical:
                self._historical_identity(plan_id, run_id, attempt, path)
                position = self._apply_historical(run_id, attempt, events)
            else:
                position = self.push(plan_id, run_id, attempt, events, journal=True)
            # Bookmarks are authoritative only AFTER the corresponding SQLite
            # commit; an exception must never retain an advanced byte boundary.
            self._bookmarks[key] = bookmark
            return position
        except BaseException as error:
            # Never retain an advanced file position after a failed DB commit.
            self._readers.pop(key, None)
            # Keep the prior committed boundary (if any), never the advanced
            # failed worker state. Its inode pin also latches replacements.
            if key not in self._bookmarks:
                self._resume_keys.discard(key)
            if handle is not None:
                handle.cancel()
            if isinstance(error, ProtocolError) and not historical:
                self.reject_runtime_events(plan_id, run_id, attempt, str(error))
            raise
        finally:
            self._busy.discard(key)

    def forget(self, run_id, attempt):
        """Release framing state for attempts no longer being followed."""
        self._readers.pop((run_id, attempt), None)
        self._bookmarks.pop((run_id, attempt), None)
        self._resume_keys.discard((run_id, attempt))
