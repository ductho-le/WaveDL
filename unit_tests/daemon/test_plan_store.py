"""Plans are immutable, keyed by request, and replayed rather than duplicated."""

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import pytest

from wavedl_daemon.backends.contracts import Runtime
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.runs.model import Operation
from wavedl_daemon.runs.service import build_spec
from wavedl_daemon.storage.database import open_database
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import IdempotencyConflict, PlanStore


PAYLOAD = {"operation": "train", "config": {"epochs": 3}}


@pytest.fixture
def state(tmp_path):
    with acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock:
        yield tmp_path / "registry.db", lock


@pytest.fixture
def store(state):
    path, lock = state
    connection = open_database(path, dispatch_lock=lock)
    yield PlanStore(connection)
    connection.close()


def make_plan(tmp_path):
    spec = build_spec(
        tmp_path / "runs" / str(uuid4()),
        operation=Operation.TRAIN,
        resolved_config={"epochs": 3},
        runtime=Runtime(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    return LocalBackend().plan(spec)


@pytest.fixture
def plan(tmp_path):
    return make_plan(tmp_path)


def put(store, plan, key="request-key", payload=PAYLOAD):
    return store.put(plan, idempotency_key=key, request_payload=payload)


def test_a_lost_response_replays_the_same_record_after_reopen(state, store, plan):
    original = put(store, plan)
    path, lock = state
    connection = open_database(path, dispatch_lock=lock)
    try:
        reopened = PlanStore(connection)
        # A replanner may have selected different defaults after restart.
        assert put(reopened, replace(plan, warnings=("changed defaults",))) == original
        assert reopened.get(original.plan_id) == original
        assert reopened.by_run(plan.run_id) == original
    finally:
        connection.close()
    assert original.run_id == plan.run_id
    assert original.request_id != original.plan_id
    assert original.plan == plan


def test_a_request_key_is_bound_to_its_canonical_payload(store, plan):
    original = put(store, plan)
    assert (
        put(store, plan, payload={"config": {"epochs": 3}, "operation": "train"})
        == original
    )
    with pytest.raises(IdempotencyConflict):
        put(store, plan, payload={"operation": "train", "config": {"epochs": 4}})
    assert store.get_by_key("request-key") == original


def test_a_run_belongs_to_exactly_one_plan(store, plan):
    put(store, plan)
    with pytest.raises(sqlite3.IntegrityError):
        put(store, plan, key="other-key")
    assert store.get_by_key("other-key") is None


def test_plans_cannot_be_changed_or_deleted(store, plan):
    original = put(store, plan)
    connection = store._connection
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            "UPDATE plans SET plan_json='{}' WHERE plan_id=?", (original.plan_id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute("DELETE FROM plans")
    assert store.get(original.plan_id) == original


def test_lookups_of_unknown_identities_are_empty(store):
    assert store.get(str(uuid4())) is None
    assert store.get_by_key("missing") is None
    assert store.by_run(str(uuid4())) is None
    assert store.page(limit=10, offset=0) == []


def test_pages_are_in_creation_order(store, tmp_path):
    records = [
        put(store, make_plan(tmp_path), key=f"key-{index}") for index in range(5)
    ]
    assert store.page(limit=10, offset=0) == records
    assert store.page(limit=2, offset=1) == records[1:3]
    assert store.page(limit=10, offset=5) == []


def test_separate_connections_collapse_concurrent_identical_requests(
    state, store, plan
):
    path, lock = state

    def write(_):
        connection = open_database(path, dispatch_lock=lock)
        try:
            return put(PlanStore(connection), plan)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(write, range(8)))
    assert len({record.plan_id for record in records}) == 1
    assert store.page(limit=10, offset=0) == [records[0]]


@pytest.mark.parametrize(
    "payload",
    [
        {"n": float("nan")},
        {"n": float("inf")},
        {1: "ambiguous"},
        {"nested": {True: "ambiguous"}},
        {"tuple": (1, 2)},
        {"path": Path("/x")},
        MappingProxyType({"custom": "mapping"}),
    ],
)
def test_an_invalid_payload_cannot_reserve_a_key(store, plan, payload):
    with pytest.raises(ValueError):
        put(store, plan, key="bad", payload=payload)
    assert store.get_by_key("bad") is None


@pytest.mark.parametrize("kind", ["cycle", "deep"])
def test_cyclic_and_deep_payloads_cannot_reserve_a_key(store, plan, kind):
    payload = {}
    if kind == "cycle":
        payload["self"] = payload
    else:
        child = payload
        for _ in range(70):
            child["nested"] = {}
            child = child["nested"]
    with pytest.raises(ValueError, match=r"cycle|nesting"):
        put(store, plan, key="bad", payload=payload)
    assert store.get_by_key("bad") is None


def test_an_empty_key_is_refused(store, plan):
    with pytest.raises(ValueError, match="idempotency_key"):
        put(store, plan, key="")
    assert store.page(limit=10, offset=0) == []
