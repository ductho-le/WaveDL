import pytest

from wavedl.core.events import (
    PAYLOADS,
    SupersedePayload,
    UnknownEventWarning,
    decode_payload,
    validate_supersession,
)
from wavedl.core.protocol import Envelope, ProtocolError, decode, encode


CASES = [
    (
        "hello",
        {
            "protocol_min": 1,
            "protocol_max": 1,
            "runtime_version": "1.8.1",
            "python": "3.11.9",
            "device": "cpu",
            "capabilities": ["journal"],
            "interpreter": "/runtime/python",
        },
        "protocol_min",
        "1",
    ),
    ("state", {"state": "running"}, "state", 1),
    ("metric", {"name": "epoch", "value": None, "step": 1}, "step", True),
    (
        "artifact",
        {
            "kind": "checkpoint",
            "path": "checkpoints/gen-000001",
            "sha256": "a" * 64,
            "size": 12,
        },
        "size",
        "12",
    ),
    ("exit", {"code": 0, "status": "completed"}, "code", False),
    (
        "supersede",
        {"source_attempt": 1, "step": 90, "generation": "gen-000001"},
        "step",
        "90",
    ),
    ("log", {"stream": "stderr", "level": "info", "message": "saved"}, "message", 42),
    ("warning", {"code": "late", "message": "retry"}, "details", []),
    ("error", {"code": "io", "message": "full"}, "recoverable", "false"),
]


def envelope(kind, payload):
    return Envelope(
        run_id="00000000-0000-4000-8000-000000000001",
        attempt=1,
        type=kind,
        seq=0,
        ts="2026-09-06T00:00:00Z",
        payload=payload,
    )


@pytest.mark.parametrize(("kind", "payload", "field", "bad"), CASES)
def test_known_payload_and_extension_fields(kind, payload, field, bad):
    parsed = decode_payload(decode(encode(envelope(kind, payload | {"future": 7}))))
    assert isinstance(parsed, PAYLOADS[kind])
    assert not hasattr(parsed, "future")


@pytest.mark.parametrize(("kind", "payload", "field", "bad"), CASES)
def test_known_field_wrong_type(kind, payload, field, bad):
    with pytest.raises(ProtocolError):
        decode_payload(envelope(kind, payload | {field: bad}))


def test_unknown_event_warns_without_rejecting_envelope():
    original = envelope("future-event", {"arbitrary": [1, 2]})
    with pytest.warns(UnknownEventWarning, match="future-event"):
        assert decode_payload(decode(encode(original))) is None
    assert decode(encode(original)) == original


def test_supersession_rejects_absent_attempt_step_without_mutation():
    payload = SupersedePayload.from_dict(
        {"source_attempt": 1, "step": 90, "generation": "gen-000001"}
    )
    known = {(1, 89), (2, 90)}
    with pytest.raises(ProtocolError, match="absent step"):
        validate_supersession(payload, known)
    assert known == {(1, 89), (2, 90)}


def test_supersession_accepts_exact_restored_step():
    payload = SupersedePayload(1, 90, "gen-000001")
    assert validate_supersession(payload, {(1, 90), (1, 100)}) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "epoch", "value": 1, "step": -1},
        {"name": "epoch", "value": 1, "step": 1, "metrics": {"lr": "fast"}},
        {"name": "epoch", "value": 1, "step": 1, "metrics": {"per_param_mae": []}},
    ],
)
def test_metric_known_semantics(payload):
    with pytest.raises(ProtocolError):
        decode_payload(envelope("metric", payload))


def test_nested_exit_artifact_is_validated():
    artifact = {
        "kind": "checkpoint",
        "path": "checkpoints/gen-000001",
        "sha256": "a" * 64,
        "size": 12,
    }
    payload = {"code": 0, "status": "completed", "artifacts": [artifact]}
    assert decode_payload(envelope("exit", payload)).artifacts[0].size == 12
    with pytest.raises(ProtocolError):
        decode_payload(
            envelope(
                "exit",
                {
                    "code": 0,
                    "status": "completed",
                    "artifacts": [artifact | {"size": "wrong"}],
                },
            )
        )


@pytest.mark.parametrize(
    "name",
    [
        "grad_norm",
        "mae_avg",
        "val_r2",
        "val_pearson",
        "val_mae_avg",
        "total_time",
        "epoch",
        "total_epochs",
        "patience_counter",
        "max_patience",
        "mae_per_param",
        "MAE_Phys_P0",
    ],
)
def test_existing_epoch_metric_types_are_enforced(name):
    with pytest.raises(ProtocolError):
        decode_payload(
            envelope(
                "metric",
                {
                    "name": "epoch",
                    "value": 1,
                    "step": 1,
                    "metrics": {name: "wrong"},
                },
            )
        )


def test_existing_epoch_metric_shapes_accept_unavailable_values():
    parsed = decode_payload(
        envelope(
            "metric",
            {
                "name": "epoch",
                "value": 1,
                "step": 1,
                "metrics": {
                    "epoch": 1,
                    "total_epochs": 20,
                    "patience_counter": 0,
                    "max_patience": 5,
                    "mae_per_param": [None, 0.5],
                    "MAE_Phys_P0": None,
                    "best_val_loss": None,
                },
            },
        )
    )
    assert parsed.metrics["mae_per_param"] == [None, 0.5]


@pytest.mark.parametrize("field", ["value", "lr", "mae_per_param"])
def test_numeric_overflow_is_a_protocol_error(field):
    payload = {"name": "epoch", "value": 1, "step": 1}
    if field == "value":
        payload[field] = 10**400
    else:
        value = [10**400] if field == "mae_per_param" else 10**400
        payload["metrics"] = {field: value}
    with pytest.raises(ProtocolError):
        decode_payload(envelope("metric", payload))
