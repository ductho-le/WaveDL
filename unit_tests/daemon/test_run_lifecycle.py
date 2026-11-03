"""The run transition table, status projection and pure control law."""

from functools import reduce
from pathlib import Path

import pytest

from wavedl_daemon.events.metrics_tail import TailChunk
from wavedl_daemon.runs.lifecycle import (
    Append,
    AttemptFiles,
    IllegalTransition,
    RunRef,
    RunState,
    Work,
    WorkKind,
    evolve,
    next_work,
    react,
    settle,
    slurm_outcome,
    status,
)
from wavedl_daemon.runs.model import (
    AttemptOpened,
    BackendKind,
    CancelDelivered,
    CancelRequested,
    Exited,
    ExitRecord,
    LocalExit,
    LocalHandle,
    Metric,
    NotOwned,
    ObservationFailed,
    ObservationLost,
    Observed,
    Operation,
    Outcome,
    Phase,
    PlanId,
    Probed,
    RunId,
    SlurmHandle,
    SlurmObservation,
    SourceWarning,
    Submitted,
    Submitting,
    SupervisionFailed,
    WarningCode,
)


SUBMIT = AttemptOpened("submit", None)
LOCAL = LocalHandle(4242, "supervisor-1")
SLURM = SlurmHandle("4242", "c1", "alice")


def ref(backend=BackendKind.SLURM, operation=Operation.TRAIN):
    return RunRef(RunId("run-1"), PlanId("plan-1"), backend, operation, Path("/runs/run-1"))


def fold(events, run=None):
    return reduce(evolve, events, run or ref())


def metric(epoch, end):
    return Metric(epoch, {"epoch": epoch, "val_loss": 1 / epoch}, end)


def observation(**fields):
    values = {
        "restart_count": 0,
        "phase": Phase.RUNNING,
        "terminal": False,
        "native_state": "RUNNING",
        "reason": None,
        "exit_code": None,
        "signal": None,
        "earlier": {},
    }
    return SlurmObservation(**(values | fields))


LIVE = [SUBMIT, Submitting(), Submitted(SLURM)]
STAGES = {
    "requested": [SUBMIT],
    "requested+cancel": [SUBMIT, CancelRequested()],
    "claimed": [SUBMIT, Submitting()],
    "live": LIVE,
    "live+cancel": [*LIVE, CancelRequested()],
    "live+delivered": [*LIVE, CancelRequested(), CancelDelivered()],
    "requeued": [*LIVE, Exited(Outcome.REQUEUED, False)],
    "ended": [*LIVE, Exited(Outcome.COMPLETED, True)],
}
EVENTS = {
    "submit": SUBMIT,
    "requeue": AttemptOpened("requeue", 1),
    "submitting": Submitting(),
    "probed": Probed("3.11", "/usr/bin/python3", "/usr", ("cpu",)),
    "submitted_local": Submitted(LOCAL),
    "submitted_slurm": Submitted(SLURM),
    "observed": Observed(Phase.QUEUED, "PENDING", "Priority"),
    "observation_lost": ObservationLost("squeue timed out"),
    "cancel_requested": CancelRequested(),
    "cancel_delivered": CancelDelivered(),
    "exit_completed": Exited(Outcome.COMPLETED, True),
    "exit_cancelled": Exited(Outcome.CANCELLED, True),
    "exit_failed": Exited(Outcome.FAILED, True),
    "exit_unknown": Exited(Outcome.UNKNOWN, False),
    "exit_lost": Exited(Outcome.LOST, False),
    "exit_requeued": Exited(Outcome.REQUEUED, False),
    "metric": metric(1, 10),
}
ALL_EXITS = {
    "exit_completed",
    "exit_cancelled",
    "exit_failed",
    "exit_unknown",
    "exit_lost",
    "exit_requeued",
}
LIVE_EVENTS = {"observed", "observation_lost", "metric", *ALL_EXITS}
LEGAL = {
    "requested": {"submitting", "cancel_requested"},
    "requested+cancel": {"exit_cancelled"},
    "claimed": {
        "probed",
        "submitted_slurm",
        "cancel_requested",
        "exit_failed",
        "exit_unknown",
        "exit_lost",
        "metric",
    },
    "live": {*LIVE_EVENTS, "cancel_requested"},
    "live+cancel": {*LIVE_EVENTS, "cancel_delivered"},
    "live+delivered": LIVE_EVENTS,
    "requeued": {"requeue"},
    "ended": set(),
}


def test_every_stage_and_event_pair_follows_the_transition_table():
    wrong = []
    for stage, history in STAGES.items():
        head = fold(history)
        for name, event in EVENTS.items():
            try:
                evolve(head, event)
            except IllegalTransition:
                legal = False
            else:
                legal = True
            if legal != (name in LEGAL[stage]):
                wrong.append((stage, name, legal))
    assert wrong == []


def test_a_run_starts_only_with_a_submit():
    for event in EVENTS.values():
        if event != SUBMIT:
            with pytest.raises(IllegalTransition):
                evolve(ref(), event)


def test_positions_are_contiguous_and_a_requeue_starts_the_next_attempt_at_zero():
    history = [
        *LIVE,
        metric(1, 10),
        Exited(Outcome.REQUEUED, False),
        AttemptOpened("requeue", 1),
        Observed(Phase.RUNNING, "RUNNING", None),
        metric(2, 7),
    ]
    run, positions = ref(), []
    for event in history:
        run = evolve(run, event)
        positions.append((run.attempt, run.seq))
    assert positions == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2)]
    assert run.stage.handle == SLURM
    assert run.metrics_offset == 7
    assert run.restart_count == 1


def test_metric_rows_move_the_bookmark_but_not_the_revision():
    live = fold(LIVE)
    after = evolve(live, metric(1, 10))
    assert (after.revision, after.metrics_offset) == (live.revision, 10)
    assert evolve(after, Observed(Phase.RUNNING, "RUNNING", None)).revision == live.revision + 1


@pytest.mark.parametrize(
    "event",
    [metric(2, 10), metric(2, 5), SourceWarning(WarningCode.TRUNCATED, "shrank", 11)],
)
def test_file_rows_must_read_forward(event):
    with pytest.raises(IllegalTransition):
        evolve(fold([*LIVE, metric(1, 10)]), event)


def test_a_sealing_warning_ends_reading_the_attempts_file():
    for warning in (
        SourceWarning(WarningCode.TRUNCATED, "shrank", 10),
        SourceWarning(WarningCode.LINE_TOO_LONG, "long", 99),
        SourceWarning(WarningCode.TORN_TAIL, "torn", 12),
    ):
        sealed = fold([*LIVE, metric(1, 10), warning])
        assert sealed.metrics_sealed
        with pytest.raises(IllegalTransition):
            evolve(sealed, metric(2, 200))
    malformed = fold([*LIVE, SourceWarning(WarningCode.MALFORMED_LINE, "bad", 4)])
    assert evolve(malformed, metric(1, 10)).metrics_offset == 10


def test_local_runs_take_only_local_handles_and_never_requeue():
    local = ref(BackendKind.LOCAL)
    claimed = fold([SUBMIT, Submitting()], local)
    with pytest.raises(IllegalTransition):
        evolve(claimed, Submitted(SLURM))
    live = evolve(claimed, Submitted(LOCAL))
    with pytest.raises(IllegalTransition):
        evolve(live, Exited(Outcome.REQUEUED, False))


def test_requeue_must_name_the_restart_that_ended_the_attempt():
    requeued = fold(STAGES["requeued"])
    for restart in (None, 0, 2):
        with pytest.raises(IllegalTransition):
            evolve(requeued, AttemptOpened("requeue", restart))


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([SUBMIT], (RunState.PENDING, True, "")),
        ([SUBMIT, Submitting()], (RunState.SUBMISSION_UNKNOWN, False, "")),
        (LIVE, (RunState.SUBMITTED, True, "")),
        ([*LIVE, Observed(Phase.QUEUED, "PENDING", "Priority")], (RunState.QUEUED, True, "Priority")),
        ([*LIVE, Observed(Phase.HELD, "PENDING", "JobHeldUser")], (RunState.HELD, True, "JobHeldUser")),
        ([*LIVE, Observed(Phase.RUNNING, "RUNNING", None)], (RunState.RUNNING, True, "")),
        (
            [*LIVE, Observed(Phase.RUNNING, "RUNNING", None), CancelRequested()],
            (RunState.STOPPING, True, ""),
        ),
        (
            [*LIVE, Observed(Phase.RUNNING, "RUNNING", None), ObservationLost("squeue failed")],
            (RunState.OBSERVATION_UNAVAILABLE, False, "squeue failed"),
        ),
        ([*LIVE, Exited(Outcome.COMPLETED, True)], (RunState.COMPLETED, True, "")),
        ([*LIVE, Exited(Outcome.FAILED, True, detail="x")], (RunState.FAILED, True, "x")),
        ([*LIVE, Exited(Outcome.CANCELLED, True)], (RunState.CANCELLED, True, "")),
        ([*LIVE, Exited(Outcome.UNKNOWN, False)], (RunState.TERMINAL_UNKNOWN, False, "")),
        ([*LIVE, Exited(Outcome.LOST, False)], (RunState.SUPERVISION_LOST, False, "")),
    ],
)
def test_status_is_a_projection_of_the_stage(history, expected):
    result = status(fold(history))
    assert (result.state, result.confirmed, result.reason) == expected


@pytest.mark.parametrize(
    ("native", "record", "expected"),
    [
        ("COMPLETED", ExitRecord(0, 0, 0), (Outcome.COMPLETED, True)),
        ("COMPLETED", ExitRecord(0, 1, 0), (Outcome.FAILED, True)),
        ("COMPLETED", ExitRecord(0, 0, 2), (Outcome.FAILED, True)),
        ("COMPLETED", None, (Outcome.UNKNOWN, False)),
        ("CANCELLED by 1000", None, (Outcome.CANCELLED, True)),
        ("CANCELLED+", ExitRecord(0, 0, 0), (Outcome.CANCELLED, True)),
        ("FAILED", None, (Outcome.FAILED, True)),
        ("TIMEOUT", None, (Outcome.FAILED, True)),
        ("OUT_OF_MEMORY", None, (Outcome.FAILED, True)),
        ("NODE_FAIL", None, (Outcome.FAILED, True)),
        ("BOOT_FAIL", None, (Outcome.FAILED, True)),
        (None, None, (Outcome.UNKNOWN, False)),
        ("SOMETHING_NEW", ExitRecord(0, 0, 0), (Outcome.UNKNOWN, False)),
    ],
)
def test_slurm_outcome(native, record, expected):
    assert slurm_outcome(native, record) == expected


def test_next_work_claims_before_submitting_and_ends_a_cancelled_request():
    assert next_work(fold([SUBMIT]), local_poll=None) == Append(
        (Submitting(),), then=WorkKind.SBATCH
    )
    local = fold([SUBMIT], ref(BackendKind.LOCAL))
    assert next_work(local, local_poll=None) == Append((Submitting(),), then=WorkKind.SPAWN)
    cancelled = next_work(fold(STAGES["requested+cancel"]), local_poll=None)
    assert [event.outcome for event in cancelled.events] == [Outcome.CANCELLED]
    assert cancelled.then is None


def test_next_work_for_submitted_runs():
    def kind(head, poll=None):
        work = next_work(head, local_poll=poll)
        return work.kind if isinstance(work, Work) else work

    local = fold([SUBMIT, Submitting(), Submitted(LOCAL)], ref(BackendKind.LOCAL))
    assert kind(fold([SUBMIT, Submitting()])) is WorkKind.RECOVER
    assert kind(local, LocalExit(0, None)) is WorkKind.SETTLE
    assert kind(local, NotOwned()) is WorkKind.SETTLE
    assert kind(evolve(local, CancelRequested())) is WorkKind.CANCEL
    assert kind(local) is WorkKind.TAIL
    assert kind(fold(STAGES["live+delivered"])) is WorkKind.TAIL
    tested = fold([SUBMIT, Submitting(), Submitted(LOCAL)], ref(BackendKind.LOCAL, Operation.TEST))
    assert kind(tested) is None
    assert kind(fold(STAGES["ended"])) is None
    work = next_work(evolve(local, metric(1, 10)), local_poll=None)
    assert (work.revision, work.attempt, work.offset) == (3, 1, 10)


def test_react_appends_only_changes():
    running = fold([*LIVE, Observed(Phase.RUNNING, "RUNNING", None)])
    assert react(running, observation()) is None
    assert react(running, observation(phase=None, native_state=None)) is None
    assert react(running, observation(phase=Phase.HELD, native_state="SUSPENDED")) == Append(
        (Observed(Phase.HELD, "SUSPENDED", None),)
    )
    lost = react(running, ObservationFailed("squeue timed out"))
    assert lost == Append((ObservationLost("squeue timed out"),))
    unobserved = evolve(running, ObservationLost("squeue timed out"))
    assert react(unobserved, ObservationFailed("again")) is None
    assert react(unobserved, observation()) == Append((Observed(Phase.RUNNING, "RUNNING", None),))
    assert react(fold(STAGES["ended"]), observation()) is None


def test_react_settles_terminal_and_restarted_jobs():
    running = fold(LIVE)
    for seen in (observation(terminal=True, phase=None), observation(restart_count=1)):
        assert react(running, seen).kind is WorkKind.SETTLE


def test_settle_local_exit_outcomes():
    local = fold([SUBMIT, Submitting(), Submitted(LOCAL)], ref(BackendKind.LOCAL, Operation.TEST))
    nothing = AttemptFiles(None, None)

    def ending(head, cause):
        (event,) = settle(head, cause, nothing)
        return event.outcome, event.confirmed

    assert ending(local, LocalExit(0, None)) == (Outcome.COMPLETED, True)
    assert ending(local, LocalExit(1, None)) == (Outcome.FAILED, True)
    assert ending(evolve(local, CancelRequested()), LocalExit(None, 15)) == (Outcome.CANCELLED, True)
    assert ending(local, NotOwned()) == (Outcome.LOST, False)
    assert ending(local, SupervisionFailed("pipes open")) == (Outcome.UNKNOWN, False)


def test_settle_drains_the_file_and_records_a_torn_last_line():
    local = fold([SUBMIT, Submitting(), Submitted(LOCAL)], ref(BackendKind.LOCAL))
    unread = TailChunk(0, (metric(1, 10),), 4, 100)
    assert settle(local, LocalExit(0, None), AttemptFiles(unread, None)) == (metric(1, 10),)
    torn = TailChunk(0, (metric(1, 10),), 4, 14)
    rows = settle(local, LocalExit(1, None), AttemptFiles(torn, None))
    assert rows[0] == metric(1, 10)
    assert (rows[1].code, rows[1].source_end) == (WarningCode.TORN_TAIL, 14)
    assert rows[2].outcome is Outcome.FAILED
    assert fold(rows, local).metrics_offset == 14


def test_settle_a_requeue_ends_one_attempt_and_opens_the_next():
    running = fold([*LIVE, Observed(Phase.RUNNING, "RUNNING", None)])
    seen = observation(phase=Phase.QUEUED, native_state="PENDING", earlier={0: "NODE_FAIL"}, restart_count=1)
    rows = settle(running, seen, AttemptFiles(TailChunk(0, (), 0, 0), None))
    assert rows == (
        Exited(Outcome.REQUEUED, False, native_state="NODE_FAIL", detail="Requeued by Slurm"),
        AttemptOpened("requeue", 1),
        Observed(Phase.QUEUED, "PENDING", None),
    )
    two_behind = settle(running, observation(restart_count=2), AttemptFiles(None, None))
    assert [type(event) for event in two_behind] == [Exited, AttemptOpened]
    assert fold(two_behind, running).attempt == 2


def test_settle_a_terminal_slurm_job_uses_its_exit_record():
    running = fold(LIVE)
    seen = observation(terminal=True, phase=None, native_state="COMPLETED", exit_code=0)
    (ending,) = settle(running, seen, AttemptFiles(None, ExitRecord(0, 0, 0)))
    assert (ending.outcome, ending.confirmed, ending.attempt_exit) == (
        Outcome.COMPLETED,
        True,
        ExitRecord(0, 0, 0),
    )
