# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Owner-thread atomic registration after bounded standalone file verification."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from wavedl_daemon.backends.contracts import Plan, UnsupportedPlan
from wavedl_daemon.runs.adoption import verify_standalone_run
from wavedl_daemon.storage.adoptions import AdoptionRegistrationStore


@dataclass(frozen=True)
class AdoptionResponse:
    plan_id: str
    run_id: str
    request_id: str
    operation: str
    attempt: int
    backend: str = "local"
    state: str = "supervision_lost"
    uncertainty: bool = True
    submission_requested: bool = True
    cancellation_requested: bool = False
    backend_status: str = "unknown"
    adopted: bool = True

    def to_dict(self):
        return asdict(self)


class _ProofOwner:
    def __init__(self):
        self.lock = threading.Lock()
        self.proof = None
        self.abandoned = False

    def verify(self, path, attempt):
        proof = verify_standalone_run(path, attempt=attempt)
        with self.lock:
            abandoned = self.abandoned
            if not abandoned:
                self.proof = proof
        if abandoned:
            proof.close()
        return proof

    def close(self):
        with self.lock:
            self.abandoned = True
            proof, self.proof = self.proof, None
        if proof is not None:
            proof.close()


class VerifiedAdoptionService:
    """No backend submission/cancellation; adopted identity has no supervisor."""

    def __init__(
        self,
        settings,
        plans,
        executions,
        provenance,
        registry,
        controller,
        pool,
        backends,
    ):
        self.settings, self.plans, self.executions = settings, plans, executions
        self.provenance, self.registry, self.controller = (
            provenance,
            registry,
            controller,
        )
        self.pool, self.backends = pool, dict(backends)
        self._thread = threading.get_ident()
        for store in (executions, provenance, registry):
            if store is not None and store.path != self.plans.path:
                raise ValueError(
                    "Adoption stores must share one authoritative database"
                )
        info = self.plans.path.stat()
        self._db_identity = info.st_dev, info.st_ino

    def _guard(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("Adoption service must stay on its store owner thread")

    async def adopt(self, path, *, attempt=None, idempotency_key):
        self._guard()
        if (
            type(path) is not str
            or not path
            or "\0" in path
            or len(path.encode()) > 4096
        ):
            raise ValueError("Invalid adoption path")
        if not Path(path).is_absolute():
            raise ValueError("Adoption path must be absolute")
        if attempt is not None and (type(attempt) is not int or attempt < 1):
            raise ValueError("Invalid adoption attempt")
        if type(idempotency_key) is not str or not 1 <= len(idempotency_key) <= 256:
            raise ValueError("A bounded idempotency key is required")
        if any(ord(char) < 32 or ord(char) == 127 for char in idempotency_key):
            raise ValueError("Invalid idempotency key")
        if any(
            store is not None and store.read_only
            for store in (self.plans, self.executions, self.provenance, self.registry)
        ):
            raise PermissionError("Adoption is unavailable in read-only recovery")
        owner = _ProofOwner()
        try:
            verified = await self.pool.submit(
                lambda token: owner.verify(path, attempt)
            ).wait()
            self._guard()
            manifest = verified.request_manifest
            if Path(manifest.run_directory).resolve() != verified.layout.run_dir:
                raise UnsupportedPlan(
                    "Adoption source is not the daemon-side run directory"
                )
            plan = Plan(
                target="local",
                manifest_json=json.dumps(manifest.to_dict(), allow_nan=False),
                argv=(manifest.runtime.python, "-m", "wavedl.train"),
                environment=tuple(sorted(manifest.runtime.environment.items())),
                warnings=("Observed standalone run; no daemon supervisor was adopted",),
            )
            backend = self.backends.get("local")
            if backend is None or not getattr(backend, "owner", None):
                raise UnsupportedPlan("Local owner identity is unavailable")
            with AdoptionRegistrationStore(
                self.plans.path,
                dispatch_lock=self.plans.dispatch_lock,
                read_only=self.plans.read_only,
                expected_identity=self._db_identity,
            ) as store:
                record = store.register_adoption(
                    plan,
                    request_id=verified.request_id,
                    attempt=verified.attempt,
                    owner=backend.owner,
                    idempotency_key=idempotency_key,
                    payload={"path": path, "attempt": attempt},
                    hello=verified.hello,
                    verified_guard=verified.guard,
                )
            return AdoptionResponse(
                record.plan_id,
                verified.layout.run_id,
                record.request_id,
                str(manifest.operation),
                verified.attempt,
            )
        finally:
            owner.close()
