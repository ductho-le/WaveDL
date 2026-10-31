"""Ephemeral single-use bootstrap and access-log credential regression tests."""

from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pytest

from wavedl_daemon.http.bootstrap import (
    AccessLogRedactionFilter,
    BootstrapCapacityError,
    BootstrapCredentials,
    BootstrapExchangeError,
    install_access_log_redaction,
)


TOKEN = "daemon-secret+/=private"


def test_launch_code_is_reusable_and_never_expires():
    now = [0.0]
    pool = BootstrapCredentials(TOKEN, clock=lambda: now[0])
    code = pool.mint_launch()
    assert len(code) == 43
    assert code not in repr(pool.__dict__)
    first = pool.exchange(code)
    now[0] = 10_000.0
    second = pool.exchange(code)
    assert first.token != second.token


def test_launch_code_minted_once():
    pool = BootstrapCredentials(TOKEN)
    pool.mint_launch()
    with pytest.raises(BootstrapCapacityError):
        pool.mint_launch()


def test_launch_code_survives_session_eviction():
    pool = BootstrapCredentials(TOKEN, max_sessions=16)
    code = pool.mint_launch()
    grants = [pool.exchange(code) for _ in range(17)]
    assert len({grant.token for grant in grants}) == 17
    assert not pool.authenticate(grants[0].token)


def test_one_time_code_still_single_use_beside_launch_code():
    pool = BootstrapCredentials(TOKEN)
    pool.mint_launch()
    once = pool.mint()
    pool.exchange(once)
    with pytest.raises(BootstrapExchangeError):
        pool.exchange(once)


@pytest.mark.parametrize("operation", ["mint", "mint_session", "exchange"])
def test_launch_digest_collision_fails_closed(monkeypatch, operation):
    pool = BootstrapCredentials(TOKEN)
    once = pool.mint()
    launch = pool.mint_launch()
    monkeypatch.setattr(
        "wavedl_daemon.http.bootstrap.secrets.token_urlsafe", lambda size: launch
    )
    with pytest.raises(BootstrapCapacityError):
        if operation == "exchange":
            pool.exchange(once)
        else:
            getattr(pool, operation)()
    assert pool._launch_digest == hashlib.sha256(launch.encode()).digest()


def test_code_is_random_and_only_digests_are_stored():
    credentials = BootstrapCredentials(TOKEN)
    first, second = credentials.mint(), credentials.mint()
    assert first != second
    assert len(first) == 43
    assert hashlib.sha256(first.encode()).digest() in credentials._pending
    assert TOKEN not in repr(credentials.__dict__)
    assert first not in repr(credentials.__dict__)
    exchanged = credentials.exchange(first)
    assert exchanged.token not in repr(credentials.__dict__)
    assert credentials.authenticate(exchanged.token)
    assert not credentials.is_master(exchanged.token)
    with pytest.raises(BootstrapExchangeError, match="Invalid or expired"):
        credentials.exchange(first)
    assert credentials.authenticate(credentials.exchange(second).token)


def test_sixteen_concurrent_exchanges_have_exactly_one_winner():
    credentials = BootstrapCredentials(TOKEN)
    code = credentials.mint()
    barrier = threading.Barrier(16)

    def exchange(_):
        barrier.wait(timeout=10)
        try:
            return credentials.exchange(code).token
        except BootstrapExchangeError:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(exchange, range(16)))
    assert sum(result is not None for result in results) == 1
    assert results.count(None) == 15
    assert not credentials._pending


def test_deadline_is_exclusive_and_capacity_cleanup_is_bounded():
    now = [10.0]
    credentials = BootstrapCredentials(
        TOKEN, ttl_seconds=2, max_pending=2, clock=lambda: now[0]
    )
    first, second = credentials.mint(), credentials.mint()
    with pytest.raises(BootstrapCapacityError):
        credentials.mint()
    assert len(credentials._pending) == 2
    now[0] = 11.999
    first_grant = credentials.exchange(first)
    assert credentials.authenticate(first_grant.token)
    third = credentials.mint()
    now[0] = 12.0
    with pytest.raises(BootstrapExchangeError):
        credentials.exchange(second)
    third_grant = credentials.exchange(third)
    assert credentials.authenticate(third_grant.token)
    now[0] = 20
    code = credentials.mint()
    now[0] = 22
    with pytest.raises(BootstrapExchangeError):
        credentials.exchange(code)
    for _ in range(100):
        credentials.mint()
        now[0] += 2
    assert len(credentials._pending) <= 2


@pytest.mark.parametrize(
    "code", [None, False, 23, [], {}, TOKEN, "é" * 43, "x" * 10000]
)
def test_invalid_codes_have_identical_secret_free_error(code):
    credentials = BootstrapCredentials(TOKEN)
    valid = credentials.mint()
    with pytest.raises(BootstrapExchangeError) as failure:
        credentials.exchange(code)
    assert str(failure.value) == "Invalid or expired bootstrap code"
    assert TOKEN not in str(failure.value)
    assert credentials.authenticate(credentials.exchange(valid).token)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl_seconds": 0},
        {"ttl_seconds": -1},
        {"ttl_seconds": float("nan")},
        {"ttl_seconds": float("inf")},
        {"ttl_seconds": True},
        {"session_idle_seconds": 0},
        {"session_absolute_seconds": float("inf")},
        {"max_pending": 0},
        {"max_pending": True},
        {"max_sessions": 0},
        {"max_sessions": True},
        {"max_pending": 1.5},
        {"clock": None},
        {"wall_clock": None},
    ],
)
def test_invalid_pool_configuration_fails_without_secret(kwargs):
    with pytest.raises(ValueError) as failure:
        BootstrapCredentials(TOKEN, **kwargs)
    assert TOKEN not in str(failure.value)


def test_nonfinite_clock_fails_closed_without_consuming_code():
    now = [1.0]
    credentials = BootstrapCredentials(TOKEN, clock=lambda: now[0])
    code = credentials.mint()
    now[0] = float("nan")
    with pytest.raises(BootstrapCapacityError):
        credentials.mint()
    with pytest.raises(BootstrapExchangeError):
        credentials.exchange(code)
    now[0] = 2.0
    assert credentials.authenticate(credentials.exchange(code).token)


def test_matches_token_validates_injected_app_identity():
    credentials = BootstrapCredentials(TOKEN)
    assert credentials.matches_token(TOKEN)
    assert not credentials.matches_token("other")


def test_session_idle_expiry_slides_but_is_exclusive():
    now = [100.0]
    credentials = BootstrapCredentials(
        TOKEN,
        session_idle_seconds=10,
        session_absolute_seconds=100,
        clock=lambda: now[0],
        wall_clock=lambda: 1000.0 + now[0],
    )
    token = credentials.exchange(credentials.mint()).token
    assert credentials.authenticate(token)
    now[0] = 109.999
    assert credentials.authenticate(token)
    now[0] = 119.998
    assert credentials.authenticate(token)
    now[0] = 129.998
    assert not credentials.authenticate(token)


def test_absolute_expiry_bounds_sliding_sessions_and_rotation():
    now = [100.0]
    credentials = BootstrapCredentials(
        TOKEN,
        session_idle_seconds=10,
        session_absolute_seconds=25,
        clock=lambda: now[0],
        wall_clock=lambda: 2000.0 + now[0],
    )
    original = credentials.exchange(credentials.mint())
    now[0] = 109.0
    rotated = credentials.rotate(original.token)
    assert not credentials.authenticate(original.token)
    assert credentials.authenticate(rotated.token)
    now[0] = 118.999
    assert credentials.authenticate(rotated.token)
    now[0] = 124.999
    assert credentials.authenticate(rotated.token)
    now[0] = 125.0
    assert not credentials.authenticate(rotated.token)


def test_session_lru_evicts_oldest_authority_and_rotation_works_at_capacity():
    credentials = BootstrapCredentials(TOKEN, max_sessions=2)
    first = credentials.mint_session()
    second = credentials.mint_session()
    assert len(credentials._sessions) == 2
    assert credentials.authenticate(first.token)
    third = credentials.mint_session()
    assert len(credentials._sessions) == 2
    assert credentials.authenticate(first.token)
    assert not credentials.authenticate(second.token)
    assert credentials.authenticate(third.token)
    rotated = credentials.rotate(first.token)
    assert len(credentials._sessions) == 2
    assert not credentials.authenticate(first.token)
    assert credentials.authenticate(third.token)
    assert credentials.authenticate(rotated.token)
    assert credentials.is_master(TOKEN)
    assert credentials.revoke(third.token)
    assert not credentials.authenticate(third.token)
    assert not credentials.revoke(TOKEN)


def test_bootstrap_exchange_evicts_lru_when_session_pool_is_full():
    credentials = BootstrapCredentials(TOKEN, max_sessions=1)
    first = credentials.mint_session()
    code = credentials.mint()
    exchanged = credentials.exchange(code)
    assert len(credentials._pending) == 0
    assert len(credentials._sessions) == 1
    assert not credentials.authenticate(first.token)
    assert credentials.authenticate(exchanged.token)


def log_record(message, args=()):
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, message, args, None
    )


def test_uvicorn_access_format_redacts_and_preserves_pagination():
    credentials = BootstrapCredentials(TOKEN)
    code = credentials.mint()
    record = log_record(
        '%s - "%s %s HTTP/%s" %d',
        (
            "127.0.0.1:1234",
            "GET",
            f"/?code={code}&token={quote(TOKEN, safe='')}&page=2&limit=25",
            "1.1",
            200,
        ),
    )
    redactor = AccessLogRedactionFilter(credentials, master_token=TOKEN)
    assert redactor.filter(record)
    message = record.getMessage()
    assert code not in message
    assert TOKEN not in message
    assert "page=2&limit=25" in message
    assert 'HTTP/1.1" 200' in message
    from uvicorn.logging import AccessFormatter

    formatted = AccessFormatter(
        '%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False
    ).format(record)
    assert code not in formatted
    assert TOKEN not in formatted
    assert "page=2&limit=25" in formatted


@pytest.mark.parametrize(
    "query",
    [
        "?token=unknown&page=2",
        "?access_token=unknown&page=2",
        "?bootstrap=unknown&page=2",
        "?bootstrap_code=unknown&page=2",
        "?code=unknown&page=2",
        "?%74oken=unknown&page=2",
        "?%2574oken=unknown&page=2",
        "?authorization=unknown&page=2",
        "?token=%E0%A4%A&page=2",
        "?token=unknown;page=2",
        "?TOKEN=unknown&page=2",
    ],
)
def test_query_parameter_redaction_covers_encoding_malformed_and_case(query):
    redactor = AccessLogRedactionFilter(TOKEN)
    message = redactor.redact("GET http://[malformed/" + query)
    assert "unknown" not in message
    assert "%E0%A4%A" not in message
    assert "page=2" in message
    assert "[REDACTED]" in message


def test_unicode_and_benign_encoded_query_are_preserved():
    redactor = AccessLogRedactionFilter(TOKEN)
    assert (
        redactor.redact("GET /search?q=%E6%B3%A2&name=波&page=2")
        == "GET /search?q=%E6%B3%A2&name=波&page=2"
    )


def test_raw_messages_redact_token_headers_and_consumed_bootstrap():
    credentials = BootstrapCredentials(TOKEN)
    code = credentials.mint()
    credentials.exchange(code)
    encoded_code = "".join(f"%{ord(char):02X}" for char in code)
    redactor = AccessLogRedactionFilter(credentials, master_token=TOKEN)
    session = credentials.exchange(credentials.mint()).token
    for text in [
        TOKEN,
        quote(TOKEN, safe=""),
        quote(quote(TOKEN, safe=""), safe=""),
        f"Authorization: Bearer {TOKEN}",
        code,
        encoded_code,
        session,
        "prefix" + code + "suffix",
    ]:
        record = log_record("fallback %s", (text,))
        assert redactor.filter(record)
        assert text not in record.getMessage()
        assert "[REDACTED]" in record.getMessage()
    assert not credentials._pending


def test_filter_malformed_format_fails_closed():
    record = log_record("bad %d", (TOKEN,))
    record.exc_text = TOKEN
    record.stack_info = TOKEN
    AccessLogRedactionFilter(TOKEN).filter(record)
    assert TOKEN not in record.getMessage()
    assert record.exc_text is None
    assert record.stack_info is None


def test_explicit_install_does_not_mutate_unrelated_logger():
    logger = logging.Logger("test-isolated-access")
    other = logging.Logger("test-isolated-other")
    redactor = install_access_log_redaction(TOKEN, logger)
    assert logger.filters == [redactor]
    assert not other.filters
