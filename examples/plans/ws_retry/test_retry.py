"""Deterministic spec-test for retry_util.py (stdlib only, no real sleeping).

The agentic job must create retry_util.py so this passes. Do not modify this file."""
import retry_util

# (1) fails twice then succeeds -> returns 'ok' on the 3rd call
calls = {"n": 0}
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ValueError("transient")
    return "ok"
assert retry_util.fetch_with_retry(flaky) == "ok", "should succeed after retries"
assert calls["n"] == 3, f"expected exactly 3 calls, got {calls['n']}"

# (2) always fails -> re-raises the last exception after exhausting attempts
def always():
    raise RuntimeError("permanent")
try:
    retry_util.fetch_with_retry(always)
    raise SystemExit("FAIL: expected an exception after exhausting attempts")
except RuntimeError:
    pass

# (3) retry() returns immediately on first success, calling fn exactly once
calls2 = {"n": 0}
def good():
    calls2["n"] += 1
    return 42
assert retry_util.retry(good) == 42
assert calls2["n"] == 1, f"expected exactly 1 call, got {calls2['n']}"

print("OK all retry tests passed")
