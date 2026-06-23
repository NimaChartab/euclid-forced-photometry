"""Retry-with-backoff for the package's network calls.

IRSA TAP/SIA, S3 byte-range reads and unwise.me downloads fail
transiently; every network call goes through :func:`retry` (exponential
backoff with jitter, logged through the package logger).
"""
from __future__ import annotations

import random
import time

from ._log import logger

# OSError covers socket/timeout errors (s3fs, urllib); pyvo/astroquery raise
# requests exceptions, which subclass OSError in requests>=2.
_DEFAULT_EXCEPTIONS: tuple = (OSError, TimeoutError)

# fsspec kwargs for every S3 open. Without explicit botocore timeouts a
# stalled byte-range read blocks indefinitely (the server can hold a dead
# connection open); with them it raises and the caller's retry takes over.
S3_FSSPEC_KWARGS = {
    "anon": True,
    "config_kwargs": {
        "connect_timeout": 30,
        "read_timeout": 60,
        "retries": {"max_attempts": 5, "mode": "adaptive"},
    },
}


def retry(fn, *, attempts: int = 3, base_delay: float = 2.0,
          exceptions: tuple = _DEFAULT_EXCEPTIONS, what: str = ""):
    """Call ``fn()`` with up to ``attempts`` tries and exponential backoff.

    ``fn`` is a zero-argument callable (close over arguments). Delays are
    ``base_delay * 2**k`` with +/-25% jitter. The last failure re-raises.

    Only ``exceptions`` are retried; a ``ValueError`` (bad request, missing
    product) fails immediately.
    """
    last = None
    for k in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            last = exc
            if k == attempts - 1:
                break
            delay = base_delay * (2 ** k) * random.uniform(0.75, 1.25)
            logger.warning("%s failed (%r); retry %d/%d in %.1fs",
                           what or getattr(fn, "__name__", "network call"),
                           exc, k + 1, attempts - 1, delay)
            time.sleep(delay)
    raise last
