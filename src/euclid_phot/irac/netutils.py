"""Small verified-HTTPS helpers used by the standalone pipeline."""

from __future__ import annotations

import os
import shutil
import ssl
import time
import urllib.request
from pathlib import Path


def ensure_https_ca_bundle() -> None:
    """Use certifi only when Python has no usable configured CA store."""
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return
    paths = ssl.get_default_verify_paths()
    if ((paths.cafile and Path(paths.cafile).is_file())
            or (paths.capath and Path(paths.capath).is_dir())):
        return
    try:
        import certifi
    except ImportError:
        return
    bundle = certifi.where()
    if Path(bundle).is_file():
        os.environ["SSL_CERT_FILE"] = bundle


def https_ssl_context() -> ssl.SSLContext:
    """Return a verified context that also works with aiohttp/fsspec.

    Some Python installations do not have a usable OpenSSL CA path.  Setting
    ``SSL_CERT_FILE`` is enough for urllib, but aiohttp may create its default
    context before that environment variable is set.  Supplying this context
    explicitly keeps remote FITS range reads verified and deterministic.
    """
    ensure_https_ca_bundle()
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return ssl.create_default_context(cafile=configured)
    return ssl.create_default_context()


def download_file(url: str, path: str | Path, *, force: bool = False,
                  timeout: float = 120.0) -> Path:
    """Download a file atomically over verified HTTPS."""
    path = Path(path)
    if path.exists() and not force:
        return path
    ensure_https_ca_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, \
                temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def retry(function, *, attempts: int = 3, delay_seconds: float = 2.0,
          what: str = "network operation"):
    """Retry a read-only network operation, preserving its final exception."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except Exception as error:  # network libraries expose several types
            last_error = error
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(f"{what} failed after {attempts} attempts") from last_error
