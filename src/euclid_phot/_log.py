"""Package logger.

The library never configures logging on import (a ``NullHandler`` keeps it
silent by default, per library convention); applications and the CLI opt in
via :func:`configure_logging`. The pipeline's ``verbose=True`` progress
prints are kept as plain stdout prints for notebook back-compat; the logger
carries the operational messages (retries, fallbacks, cell progress).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("euclid_phot")
logger.addHandler(logging.NullHandler())


def configure_logging(level="INFO", logfile=None) -> logging.Logger:
    """Attach a stream (and optional file) handler to the package logger.

    Safe to call repeatedly: existing non-Null handlers are replaced, not
    duplicated.

    Parameters
    ----------
    level : str or int
        Logging level for the package logger (e.g. ``"DEBUG"``).
    logfile : str or Path, optional
        Also append records to this file.
    """
    for h in list(logger.handlers):
        if not isinstance(h, logging.NullHandler):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.setLevel(level)
    return logger
