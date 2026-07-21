"""Logging setup and the DEV switch, shared by the app and the utils it drives.

Two things this bench needs and did not have:

* **Traceable runs.** A segmentation is a chain of stages -- render, describe,
  paint, snap, seed, infer, lift -- run partly in subprocesses and partly
  against a remote model. Without a line per stage there is no way to tell a
  run that skipped the VLM from one where the VLM returned nothing.
* **Loud failures.** ``GeoSAM2Segmenter.process`` was written to never raise, so
  a caller has one code path. That is the right shape for a service and the
  wrong one for a bench: the error arrives as a string, stripped of its
  traceback, at the point where the run is already over.

``DEV`` reconciles the two. On (the default), an exception is logged with its
traceback and re-raised at the point it happened. Off, the old
error-in-a-dict behaviour is restored. Set ``GEOSAM2_DEV=0`` for that.
"""

from __future__ import annotations

import logging
import os

# The logger name is bracketed and butted straight against the message, whose
# own convention is to open with a [stage] tag -- so a line reads
# "[geosam2.pipeline][inference] ..." with no gap between the two brackets.
_FORMAT = "%(asctime)s  %(levelname)s  [%(name)s]%(message)s"
_DATEFMT = "%H:%M:%S"


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


# On unless explicitly disabled: this repository is an evaluation bench, and a
# run that fails quietly and reports twelve parts costs more to diagnose than a
# traceback costs to read.
DEV = _flag("GEOSAM2_DEV", "1")

# DEBUG adds the per-line output of the inference subprocesses.
LEVEL = os.environ.get("GEOSAM2_LOG_LEVEL", "INFO").upper()

# Libraries that narrate their own startup at INFO. Importing pyrender alone
# emits several lines about OpenGL acceleration, which at the head of every run
# reads as pipeline output and is not.
_NOISY = ("OpenGL", "PIL", "httpx", "httpcore", "matplotlib", "urllib3", "trimesh")


def configure() -> None:
    """Install a console handler, unless the process already has one.

    ``basicConfig`` is a no-op once the root logger has handlers, so an
    application that sets up its own logging keeps it, and importing any module
    here does not silently reformat someone else's logs.
    """
    logging.basicConfig(level=LEVEL, format=_FORMAT, datefmt=_DATEFMT)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """A named logger, with console output configured on first use.

    Library modules normally must not touch the root logger. This one does,
    because every entry point here is a script or a local bench server, and a
    stage log nobody configured a handler for is a stage log nobody reads.
    """
    configure()
    return logging.getLogger(name)
