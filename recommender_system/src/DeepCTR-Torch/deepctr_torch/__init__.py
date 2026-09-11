import os

from . import layers
from . import models
from .utils import check_version

__version__ = '0.2.9'
# Codex-modified 2026-08-14: avoid eight unsolicited, potentially blocking
# network requests when MPI workers import the vendored package on an HPC node.
if os.environ.get("EASYRL4REC_CHECK_DEEPCTR_VERSION") == "1":
    check_version(__version__)
