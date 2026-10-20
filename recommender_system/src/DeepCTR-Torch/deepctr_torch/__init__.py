import os

from . import layers
from . import models
from .utils import check_version

__version__ = '0.2.9'
if os.environ.get("EASYRL4REC_CHECK_DEEPCTR_VERSION") == "1":
    check_version(__version__)
