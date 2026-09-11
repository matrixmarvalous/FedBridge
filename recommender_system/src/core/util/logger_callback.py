# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 7:56 下午
# @Author  : Chongming GAO
# @FileName: utils.py


import logzero
import torch
from logzero import logger
import os
# from tensorflow.python.keras.callbacks import Callback

# from util.upload import my_upload
import re

# Codex-modified 2026-08-14: remove an unused private container path while
# retaining an opt-in compatibility hook for downstream code.
REMOTE_ROOT = os.environ.get("EASYRL4REC_REMOTE_ROOT", "")
