#!/usr/bin/env python3
"""
scopus2txtsummary.py
...
"""

import sys

# ✅ FORCE UTF-8 OUTPUT (fixes Windows emoji crash)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import re
import glob
import argparse
from typing import List

import pandas as pd
