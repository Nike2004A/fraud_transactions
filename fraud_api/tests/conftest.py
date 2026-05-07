"""Configura ``sys.path`` para que ``import src.main`` funcione cuando se
corre ``pytest`` desde la raíz del repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FRAUD_API = Path(__file__).resolve().parents[1]
if str(_FRAUD_API) not in sys.path:
    sys.path.insert(0, str(_FRAUD_API))
