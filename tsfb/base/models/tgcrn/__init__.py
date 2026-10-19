from __future__ import annotations

from functools import partial
from typing import Dict

from tsfb.base.models.tgcrn.tgcrn_adapter import TGCRNAdapter

TGCRN: Dict = {
    "model_factory": partial(TGCRNAdapter),
    "model_name": "tgcrn.TGCRN",
}


__all__ = ["TGCRN"]
