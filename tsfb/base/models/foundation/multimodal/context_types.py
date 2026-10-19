"""
Context type dataclasses for multi-modal foundation models.

These types represent non-time-series modalities that can be injected
alongside a time series to condition a multi-modal foundation model.
Currently used as interface placeholders; concrete implementations will
follow when multi-modal models are integrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TextContext:
    """
    Textual context associated with a time series window.

    Examples: news headlines, event descriptions, product metadata.
    """

    text: str
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    language: str = "en"


@dataclass
class ImageContext:
    """
    Image context associated with a time series window.

    Examples: satellite imagery, traffic camera frames, store photos.
    The image array should be in HWC format (height, width, channels).
    """

    image: np.ndarray
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TabularContext:
    """
    Tabular feature context aligned with a time series.

    Examples: external regressors, static metadata, weather features.
    The DataFrame index should align with the time series index.
    """

    features: pd.DataFrame
    feature_names: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = list(self.features.columns)
