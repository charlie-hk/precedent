"""Feedback-driven learning loop — RESERVED.

This module defines the interface for the part the brief calls "genuine
machine-learning capability": learning a similarity metric and a setup-quality
score from human-labelled examples, improving recognition over time, comparing
model versions and validating on unseen data.

The interface and the evaluation contract are shown here so the architecture is
clear and testable. The fitted metric, the labelling schema and the
model-versioning/promotion logic are the parts that carry the edge and are
delivered under a paid engagement, not published in this demo. Every method
below raises NotReserved to make that explicit rather than shipping a fake
implementation.

See README, section "What is reserved".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ReservedComponent(NotImplementedError):
    """Raised by reserved components that are not part of the public demo."""


@dataclass
class LabeledExample:
    """One human decision on a detected structure: was it a valid setup, and how did
    the reviewer rate it. This is the training signal for the learned metric."""
    query_index: int
    label: str  # e.g. "valid", "invalid", "ambiguous"
    reviewer_score: float  # 0..1


class LearnedMetric:
    """Learns a distance metric so that analogs with similar *outcomes* are pulled
    together, instead of using raw cosine distance on the baseline features.

    Public demo uses the baseline cosine metric in `similarity.py`. The fitted
    metric is reserved.
    """

    def fit(self, X: np.ndarray, examples: list[LabeledExample]) -> "LearnedMetric":
        raise ReservedComponent(
            "The learned similarity metric is reserved. The public demo uses the "
            "baseline cosine metric in similarity.AnalogSearch."
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise ReservedComponent("Reserved. See README, 'What is reserved'.")


class SetupQualityModel:
    """Scores how trustworthy a precedent is (analog agreement, dispersion,
    structure context) into a single 0..1 quality. Reserved."""

    def fit(self, features: np.ndarray, examples: list[LabeledExample]) -> "SetupQualityModel":
        raise ReservedComponent("The setup-quality model is reserved.")

    def score(self, features: np.ndarray) -> float:
        raise ReservedComponent("Reserved. See README, 'What is reserved'.")


def promote_if_better(candidate_metrics: dict, champion_metrics: dict) -> bool:
    """Model-versioning gate: only promote a new model if it beats the current one
    on held-out, out-of-sample data by a margin. Reserved."""
    raise ReservedComponent("Version promotion logic is reserved.")
