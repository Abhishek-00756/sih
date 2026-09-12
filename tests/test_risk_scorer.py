"""Context-aware risk scoring tests."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_scorer import RiskScorer


def _ts(hour: int) -> float:
    return datetime(2026, 1, 15, hour, 0, tzinfo=timezone.utc).timestamp()


def test_base_score_daytime():
    scorer = RiskScorer()
    score, reasons = scorer.score(timestamp=_ts(12))
    assert score == 10
    assert reasons == ["base"]
    assert scorer.label(score) == "LOW"


def test_inbound_night_loiter_stacks():
    scorer = RiskScorer()
    score, reasons = scorer.score(
        direction="INBOUND",
        timestamp=_ts(22),
        dwell_time=90.0,
    )
    assert score == 100
    assert "inbound_tripwire" in reasons
    assert "night" in reasons
    assert "loitering" in reasons
    assert scorer.label(score) == "CRITICAL"


def test_night_window_wraps_midnight():
    scorer = RiskScorer(night_start_hour=20, night_end_hour=6)
    assert scorer.is_night(_ts(21)) is True
    assert scorer.is_night(_ts(3)) is True
    assert scorer.is_night(_ts(12)) is False
