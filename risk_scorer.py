"""Context-aware threat scores so operators see high-risk alerts first.

Base 10. +40 inbound tripwire. +20 night (20:00-06:00). +30 loitering over 60s.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

RiskResult = Tuple[int, List[str]]


class RiskScorer:
    BASE_SCORE = 10
    INBOUND_BONUS = 40
    NIGHT_BONUS = 20
    LOITER_BONUS = 30
    LOITER_SECONDS = 60.0
    MAX_SCORE = 100

    def __init__(
        self,
        night_start_hour: int = 20,
        night_end_hour: int = 6,
        loiter_seconds: float = LOITER_SECONDS,
    ) -> None:
        self.night_start_hour = int(night_start_hour)
        self.night_end_hour = int(night_end_hour)
        self.loiter_seconds = float(loiter_seconds)

    def is_night(self, timestamp: Optional[float] = None) -> bool:
        if timestamp is None:
            hour = datetime.now().hour
        else:
            hour = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).hour
        start = self.night_start_hour
        end = self.night_end_hour
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end

    def score(
        self,
        event_type: str = "",
        direction: Optional[str] = None,
        timestamp: Optional[float] = None,
        dwell_time: Optional[float] = None,
        factors: Optional[Sequence[str]] = None,
    ) -> RiskResult:
        total = self.BASE_SCORE
        reasons: List[str] = ["base"]
        extra = {str(item).lower() for item in (factors or [])}

        inbound = str(direction or "").upper() == "INBOUND" or "inbound" in extra
        if inbound:
            total += self.INBOUND_BONUS
            reasons.append("inbound_tripwire")

        if self.is_night(timestamp) or "night" in extra:
            total += self.NIGHT_BONUS
            reasons.append("night")

        loiter = dwell_time is not None and float(dwell_time) > self.loiter_seconds
        if loiter or "loitering" in extra:
            total += self.LOITER_BONUS
            reasons.append("loitering")

        if "geofence" in extra or "intrusion" in str(event_type).lower():
            if "geofence" not in reasons:
                reasons.append("geofence")

        return min(self.MAX_SCORE, int(total)), reasons

    def label(self, score: int) -> str:
        if score >= 70:
            return "CRITICAL"
        if score >= 40:
            return "HIGH"
        if score >= 20:
            return "ELEVATED"
        return "LOW"
