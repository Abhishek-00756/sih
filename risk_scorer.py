"""Context-aware threat scores so operators see high-risk alerts first.

Base 10. +40 inbound tripwire. +20 night (20:00-06:00 local time).
+30 loitering over 60s.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
        timezone_name: str = "Asia/Kolkata",
    ) -> None:
        if not 0 <= int(night_start_hour) <= 23:
            raise ValueError("night_start_hour must be between 0 and 23")
        if not 0 <= int(night_end_hour) <= 23:
            raise ValueError("night_end_hour must be between 0 and 23")
        if loiter_seconds <= 0:
            raise ValueError("loiter_seconds must be positive")
        self.night_start_hour = int(night_start_hour)
        self.night_end_hour = int(night_end_hour)
        self.loiter_seconds = float(loiter_seconds)
        self.timezone_name = str(timezone_name)
        try:
            self.timezone = ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {self.timezone_name}") from exc

    def _local_hour(self, timestamp: Optional[float] = None) -> int:
        if timestamp is None:
            current = datetime.now(self.timezone)
        else:
            current = datetime.fromtimestamp(float(timestamp), tz=self.timezone)
        return current.hour

    def is_night(self, timestamp: Optional[float] = None) -> bool:
        hour = self._local_hour(timestamp)
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
