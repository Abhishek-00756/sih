"""ANPR unit tests using an injected fake OCR reader (no EasyOCR weights)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anpr_manager import ANPRManager


class FakeReader:
    def __init__(self, results):
        self.results = results
        self.calls = 0
        self.last_image = None

    def readtext(self, image):
        self.calls += 1
        self.last_image = image
        return self.results


def _crop(h=180, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_clean_text_strips_noise():
    assert ANPRManager.clean_text("ab-12 34!") == "AB1234"
    assert ANPRManager.clean_text("  mh 12 xy 3456 ") == "MH12XY3456"


def test_reads_highest_confidence_plate():
    reader = FakeReader(
        [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "sticker", 0.99),
            ([[0, 0], [40, 0], [40, 12], [0, 12]], "MH12AB1234", 0.81),
            ([[0, 0], [40, 0], [40, 12], [0, 12]], "KA01CD99", 0.62),
        ]
    )
    anpr = ANPRManager(min_confidence=0.5, reader=reader)
    text, conf = anpr.read_license_plate(_crop())
    assert text == "MH12AB1234"
    assert conf == 0.81
    assert reader.calls == 1
    assert reader.last_image.ndim == 2


def test_rejects_short_or_low_confidence_text():
    reader = FakeReader(
        [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "AB1", 0.99),
            ([[0, 0], [40, 0], [40, 12], [0, 12]], "MH12AB1234", 0.2),
        ]
    )
    anpr = ANPRManager(min_confidence=0.5, reader=reader)
    text, conf = anpr.read_license_plate(_crop())
    assert text is None
    assert conf == 0.0


def test_empty_crop_and_disabled_reader():
    anpr = ANPRManager(reader=FakeReader([([[0, 0], [1, 0], [1, 1], [0, 1]], "ABC1234", 0.9)]))
    assert anpr.read_license_plate(None) == (None, 0.0)
    assert anpr.read_license_plate(np.zeros((0, 0, 3), dtype=np.uint8)) == (None, 0.0)
    assert anpr.read_license_plate(np.zeros((4, 4, 3), dtype=np.uint8)) == (None, 0.0)

    disabled = ANPRManager(reader=FakeReader([]))
    disabled.reader = None
    assert disabled.read_license_plate(_crop()) == (None, 0.0)


def test_crop_is_readable_requires_nearby_vehicle():
    anpr = ANPRManager(reader=FakeReader([]))
    assert anpr.crop_is_readable(200, 180) is True
    assert anpr.crop_is_readable(149, 200) is False
    assert anpr.crop_is_readable(200, 149) is False


def test_rejects_invalid_constructor_args():
    try:
        ANPRManager(min_confidence=1.5, reader=FakeReader([]))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        ANPRManager(min_plate_len=8, max_plate_len=3, reader=FakeReader([]))
        assert False, "expected ValueError"
    except ValueError:
        pass
