"""
Tests for src/preprocessing.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    get_scene_data, get_scene_indices, classify_scene_type,
    get_scene_images, get_scene_duration_ms,
    extract_fixations, extract_blinks,
    compute_scene_quality, compute_session_quality,
)

class TestSceneSlicing:
    def test_get_scene_data_returns_only_matching_rows(self, two_image_scene):
        result = get_scene_data(two_image_scene, 1)
        assert len(result) == len(two_image_scene)

    def test_get_scene_data_returns_empty_for_missing_scene(self, two_image_scene):
        assert len(get_scene_data(two_image_scene, 999)) == 0

    def test_classify_scene_type_odd_is_stimulus(self):
        for idx in [1, 3, 99]:
            assert classify_scene_type(idx) == "stimulus"

    def test_classify_scene_type_even_is_fixation_cross(self):
        for idx in [0, 2, 100]:
            assert classify_scene_type(idx) == "fixation_cross"

class TestGetSceneImages:
    def test_returns_real_image_ids(self, two_image_scene):
        assert set(get_scene_images(two_image_scene)) == {"img_A", "img_B"}

    def test_excludes_no_image_sentinel(self):
        df = pd.DataFrame({"IMAGE": ["img_A", "no_image", "img_B", "no_image"]})
        assert set(get_scene_images(df)) == {"img_A", "img_B"}

    def test_excludes_empty_string(self):
        df = pd.DataFrame({"IMAGE": ["img_A", "", "img_B"]})
        assert set(get_scene_images(df)) == {"img_A", "img_B"}

    def test_returns_empty_when_no_image_column(self):
        assert get_scene_images(pd.DataFrame({"TIMESTAMP": [0, 1, 2]})) == []

class TestSceneDuration:
    def test_duration_spans_min_to_max_timestamp(self):
        df = pd.DataFrame({"TIMESTAMP": [100.0, 200.0, 500.0, 1500.0]})
        assert get_scene_duration_ms(df) == 1400.0

    def test_single_row_has_zero_duration(self):
        assert get_scene_duration_ms(pd.DataFrame({"TIMESTAMP": [100.0]})) == 0.0

    def test_empty_df_has_zero_duration(self):
        assert get_scene_duration_ms(pd.DataFrame({"TIMESTAMP": []})) == 0.0

class TestExtractFixations:
    def test_empty_scene_returns_empty_df(self, empty_scene):
        assert len(extract_fixations(empty_scene)) == 0

    def test_scene_without_fixation_events_returns_empty(self, no_fixation_scene):
        assert len(extract_fixations(no_fixation_scene)) == 0

    def test_single_fixation_extracted_correctly(self, single_fixation_scene):
        result = extract_fixations(single_fixation_scene)
        assert len(result) == 1
        assert result.iloc[0]["image"] == "img_A"
        assert result.iloc[0]["duration_ms"] == 200.0

    def test_returns_df_with_expected_columns(self, single_fixation_scene):
        result = extract_fixations(single_fixation_scene)
        expected = {"start_timestamp", "end_timestamp", "duration_ms", "fpog_x", "fpog_y", "rx", "ry", "image", "n_samples"}
        assert expected.issubset(set(result.columns))

class TestExtractBlinks:
    def test_empty_scene_returns_empty(self, empty_scene):
        assert len(extract_blinks(empty_scene)) == 0

    def test_scene_without_blinks_returns_empty(self, single_fixation_scene):
        assert len(extract_blinks(single_fixation_scene)) == 0

    def test_two_blinks_identified(self, blink_scene):
        assert len(extract_blinks(blink_scene)) == 2

    def test_blink_at_scene_start_detected(self):
        from tests.conftest import _row
        rows = [_row(i * 50, blink=True) for i in range(3)]
        rows += [_row(i * 50) for i in range(3, 10)]
        result = extract_blinks(pd.DataFrame(rows))
        assert len(result) == 1
        assert result.iloc[0]["start_timestamp"] == 0.0

    def test_blink_at_scene_end_detected(self):
        from tests.conftest import _row
        rows = [_row(i * 50) for i in range(7)]
        rows += [_row(i * 50, blink=True) for i in range(7, 10)]
        result = extract_blinks(pd.DataFrame(rows))
        assert len(result) == 1

class TestSceneQuality:
    def _generate_scene(self, n=500, duration_ms=5000, blink_pct=0.0, missing_pct=0.0):
        rows = []
        for i in range(n):
            ts = i * (duration_ms / n)
            is_blink = i < int(n * blink_pct)
            is_missing = int(n * blink_pct) <= i < int(n * blink_pct) + int(n * missing_pct)
            fev = 1 if i == 0 else (3 if i == n - 1 else 2)
            rows.append({
                "TIMESTAMP": float(ts), "TEV": 1, "FEV": fev,
                "FDUR": 100.0 if fev == 3 else 0.0,
                "FPOGX": np.nan if is_missing else 0.5,
                "FPOGY": np.nan if is_missing else 0.5,
                "RX": np.nan if (is_blink or is_missing) else 0.5,
                "RY": np.nan if (is_blink or is_missing) else 0.5,
                "BLINK": is_blink,
                "IMAGE": "img_A",
                "SCENE_INDEX": 1,
            })
        return pd.DataFrame(rows)

    def test_empty_scene_is_invalid(self, empty_scene):
        q = compute_scene_quality(empty_scene)
        assert not q["is_valid"]
        assert q["n_samples"] == 0

    def test_too_few_samples_is_invalid(self):
        q = compute_scene_quality(self._generate_scene(n=10, duration_ms=5000))
        assert not q["is_valid"]

    def test_too_much_missing_gaze_is_invalid(self):
        q = compute_scene_quality(self._generate_scene(n=500, missing_pct=0.60))
        assert not q["is_valid"]

    def test_too_many_blinks_is_invalid(self):
        q = compute_scene_quality(self._generate_scene(n=500, blink_pct=0.50))
        assert not q["is_valid"]

class TestSessionQuality:
    def test_empty_session_is_invalid(self, empty_scene):
        q = compute_session_quality(empty_scene)
        assert not q["is_valid"]
        assert q["total_samples"] == 0

    def test_too_few_samples_is_invalid(self):
        df = pd.DataFrame({
            "TIMESTAMP": np.linspace(0, 60_000, 50),
            "BLINK": [False] * 50,
            "RX": [0.5] * 50, "RY": [0.5] * 50,
            "SCENE_INDEX": list(range(50)),
        })
        assert not compute_session_quality(df)["is_valid"]

    def test_too_few_scenes_is_invalid(self):
        df = pd.DataFrame({
            "TIMESTAMP": np.linspace(0, 60_000, 500),
            "BLINK": [False] * 500,
            "RX": [0.5] * 500, "RY": [0.5] * 500,
            "SCENE_INDEX": [i % 5 for i in range(500)],
        })
        assert not compute_session_quality(df)["is_valid"]

    def test_good_session_is_valid(self):
        df = pd.DataFrame({
            "TIMESTAMP": np.linspace(0, 60_000, 500),
            "BLINK": [False] * 500,
            "RX": [0.5] * 500, "RY": [0.5] * 500,
            "SCENE_INDEX": [i % 20 for i in range(500)],
        })
        assert compute_session_quality(df)["is_valid"]
