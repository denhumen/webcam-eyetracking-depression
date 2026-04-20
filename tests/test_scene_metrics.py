"""
Tests for src/scene_metrics.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.scene_metrics import (
    derive_valence, mean_fixation_duration, total_fixation_duration, fixation_rate,
    dwell_time_per_image, fixation_count_per_image, fixation_proportion_per_image,
    _pair_bias, first_fixation_image, first_fixation_duration,
    time_to_first_fixation_on_image, revisit_count_per_image, dwell_time_first_epoch,
    scanpath_length, saccade_count, saccade_rate, mean_saccade_amplitude,
    blink_count, blink_rate,
    gaze_transition_matrix, transition_matrix_density, gaze_transition_entropy,
)

class TestDeriveValence:
    def test_neutral_category_returns_neutral(self):
        assert derive_valence("neutral", []) == "neutral"

    def test_neutral_category_with_trailing_whitespace(self):
        assert derive_valence("neutral  ", []) == "neutral"

    def test_sad_face_is_negative(self):
        assert derive_valence("stimulus", ["sad face"]) == "negative"

    def test_sad_alone_is_negative(self):
        assert derive_valence("stimulus", ["sad"]) == "negative"

    def test_funeral_is_negative(self):
        assert derive_valence("stimulus", ["funeral"]) == "negative"

    def test_happy_face_is_positive(self):
        assert derive_valence("stimulus", ["happy face"]) == "positive"

    def test_positive_keyword_is_positive(self):
        assert derive_valence("stimulus", ["positive"]) == "positive"

    def test_dep_alone_is_other(self):
        assert derive_valence("stimulus", ["dep"]) == "other"

    def test_happy_face_plus_dep_is_positive(self):
        assert derive_valence("stimulus", ["happy face", "dep"]) == "positive"

    def test_empty_labels_is_other(self):
        assert derive_valence("stimulus", []) == "other"

    def test_labels_are_case_insensitive(self):
        assert derive_valence("stimulus", ["SAD FACE"]) == "negative"
        assert derive_valence("stimulus", ["Happy Face"]) == "positive"


class TestBasicCountMetrics:
    def test_mean_fixation_duration_empty_is_nan(self, fixations_df_empty):
        assert np.isnan(mean_fixation_duration(fixations_df_empty))

    def test_mean_fixation_duration_value(self, fixations_df_simple):
        assert mean_fixation_duration(fixations_df_simple) == 200.0

    def test_total_fixation_duration_empty_is_zero(self, fixations_df_empty):
        assert total_fixation_duration(fixations_df_empty) == 0.0

    def test_total_fixation_duration_value(self, fixations_df_simple):
        assert total_fixation_duration(fixations_df_simple) == 400.0

    def test_fixation_rate_zero_duration_returns_nan(self, fixations_df_simple):
        assert np.isnan(fixation_rate(fixations_df_simple, 0.0))

    def test_fixation_rate_correct_value(self, fixations_df_simple):
        assert fixation_rate(fixations_df_simple, 1000.0) == 2.0


class TestPerImageMetrics:
    def test_dwell_time_per_image(self, fixations_df_simple):
        assert dwell_time_per_image(fixations_df_simple) == {"img_A": 200.0, "img_B": 200.0}

    def test_dwell_time_empty(self, fixations_df_empty):
        assert dwell_time_per_image(fixations_df_empty) == {}

    def test_fixation_count_per_image(self, fixations_df_simple):
        assert fixation_count_per_image(fixations_df_simple) == {"img_A": 1, "img_B": 1}

    def test_fixation_proportion_per_image_sums_to_one(self, fixations_df_simple):
        result = fixation_proportion_per_image(fixations_df_simple)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_fixation_proportion_balanced_scene(self, fixations_df_simple):
        result = fixation_proportion_per_image(fixations_df_simple)
        assert result["img_A"] == pytest.approx(0.5)
        assert result["img_B"] == pytest.approx(0.5)


class TestPairBias:
    def test_nan_when_stimulus_config_is_none(self, fixations_df_simple):
        result = _pair_bias(fixations_df_simple, None, ["img_A", "img_B"], "negative", "positive")
        assert np.isnan(result)

    def test_nan_when_no_images(self, fixations_df_simple, stimulus_config_basic):
        result = _pair_bias(fixations_df_simple, stimulus_config_basic, [], "negative", "positive")
        assert np.isnan(result)

    def test_nan_when_only_one_valence_present(self, fixations_df_single, stimulus_config_basic):
        result = _pair_bias(fixations_df_single, stimulus_config_basic, ["img_A"], "negative", "positive")
        assert np.isnan(result)

    def test_balanced_dwell_gives_zero(self, fixations_df_simple, stimulus_config_basic):
        result = _pair_bias(fixations_df_simple, stimulus_config_basic, ["img_A", "img_B"], "negative", "positive")
        assert result == pytest.approx(0.0)

    def test_full_dwell_on_a_gives_positive_one(self, stimulus_config_basic):
        fixations = pd.DataFrame([{
            "start_timestamp": 0.0, "end_timestamp": 500.0, "duration_ms": 500.0, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 30,
        }])
        result = _pair_bias(fixations, stimulus_config_basic, ["img_A", "img_B"], "negative", "positive")
        assert result == pytest.approx(1.0)

    def test_skewed_dwell_gives_intermediate_value(self, stimulus_config_basic):
        fixations = pd.DataFrame([
            {"start_timestamp": 0.0, "end_timestamp": 300.0, "duration_ms": 300.0, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 20},
            {"start_timestamp": 400.0, "end_timestamp": 500.0, "duration_ms": 100.0, "rx": 0.7, "ry": 0.6, "image": "img_B", "n_samples": 7},
        ])
        result = _pair_bias(fixations, stimulus_config_basic, ["img_A", "img_B"], "negative", "positive")
        assert result == pytest.approx(0.5)

    def test_bias_is_anti_symmetric(self, stimulus_config_basic):
        fixations = pd.DataFrame([
            {"start_timestamp": 0.0, "end_timestamp": 300.0, "duration_ms": 300.0, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 20},
            {"start_timestamp": 400.0, "end_timestamp": 500.0, "duration_ms": 100.0, "rx": 0.7, "ry": 0.6, "image": "img_B", "n_samples": 7},
        ])
        ab = _pair_bias(fixations, stimulus_config_basic, ["img_A", "img_B"], "negative", "positive")
        ba = _pair_bias(fixations, stimulus_config_basic, ["img_A", "img_B"], "positive", "negative")
        assert ab == pytest.approx(-ba)

    def test_bias_bounded_minus_one_to_one(self, stimulus_config_basic):
        fixations = pd.DataFrame([
            {"start_timestamp": 0.0, "end_timestamp": 999.0, "duration_ms": 999.0, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 60},
            {"start_timestamp": 1000.0, "end_timestamp": 1001.0, "duration_ms": 1.0, "rx": 0.7, "ry": 0.6, "image": "img_B", "n_samples": 2},
        ])
        result = _pair_bias(fixations, stimulus_config_basic, ["img_A", "img_B"], "negative", "positive")
        assert -1.0 <= result <= 1.0


class TestFirstFixation:
    def test_first_fixation_image_empty(self, fixations_df_empty):
        assert first_fixation_image(fixations_df_empty) is None

    def test_first_fixation_image_correct(self, fixations_df_simple):
        assert first_fixation_image(fixations_df_simple) == "img_A"

    def test_first_fixation_duration_empty_is_nan(self, fixations_df_empty):
        assert np.isnan(first_fixation_duration(fixations_df_empty))

    def test_first_fixation_duration_value(self, fixations_df_simple):
        assert first_fixation_duration(fixations_df_simple) == 200.0

class TestTimeToFirstFixation:
    def test_ttff_returns_nan_when_target_not_visited(self, fixations_df_simple):
        result = time_to_first_fixation_on_image(fixations_df_simple, scene_start_ts=0.0, image_id="img_MISSING")
        assert np.isnan(result)

    def test_ttff_computed_relative_to_scene_start(self, fixations_df_simple):
        result = time_to_first_fixation_on_image(fixations_df_simple, scene_start_ts=0.0, image_id="img_A")
        assert result == 0.0

    def test_ttff_for_second_image(self, fixations_df_simple):
        result = time_to_first_fixation_on_image(fixations_df_simple, scene_start_ts=0.0, image_id="img_B")
        assert result == 320.0

class TestRevisitCount:
    def test_empty_returns_empty_dict(self, fixations_df_empty):
        assert revisit_count_per_image(fixations_df_empty) == {}

    def test_no_revisits_when_each_image_visited_once(self, fixations_df_simple):
        assert revisit_count_per_image(fixations_df_simple) == {"img_A": 0, "img_B": 0}

    def test_revisit_counted_for_abA_sequence(self):
        fixations = pd.DataFrame([
            {"start_timestamp": 0, "end_timestamp": 100, "duration_ms": 100, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 7},
            {"start_timestamp": 150, "end_timestamp": 250, "duration_ms": 100, "rx": 0.7, "ry": 0.6, "image": "img_B", "n_samples": 7},
            {"start_timestamp": 300, "end_timestamp": 400, "duration_ms": 100, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 7},
        ])
        assert revisit_count_per_image(fixations) == {"img_A": 1, "img_B": 0}

class TestDwellTimeFirstEpoch:
    def test_fixation_entirely_within_window(self):
        fixations = pd.DataFrame([{
            "start_timestamp": 0.0, "end_timestamp": 200.0, "duration_ms": 200.0, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 13,
        }])
        result = dwell_time_first_epoch(fixations, scene_start_ts=0.0, epoch_ms=500.0)
        assert result["img_A"] == 200.0

    def test_fixation_entirely_after_window(self):
        fixations = pd.DataFrame([{
            "start_timestamp": 600.0, "end_timestamp": 800.0, "duration_ms": 200.0, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 13,
        }])
        result = dwell_time_first_epoch(fixations, scene_start_ts=0.0, epoch_ms=500.0)
        assert result == {}

    def test_fixation_straddles_window_boundary(self):
        """Regression: before the fix, full 300ms would be counted. After, only
        the 400-500 overlap (100ms) is counted."""
        fixations = pd.DataFrame([{
            "start_timestamp": 400.0, "end_timestamp": 700.0, "duration_ms": 300.0, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 20,
        }])
        result = dwell_time_first_epoch(fixations, scene_start_ts=0.0, epoch_ms=500.0)
        assert result["img_A"] == pytest.approx(100.0)

    def test_empty_fixations_returns_empty(self, fixations_df_empty):
        assert dwell_time_first_epoch(fixations_df_empty, scene_start_ts=0.0) == {}

    def test_nan_scene_start_returns_empty(self, fixations_df_simple):
        assert dwell_time_first_epoch(fixations_df_simple, scene_start_ts=np.nan) == {}

class TestScanpathLength:
    def test_zero_fixations_is_zero(self, fixations_df_empty):
        assert scanpath_length(fixations_df_empty) == 0.0

    def test_single_fixation_is_zero(self, fixations_df_single):
        assert scanpath_length(fixations_df_single) == 0.0

    def test_two_fixations_euclidean_distance(self):
        fixations = pd.DataFrame([
            {"start_timestamp": 0, "end_timestamp": 100, "duration_ms": 100, "rx": 0.0, "ry": 0.0, "image": "img_A", "n_samples": 7},
            {"start_timestamp": 150, "end_timestamp": 250, "duration_ms": 100, "rx": 3.0, "ry": 4.0, "image": "img_B", "n_samples": 7},
        ])
        assert scanpath_length(fixations) == pytest.approx(5.0)

    def test_joint_mask_excludes_fixation_with_nan_coord(self):
        """
        Regression for the rx/ry-independent-dropna bug. Middle fixation
        has rx valid but ry NaN and must be excluded from the path entirely
        """
        fixations = pd.DataFrame([
            {"start_timestamp": 0, "end_timestamp": 100, "duration_ms": 100, "rx": 0.0, "ry": 0.0, "image": "img_A", "n_samples": 7},
            {"start_timestamp": 150, "end_timestamp": 250, "duration_ms": 100, "rx": 100.0, "ry": np.nan, "image": "img_A", "n_samples": 7},
            {"start_timestamp": 300, "end_timestamp": 400, "duration_ms": 100, "rx": 3.0, "ry": 4.0, "image": "img_B", "n_samples": 7},
        ])
        assert scanpath_length(fixations) == pytest.approx(5.0)


class TestSaccadeMetrics:
    def test_saccade_count_zero_for_empty(self, fixations_df_empty):
        assert saccade_count(fixations_df_empty) == 0

    def test_saccade_count_zero_for_single_fixation(self, fixations_df_single):
        assert saccade_count(fixations_df_single) == 0

    def test_saccade_count_is_n_minus_one(self, fixations_df_simple):
        assert saccade_count(fixations_df_simple) == 1

    def test_saccade_rate_zero_duration(self, fixations_df_simple):
        assert np.isnan(saccade_rate(fixations_df_simple, 0.0))

    def test_saccade_rate_correct(self, fixations_df_simple):
        assert saccade_rate(fixations_df_simple, 1000.0) == 1.0

    def test_mean_saccade_amplitude_is_nan_for_too_few_fixations(self, fixations_df_single):
        assert np.isnan(mean_saccade_amplitude(fixations_df_single))

    def test_mean_saccade_amplitude_computed(self):
        fixations = pd.DataFrame([
            {"start_timestamp": 0, "end_timestamp": 100, "duration_ms": 100, "rx": 0.0, "ry": 0.0, "image": "img_A", "n_samples": 7},
            {"start_timestamp": 150, "end_timestamp": 250, "duration_ms": 100, "rx": 3.0, "ry": 4.0, "image": "img_B", "n_samples": 7},
        ])
        assert mean_saccade_amplitude(fixations) == pytest.approx(5.0)

class TestBlinkMetrics:
    def _empty_blinks(self):
        return pd.DataFrame(columns=["start_timestamp", "end_timestamp", "duration_ms", "n_samples"])

    def test_blink_count_empty(self):
        assert blink_count(self._empty_blinks()) == 0

    def test_blink_rate_zero_duration_is_nan(self):
        assert np.isnan(blink_rate(self._empty_blinks(), 0.0))

    def test_blink_rate_correct(self):
        blinks = pd.DataFrame([{"start_timestamp": 0}, {"start_timestamp": 30000}])
        assert blink_rate(blinks, 60000.0) == 2.0
