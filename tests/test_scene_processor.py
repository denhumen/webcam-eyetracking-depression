"""
Integration tests for src/scene_processor.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.scene_processor import compute_scene_metrics

class TestSceneProcessorBasics:
    def test_returns_dict(self, two_image_scene, stimulus_config_basic):
        assert isinstance(compute_scene_metrics(two_image_scene, 1, stimulus_config_basic), dict)

    def test_contains_expected_core_keys(self, two_image_scene, stimulus_config_basic):
        result = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)
        expected = {"scene_index", "scene_type", "n_images", "duration_ms", "fixation_count", "mean_fixation_duration_ms", "bias_neg_vs_pos", "bias_neg_vs_neu", "bias_pos_vs_neu", "scanpath_length", "saccade_count", "blink_count"}
        assert expected.issubset(result.keys())

    def test_odd_scene_index_is_stimulus(self, two_image_scene, stimulus_config_basic):
        assert compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)["scene_type"] == "stimulus"

    def test_even_scene_index_is_fixation_cross(self, two_image_scene, stimulus_config_basic):
        assert compute_scene_metrics(two_image_scene, 2, stimulus_config_basic)["scene_type"] == "fixation_cross"

class TestSceneValencePair:
    def test_two_image_scene_produces_pair_label(self, two_image_scene, stimulus_config_basic):
        result = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)
        assert result.get("scene_valence_pair") == "negative_vs_positive"

    def test_pair_label_is_alphabetically_sorted(self, two_image_scene, stimulus_config_basic):
        pair = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic).get("scene_valence_pair", "")
        parts = pair.split("_vs_")
        assert parts == sorted(parts)

    def test_single_image_scene_produces_single_valence_label(self, single_fixation_scene, stimulus_config_basic):
        result = compute_scene_metrics(single_fixation_scene, 1, stimulus_config_basic)
        assert result.get("scene_valence_pair") == "negative"

class TestBiasMetricsByPairType:
    def test_neg_vs_pos_scene_has_neg_vs_pos_bias_defined(self, two_image_scene, stimulus_config_basic):
        result = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)
        assert not np.isnan(result["bias_neg_vs_pos"])

    def test_neg_vs_pos_scene_has_neu_biases_as_nan(self, two_image_scene, stimulus_config_basic):
        result = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)
        assert np.isnan(result["bias_neg_vs_neu"])
        assert np.isnan(result["bias_pos_vs_neu"])

    def test_single_valence_scene_has_all_biases_nan(self, single_fixation_scene, stimulus_config_basic):
        result = compute_scene_metrics(single_fixation_scene, 1, stimulus_config_basic)
        for key in ("bias_neg_vs_pos", "bias_neg_vs_neu", "bias_pos_vs_neu"):
            assert np.isnan(result[key])

class TestPerValenceKeys:
    def test_per_valence_metric_keys_are_present(self, two_image_scene, stimulus_config_basic):
        result = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)
        for v in ["negative", "positive", "neutral"]:
            for m in ["dwell_time_ms", "fixation_count", "fixation_proportion", "revisit_count", "ttff_ms", "dwell_time_500ms"]:
                assert f"{m}_{v}" in result

    def test_neutral_metrics_are_nan_when_no_neutral_image(self, two_image_scene, stimulus_config_basic):
        result = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)
        assert np.isnan(result["dwell_time_ms_neutral"])
        assert pd.isna(result["fixation_count_neutral"])

class TestBiasBounds:
    def test_all_bias_values_in_bounds(self, two_image_scene, stimulus_config_basic):
        result = compute_scene_metrics(two_image_scene, 1, stimulus_config_basic)
        for key in ("bias_neg_vs_pos", "bias_neg_vs_neu", "bias_pos_vs_neu"):
            v = result[key]
            if not np.isnan(v):
                assert -1.0 <= v <= 1.0, f"{key}={v} is out of [-1,1]"
