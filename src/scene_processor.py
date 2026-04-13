"""
Scene processing utilities
"""

from . import preprocessing as pp
from . import scene_metrics as sm
import pandas as pd
from typing import Dict, List, Optional
import numpy as np

def compute_scene_metrics(
    scene_df: pd.DataFrame,
    scene_index: int,
    stimulus_config: Optional[Dict] = None,
) -> dict:
    """
    Compute all static attention metrics for a single scene.
    
    Args:
        scene_df (DataFrame): Raw gaze data for this scene
        scene_index (int): The scene index in the session
        stimulus_config (dict, optional): The stimulus JSON config
    
    Returns
        dict: all computed metrics and scene metadata.
    """
    scene_type = pp.classify_scene_type(scene_index)
    images = pp.get_scene_images(scene_df)
    duration_ms = pp.get_scene_duration_ms(scene_df)
    quality = pp.compute_scene_quality(scene_df)
    fixations = pp.extract_fixations(scene_df)
    blinks = pp.extract_blinks(scene_df)
    scene_start = scene_df["TIMESTAMP"].iloc[0] if len(scene_df) > 0 else np.nan
    aoi_list = images + ["other"] if images else ["other"]
    
    metrics = {
        "scene_index": scene_index,
        "scene_type": scene_type,
        "image_ids": images,
        "n_images": len(images),
        "duration_ms": duration_ms,
        "n_samples": quality["n_samples"],
        "scene_quality_valid": quality["is_valid"],
        "blink_ratio": quality.get("blink_ratio", np.nan),
        "missing_gaze_ratio": quality.get("missing_gaze_ratio", np.nan),
        
        "fixation_count": sm.fixation_count(fixations),
        "mean_fixation_duration_ms": sm.mean_fixation_duration(fixations),
        "total_fixation_duration_ms": sm.total_fixation_duration(fixations),
        "fixation_rate_per_sec": sm.fixation_rate(fixations, duration_ms),
        "fixation_bias": np.nan,

        "first_fixation_image": sm.first_fixation_image(fixations),
        "first_fixation_duration_ms": sm.first_fixation_duration(fixations),
        "second_fixation_image": sm.second_fixation_image(fixations),
        "second_fixation_duration_ms": sm.second_fixation_duration(fixations),
        
        "scanpath_length": sm.scanpath_length(fixations),
        "saccade_count": sm.saccade_count(fixations),
        "saccade_rate_per_sec": sm.saccade_rate(fixations, duration_ms),
        "mean_saccade_amplitude": sm.mean_saccade_amplitude(fixations),
        
        "blink_count": sm.blink_count(blinks),
        "blink_rate_per_min": sm.blink_rate(blinks, duration_ms),
        
        "transition_matrix_density": np.nan,
        "gaze_transition_entropy": np.nan,

        "dwell_time_500ms_negative": np.nan,
        "dwell_time_500ms_positive": np.nan,
        "dwell_time_500ms_neutral": np.nan,
    }
    
    if images:
        dwell = sm.dwell_time_per_image(fixations)
        fix_counts = sm.fixation_count_per_image(fixations)
        fix_props = sm.fixation_proportion_per_image(fixations)
        revisits = sm.revisit_count_per_image(fixations)
        
        image_valences = {}
        for img_id in images:
            if stimulus_config and img_id in stimulus_config:
                image_valences[img_id] = sm.derive_valence(
                    stimulus_config[img_id].get("category", ""),
                    stimulus_config[img_id].get("labels", []),
                )
            else:
                image_valences[img_id] = "unknown"

        early_dwell = sm.dwell_time_first_epoch(fixations, scene_start, epoch_ms=500.0)
        early_valence_dwell = {}
        for img_id, dt in early_dwell.items():
            v = image_valences.get(img_id, "unknown")
            early_valence_dwell[v] = early_valence_dwell.get(v, 0.0) + dt
        
        for v in ["negative", "positive", "neutral"]:
            metrics[f"dwell_time_500ms_{v}"] = early_valence_dwell.get(v, np.nan)
        
        valence_dwell: Dict[str, float] = {}
        valence_fix_count: Dict[str, int] = {}
        valence_fix_prop: Dict[str, float] = {}
        valence_revisit: Dict[str, int] = {}
        valence_ttff: Dict[str, float] = {}
        
        for img_id in images:
            v = image_valences[img_id]

            valence_dwell[v] = valence_dwell.get(v, 0.0) + dwell.get(img_id, 0.0)
            valence_fix_count[v] = valence_fix_count.get(v, 0) + fix_counts.get(img_id, 0)
            valence_fix_prop[v] = valence_fix_prop.get(v, 0.0) + fix_props.get(img_id, 0.0)
            valence_revisit[v] = valence_revisit.get(v, 0) + revisits.get(img_id, 0)
            
            ttff_val = sm.time_to_first_fixation_on_image(fixations, scene_start, img_id)
            if v not in valence_ttff or (not np.isnan(ttff_val) and ttff_val < valence_ttff.get(v, np.inf)):
                valence_ttff[v] = ttff_val
        
        for v in ["negative", "positive", "neutral"]:
            metrics[f"dwell_time_ms_{v}"] = valence_dwell.get(v, np.nan)
            metrics[f"fixation_count_{v}"] = valence_fix_count.get(v, np.nan)
            metrics[f"fixation_proportion_{v}"] = valence_fix_prop.get(v, np.nan)
            metrics[f"revisit_count_{v}"] = valence_revisit.get(v, np.nan)
            metrics[f"ttff_ms_{v}"] = valence_ttff.get(v, np.nan)
        
        ff_img = sm.first_fixation_image(fixations)
        metrics["first_fixation_valence"] = image_valences.get(ff_img, None) if ff_img else None

        sf_img = sm.second_fixation_image(fixations)
        metrics["second_fixation_valence"] = image_valences.get(sf_img, None) if sf_img else None
        
        scene_valences = sorted(set(image_valences.values()))
        metrics["scene_valence_pair"] = "_vs_".join(scene_valences)
        metrics["fixation_bias"] = sm.fixation_bias(fixations, stimulus_config, images)
        
        for i, img_id in enumerate(images):
            suffix = f"_img{i}"
            metrics[f"image_id{suffix}"] = img_id
            metrics[f"valence{suffix}"] = image_valences.get(img_id, "unknown")
            metrics[f"dwell_time_ms{suffix}"] = dwell.get(img_id, 0.0)
            metrics[f"fixation_count{suffix}"] = fix_counts.get(img_id, 0)
            metrics[f"fixation_proportion{suffix}"] = fix_props.get(img_id, 0.0)
            metrics[f"revisit_count{suffix}"] = revisits.get(img_id, 0)
            metrics[f"ttff_ms{suffix}"] = sm.time_to_first_fixation_on_image(
                fixations, scene_start, img_id
            )
        
        trans_matrix = sm.gaze_transition_matrix(fixations, aoi_list)
        metrics["transition_matrix_density"] = sm.transition_matrix_density(trans_matrix)
        metrics["gaze_transition_entropy"] = sm.gaze_transition_entropy(trans_matrix)
    
    return metrics