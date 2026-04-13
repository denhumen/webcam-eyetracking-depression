"""
Preprocessing utilities for eye-tracking sessions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

FEV_SACCADE = 0 # no fixation (saccade / free gaze)
FEV_FIX_START = 1 # fixation start
FEV_FIX_CONTINUE = 2 # fixation continuing
FEV_FIX_END = 3 # fixation end

def get_scene_data(df: pd.DataFrame, scene_index: int) -> pd.DataFrame:
    """
    Extract all rows for a given scene
    """
    return df[df["SCENE_INDEX"] == scene_index].copy()


def get_scene_indices(df: pd.DataFrame) -> List[int]:
    """
    Return sorted list of unique scene indices in a session
    """
    return sorted(df["SCENE_INDEX"].unique().astype(int))


def classify_scene_type(scene_index: int) -> str:
    """
    Odd = stimulus, even = fixation cross
    """
    return "stimulus" if scene_index % 2 != 0 else "fixation_cross"


def get_scene_images(scene_df: pd.DataFrame) -> List[str]:
    """
    Get list of actual image IDs shown in a scene
    """
    if "IMAGE" not in scene_df.columns:
        return []
    
    images = scene_df["IMAGE"].dropna().unique()
    return [
        img for img in images 
        if img != "no_image" and img != "" and str(img) != "nan"
    ]


def get_scene_duration_ms(scene_df: pd.DataFrame) -> float:
    """
    Calculate scene duration in milliseconds from timestamps
    """
    if len(scene_df) < 2:
        return 0.0
    return float(scene_df["TIMESTAMP"].max() - scene_df["TIMESTAMP"].min())


def extract_fixations(scene_df):
    """
    Extract fixations using vectorized operations on FEV column
    """
    if len(scene_df) == 0:
        return _empty_fixation_df()
    
    fev = scene_df["FEV"].values
    timestamps = scene_df["TIMESTAMP"].values
    fdur = scene_df["FDUR"].values
    rx = scene_df["RX"].values
    ry = scene_df["RY"].values
    fpogx = scene_df["FPOGX"].values
    fpogy = scene_df["FPOGY"].values
    images = scene_df["IMAGE"].values if "IMAGE" in scene_df.columns else [None] * len(scene_df)
    
    starts = np.where(fev == FEV_FIX_START)[0]
    ends = np.where(fev == FEV_FIX_END)[0]
    
    if len(starts) == 0 or len(ends) == 0:
        return _empty_fixation_df()
    
    fixations = []
    end_idx = 0
    
    for s in starts:
        while end_idx < len(ends) and ends[end_idx] < s:
            end_idx += 1
        if end_idx >= len(ends):
            break
        
        e = ends[end_idx]
        end_idx += 1
        
        fix_slice = slice(s, e + 1)
        fix_rx = rx[fix_slice]
        fix_ry = ry[fix_slice]
        fix_fpogx = fpogx[fix_slice]
        fix_fpogy = fpogy[fix_slice]
        fix_images = images[fix_slice]
        
        valid_rx = fix_rx[~np.isnan(fix_rx)] if np.issubdtype(fix_rx.dtype, np.floating) else fix_rx
        valid_ry = fix_ry[~np.isnan(fix_ry)] if np.issubdtype(fix_ry.dtype, np.floating) else fix_ry
        valid_fpogx = fix_fpogx[~np.isnan(fix_fpogx)] if np.issubdtype(fix_fpogx.dtype, np.floating) else fix_fpogx
        valid_fpogy = fix_fpogy[~np.isnan(fix_fpogy)] if np.issubdtype(fix_fpogy.dtype, np.floating) else fix_fpogy
        
        img_vals = [img for img in fix_images if img not in (None, "no_image", "", "nan") and not (isinstance(img, float) and np.isnan(img))]
        primary_image = max(set(img_vals), key=img_vals.count) if img_vals else None
        
        fixations.append({
            "start_timestamp": timestamps[s],
            "end_timestamp": timestamps[e],
            "duration_ms": fdur[e],
            "fpog_x": np.mean(valid_fpogx) if len(valid_fpogx) > 0 else np.nan,
            "fpog_y": np.mean(valid_fpogy) if len(valid_fpogy) > 0 else np.nan,
            "rx": np.mean(valid_rx) if len(valid_rx) > 0 else np.nan,
            "ry": np.mean(valid_ry) if len(valid_ry) > 0 else np.nan,
            "image": primary_image,
            "n_samples": e - s + 1,
        })
    
    if not fixations:
        return _empty_fixation_df()
    
    return pd.DataFrame(fixations)

def _empty_fixation_df() -> pd.DataFrame:
    """
    Return an empty DataFrame with the fixation schema
    """
    return pd.DataFrame(columns=[
        "start_timestamp", "end_timestamp", "duration_ms",
        "fpog_x", "fpog_y", "rx", "ry", "image", "n_samples"
    ])

def extract_blinks(scene_df):
    """
    Extract blink events
    """
    if len(scene_df) == 0 or "BLINK" not in scene_df.columns:
        return _empty_blinks_df()
    
    blink_vals = scene_df["BLINK"].astype(bool).values
    timestamps = scene_df["TIMESTAMP"].values
    
    if not blink_vals.any():
        return _empty_blinks_df()
    
    diff = np.diff(blink_vals.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if blink_vals[0]:
        starts = np.insert(starts, 0, 0)
    if blink_vals[-1]:
        ends = np.append(ends, len(blink_vals) - 1)
    
    if len(starts) == 0:
        return _empty_blinks_df()
    
    blinks = []
    for s, e in zip(starts, ends):
        blinks.append({
            "start_timestamp": timestamps[s],
            "end_timestamp": timestamps[e],
            "duration_ms": timestamps[e] - timestamps[s],
            "n_samples": e - s,
        })
    
    return pd.DataFrame(blinks) if blinks else _empty_blinks_df()

def _empty_blinks_df() -> pd.DataFrame:
    """
    Return an empty DataFrame with the blinks schema
    """
    return pd.DataFrame(columns=["start_timestamp", "end_timestamp", "duration_ms", "n_samples"])

def compute_session_quality(df: pd.DataFrame) -> dict:
    """
    Compute quality indicators for a session.

    A session is valid if it matches all of requirements:
    - at least 200 samples
    - duration at least 30 seconds
    - less than 50% missing gaze overall
    - at least 10 scenes recorded
    """
    n = len(df)
    if n == 0:
        return {"total_samples": 0, "is_valid": False}

    duration = df["TIMESTAMP"].max() - df["TIMESTAMP"].min()
    blink_ratio = df["BLINK"].sum() / n if "BLINK" in df.columns else 0.0
    missing_ratio = df["RX"].isna().sum() / n
    n_scenes = df["SCENE_INDEX"].nunique()

    intervals = df["TIMESTAMP"].diff().dropna()
    mean_interval = intervals.mean() if len(intervals) > 0 else 0.0

    is_valid = (
        n >= 200
        and duration >= 30000
        and missing_ratio < 0.5
        and n_scenes >= 10
    )

    return {
        "total_samples": n,
        "total_duration_ms": float(duration),
        "blink_ratio": float(blink_ratio),
        "missing_gaze_ratio": float(missing_ratio),
        "n_scenes": n_scenes,
        "mean_sampling_interval_ms": float(mean_interval),
        "estimated_fps": 1000.0 / mean_interval if mean_interval > 0 else 0.0,
        "is_valid": is_valid,
    }

def compute_scene_quality(scene_df: pd.DataFrame) -> dict:
    """
    Compute quality indicators for a scene.

    A scene is valid if it matches all of requirements:
    - at least 20 samples
    - duration at least 1000ms
    - less than 30% missing gaze data
    - less than 40% blink samples
    - at least 1 detected fixation
    """
    n = len(scene_df)
    if n == 0:
        return {"n_samples": 0, "is_valid": False, "duration_ms": 0,
                "blink_ratio": 0.0, "missing_gaze_ratio": 0.0, "n_fixations": 0}

    duration = get_scene_duration_ms(scene_df)
    blink_ratio = scene_df["BLINK"].sum() / n if "BLINK" in scene_df.columns else 0.0
    missing_ratio = scene_df["RX"].isna().sum() / n

    fev = pd.to_numeric(scene_df["FEV"], errors="coerce") if "FEV" in scene_df.columns else pd.Series(dtype=float)
    n_fixations = int((fev == FEV_FIX_START).sum())

    is_valid = (
        n >= 20
        and duration >= 1000
        and missing_ratio < 0.3
        and blink_ratio < 0.4
        and n_fixations >= 1
    )

    return {
        "n_samples": n,
        "duration_ms": duration,
        "blink_ratio": float(blink_ratio),
        "missing_gaze_ratio": float(missing_ratio),
        "n_fixations": n_fixations,
        "is_valid": is_valid,
    }
