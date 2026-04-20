"""
Computation of static attentional indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from config import NEGATIVE_KEYWORDS, POSITIVE_KEYWORDS

def derive_valence(category: str, labels: list) -> str:
    """
    Define the valence of the image by category and labels
    Returns: 'negative', 'positive', 'neutral', 'food', 'other'
    """
    labels_lower = [l.lower() for l in labels]
    
    if category.strip() == "neutral":
        return "neutral"
    
    if any(kw in labels_lower for kw in NEGATIVE_KEYWORDS):
        return "negative"
    if any(kw in labels_lower for kw in POSITIVE_KEYWORDS):
        return "positive"
    
    return "other"

def fixation_count(fixations: pd.DataFrame) -> int:
    """
    Total number of fixations in the scene
    """
    return len(fixations)

def mean_fixation_duration(fixations: pd.DataFrame) -> float:
    """
    Average fixation duration in ms
    """
    durations = fixations["duration_ms"].dropna()
    return float(durations.mean()) if len(durations) > 0 else np.nan


def total_fixation_duration(fixations: pd.DataFrame) -> float:
    """
    Sum of all fixation durations in ms
    """
    durations = fixations["duration_ms"].dropna()
    return float(durations.sum()) if len(durations) > 0 else 0.0


def fixation_rate(fixations: pd.DataFrame, scene_duration_ms: float) -> float:
    """
    Fixations per second
    """
    if scene_duration_ms <= 0:
        return np.nan
    return len(fixations) / (scene_duration_ms / 1000.0)

def dwell_time_per_image(fixations: pd.DataFrame) -> Dict[str, float]:
    """
    Total fixation duration by image AOI (Area of interest)
    """
    if len(fixations) == 0:
        return {}
    
    on_image = fixations[fixations["image"].notna() & (fixations["image"] != "")]
    if len(on_image) == 0:
        return {}
    
    return on_image.groupby("image")["duration_ms"].sum().to_dict()


def fixation_count_per_image(fixations: pd.DataFrame) -> Dict[str, int]:
    """
    Number of fixations by image AOI (Area of interest)
    """
    if len(fixations) == 0:
        return {}
    
    on_image = fixations[fixations["image"].notna() & (fixations["image"] != "")]
    if len(on_image) == 0:
        return {}
    
    return on_image.groupby("image").size().to_dict()


def fixation_proportion_per_image(fixations: pd.DataFrame) -> Dict[str, float]:
    """
    Proportion of total fixation time spent by image AOI (Area of interest)
    """
    dwell = dwell_time_per_image(fixations)
    total = sum(dwell.values())
    if total <= 0:
        return {}
    return {img: t / total for img, t in dwell.items()}

def _pair_bias(
    fixations: pd.DataFrame,
    stimulus_config: Optional[Dict],
    images: Optional[List[str]],
    valence_a: str,
    valence_b: str,
) -> float:
    """
    Proportional dwell difference between two valences: (a - b) / (a + b).
    
    Returns NaN unless at least one image of each valence is present in the scene.
    Positive value → more dwell on valence_a; negative → more dwell on valence_b.
    """
    if not stimulus_config or not images:
        return np.nan
    
    valence_dwell: Dict[str, float] = {}
    dwell = dwell_time_per_image(fixations)
    
    for img_id in images:
        if img_id in stimulus_config:
            v = derive_valence(
                stimulus_config[img_id].get("category", ""),
                stimulus_config[img_id].get("labels", []),
            )
            valence_dwell[v] = valence_dwell.get(v, 0.0) + dwell.get(img_id, 0.0)
    
    a = valence_dwell.get(valence_a, np.nan)
    b = valence_dwell.get(valence_b, np.nan)
    
    if np.isnan(a) or np.isnan(b):
        return np.nan
    
    total = a + b
    if total <= 0:
        return np.nan
    
    return (a - b) / total

def first_fixation_image(fixations: pd.DataFrame) -> Optional[str]:
    """
    Which image received the first fixation in the scene
    Return: image_id or None
    """
    on_image = fixations[fixations["image"].notna() & (fixations["image"] != "")]
    if len(on_image) == 0:
        return None
    return on_image.iloc[0]["image"]

def first_fixation_duration(fixations: pd.DataFrame) -> float:
    """
    Duration of the first fixation that lands on any image
    """
    on_image = fixations[fixations["image"].notna() & (fixations["image"] != "")]
    if len(on_image) == 0:
        return np.nan
    return float(on_image.iloc[0]["duration_ms"])

def time_to_first_fixation_on_image(
    fixations: pd.DataFrame, 
    scene_start_ts: float,
    image_id: str,
) -> float:
    """
    Time from scene onset to first fixation on a specific image.
    """
    on_target = fixations[fixations["image"] == image_id]
    if len(on_target) == 0:
        return np.nan
    return float(on_target.iloc[0]["start_timestamp"] - scene_start_ts)

def revisit_count_per_image(fixations):
    """
    How many times gaze returned by image AOI (Area of interest)
    """
    if len(fixations) == 0:
        return {}
    
    images = fixations["image"].values
    visits = {}
    last_image = None
    
    for img in images:
        if img is None or img == "" or (isinstance(img, float) and np.isnan(img)):
            last_image = None
            continue
        if img != last_image:
            visits[img] = visits.get(img, 0) + 1
            last_image = img
    
    return {img: max(0, count - 1) for img, count in visits.items()}


def dwell_time_first_epoch(
    fixations: pd.DataFrame,
    scene_start_ts: float,
    epoch_ms: float = 500.0,
) -> Dict[str, float]:
    """
    Dwell time by image AOI within the first epoch after scene onset.
    Each fixation contributes only the portion of its duration that falls
    inside the [scene_start_ts, scene_start_ts + epoch_ms] window.
    """
    if len(fixations) == 0 or np.isnan(scene_start_ts):
        return {}
    
    epoch_end = scene_start_ts + epoch_ms
    
    overlapping = fixations[
        (fixations["end_timestamp"] >= scene_start_ts) &
        (fixations["start_timestamp"] <= epoch_end)
    ].copy()
    
    if len(overlapping) == 0:
        return {}
    
    on_image = overlapping[overlapping["image"].notna() & (overlapping["image"] != "")].copy()
    if len(on_image) == 0:
        return {}
    
    clip_start = np.maximum(on_image["start_timestamp"].values, scene_start_ts)
    clip_end = np.minimum(on_image["end_timestamp"].values, epoch_end)
    on_image["early_overlap_ms"] = np.maximum(clip_end - clip_start, 0.0)
    
    return on_image.groupby("image")["early_overlap_ms"].sum().to_dict()

def scanpath_length(fixations: pd.DataFrame) -> float:
    """
    Total Euclidean distance of the scanpath
    """
    if len(fixations) < 2:
        return 0.0
    
    mask = fixations["rx"].notna() & fixations["ry"].notna()
    rx = fixations.loc[mask, "rx"].values
    ry = fixations.loc[mask, "ry"].values
    
    if len(rx) < 2:
        return 0.0
    
    dx = np.diff(rx)
    dy = np.diff(ry)
    distances = np.sqrt(dx**2 + dy**2)
    
    return float(np.sum(distances))


def saccade_count(fixations: pd.DataFrame) -> int:
    """
    Number of saccades = number of transitions between fixations.
    """
    n = len(fixations)
    return max(0, n - 1)


def saccade_rate(fixations: pd.DataFrame, scene_duration_ms: float) -> float:
    """
    Saccades per second
    """
    if scene_duration_ms <= 0:
        return np.nan
    n_saccades = saccade_count(fixations)
    return n_saccades / (scene_duration_ms / 1000.0)


def mean_saccade_amplitude(fixations: pd.DataFrame) -> float:
    """
    Average saccade amplitude (Euclidean distance between consecutive fixations)
    """
    if len(fixations) < 2:
        return np.nan
    
    mask = fixations["rx"].notna() & fixations["ry"].notna()
    rx = fixations.loc[mask, "rx"].values
    ry = fixations.loc[mask, "ry"].values
    
    if len(rx) < 2:
        return np.nan
    
    dx = np.diff(rx)
    dy = np.diff(ry)
    amplitudes = np.sqrt(dx**2 + dy**2)
    
    return float(np.mean(amplitudes))

def blink_count(blinks: pd.DataFrame) -> int:
    """
    Total number of blinks
    """
    return len(blinks)

def blink_rate(blinks: pd.DataFrame, scene_duration_ms: float) -> float:
    """
    Blinks per minute
    """
    if scene_duration_ms <= 0:
        return np.nan
    return len(blinks) / (scene_duration_ms / 60000.0)


def gaze_transition_matrix(fixations, aoi_list):
    """
    Computes the gaze transition count matrix between AOIs.
    """
    n = len(aoi_list)
    aoi_to_idx = {aoi: i for i, aoi in enumerate(aoi_list)}
    other_idx = aoi_to_idx.get("other", n - 1)
    
    matrix = np.zeros((n, n), dtype=int)
    
    images = fixations["image"].values
    aoi_sequence = [aoi_to_idx.get(img, other_idx) for img in images]
    
    for i in range(len(aoi_sequence) - 1):
        matrix[aoi_sequence[i], aoi_sequence[i + 1]] += 1
    
    return matrix

def transition_matrix_density(matrix: np.ndarray) -> float:
    """
    Fraction of non-zero cells in the transition matrix
    """
    total_cells = matrix.size
    if total_cells == 0:
        return np.nan
    nonzero = np.count_nonzero(matrix)
    return float(nonzero / total_cells)


def gaze_transition_entropy(matrix: np.ndarray) -> float:
    """
    Shannon entropy of the transition probability matrix.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(prob_matrix > 0, np.log2(prob_matrix), 0.0)
    
    entropy = -np.sum(prob_matrix * log_probs)
    return float(entropy)