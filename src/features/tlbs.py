"""
Trial-Level Bias Score parameter extraction
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

TLBS_PARAMS = [
    "mean_pos", # average of strictly positive bias values
    "mean_neg", # average of strictly negative bias values
    "peak_pos", # max bias value
    "peak_neg", # min bias value
    "mean_abs", # mean absolute bias value
    "variability", # mean absolute first difference across trials
    "slope", # linear-regression slope over trial number
]

PAIRS = [
    ("negative_vs_positive", "neg_vs_pos", "bias_neg_vs_pos"),
    ("negative_vs_neutral",  "neg_vs_neu", "bias_neg_vs_neu"),
    ("neutral_vs_positive",  "pos_vs_neu", "bias_pos_vs_neu"),
]

_NAN_PARAMS = {p: np.nan for p in TLBS_PARAMS}

def compute_tlbs_params(values: np.ndarray, trial_nums: np.ndarray) -> Dict[str, float]:
    """
    Compute the seven TL-BS parameters from a single trial stream
    """
    n = len(values)
    if n < 3 or len(trial_nums) != n:
        return dict(_NAN_PARAMS)

    pos_vals = values[values > 0]
    neg_vals = values[values < 0]

    slope, _, _, _, _ = scipy_stats.linregress(trial_nums, values)

    return {
        "mean_pos": float(np.mean(pos_vals)) if len(pos_vals) > 0 else 0.0,
        "mean_neg": float(np.mean(neg_vals)) if len(neg_vals) > 0 else 0.0,
        "peak_pos": float(np.max(values)),
        "peak_neg": float(np.min(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "variability": float(np.mean(np.abs(np.diff(values)))),
        "slope": float(slope),
    }


def compute_tlbs_per_pair(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-pair TL-BS parameters for each session
    """
    result_rows = []

    for session_id, session_df in df_trials.groupby("session_id"):
        row = {"session_id": session_id}

        for pair_name, pair_suffix, bias_col in PAIRS:
            sub = session_df[session_df["scene_valence_pair"] == pair_name].sort_values("scene_index")
            valid = sub[sub[bias_col].notna()]
            values = valid[bias_col].values
            trial_nums = valid["scene_index"].values

            params = compute_tlbs_params(values, trial_nums)
            for p in TLBS_PARAMS:
                row[f"tlbs_{p}__{pair_suffix}"] = params[p]

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def tlbs_feature_names(pair_suffixes=None):
    """
    Return the list of TL-BS feature column names for the given pair suffixes
    """
    if pair_suffixes is None:
        pair_suffixes = [suffix for _, suffix, _ in PAIRS]
    return [f"tlbs_{p}__{suffix}" for suffix in pair_suffixes for p in TLBS_PARAMS]
