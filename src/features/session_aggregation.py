"""
Session-level aggregation of scene-level eye-tracking metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List

from pyspark.sql import functions as F

PAIRS = [
    ("negative_vs_positive", "neg_vs_pos", ["negative", "positive"]),
    ("negative_vs_neutral",  "neg_vs_neu", ["negative", "neutral"]),
    ("neutral_vs_positive",  "pos_vs_neu", ["positive", "neutral"]),
]

V_SHORT = {"negative": "neg", "positive": "pos", "neutral": "neu"}

@dataclass
class Aggregation:
    """
    Spark agg expressions + the output column names they produce, grouped by category
    """
    exprs: list
    columns: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def all_columns(self) -> List[str]:
        out = []
        for cols in self.columns.values():
            out.extend(cols)
        return out


def _pair_mean(scene_col, pair_name):
    """
    Mean of scene_col, restricted to scenes of the given pair type
    """
    return F.mean(F.when(F.col("scene_valence_pair") == pair_name, F.col(scene_col)))

STATIC_SIMPLE_METRICS = [
    ("fixation_count",             "avg_fixation_count"),
    ("mean_fixation_duration_ms",  "avg_fixation_duration_ms"),
    ("total_fixation_duration_ms", "avg_total_fixation_duration_ms"),
    ("fixation_rate_per_sec",      "avg_fixation_rate"),
    ("scanpath_length",            "avg_scanpath_length"),
    ("saccade_count",              "avg_saccade_count"),
    ("saccade_rate_per_sec",       "avg_saccade_rate"),
    ("mean_saccade_amplitude",     "avg_saccade_amplitude"),
    ("blink_count",                "avg_blink_count"),
    ("blink_rate_per_min",         "avg_blink_rate"),
    ("gaze_transition_entropy",    "avg_gaze_entropy"),
    ("first_fixation_duration_ms", "avg_first_fixation_duration_ms"),
]

STATIC_PER_VALENCE_METRICS = [
    ("avg_dwell",       "dwell_time_ms_{v}"),
    ("avg_fix_prop",    "fixation_proportion_{v}"),
    ("avg_revisit",     "revisit_count_{v}"),
    ("avg_dwell_500ms", "dwell_time_500ms_{v}"),
]

def build_static_aggregation() -> Aggregation:
    exprs = []
    simple_cols, bias_cols, pair_split_cols, ttff_cols, first_fix_cols = [], [], [], [], []

    # Simple averages.
    for scene_col, alias in STATIC_SIMPLE_METRICS:
        exprs.append(F.mean(scene_col).alias(alias))
        simple_cols.append(alias)

    # Bias scores
    for pair_name, pair_suffix, _ in PAIRS:
        scene_col = f"bias_{pair_suffix}"
        alias = f"avg_bias_{pair_suffix}"
        exprs.append(_pair_mean(scene_col, pair_name).alias(alias))
        bias_cols.append(alias)

    # Per-valence features
    for out_prefix, template in STATIC_PER_VALENCE_METRICS:
        for pair_name, pair_suffix, valences in PAIRS:
            for v in valences:
                alias = f"{out_prefix}_{V_SHORT[v]}__{pair_suffix}"
                exprs.append(_pair_mean(template.format(v=v), pair_name).alias(alias))
                pair_split_cols.append(alias)

    # ttff
    for pair_name, pair_suffix, valences in PAIRS:
        for v in valences:
            scene_col = f"ttff_ms_{v}"
            short = V_SHORT[v]
            mean_alias = f"avg_ttff_{short}__{pair_suffix}"
            miss_alias = f"ttff_miss_rate_{short}__{pair_suffix}"
            exprs.append(_pair_mean(scene_col, pair_name).alias(mean_alias))
            exprs.append(
                F.mean(F.when(F.col("scene_valence_pair") == pair_name, F.col(scene_col).isNull().cast("double"))).alias(miss_alias)
            )
            ttff_cols.extend([mean_alias, miss_alias])

    # 5. First fixation probability
    for pair_name, pair_suffix, valences in PAIRS:
        for v in valences:
            alias = f"first_fix_prob_{V_SHORT[v]}__{pair_suffix}"
            exprs.append(
                F.mean(F.when(F.col("scene_valence_pair") == pair_name, (F.col("first_fixation_valence") == v).cast("double"))).alias(alias)
            )
            first_fix_cols.append(alias)

    return Aggregation(
        exprs=exprs,
        columns={
            "simple":     simple_cols,
            "bias":       bias_cols,
            "pair_split": pair_split_cols,
            "ttff":       ttff_cols,
            "first_fix":  first_fix_cols,
        },
    )

def build_temporal_aggregation() -> Aggregation:
    raise NotImplementedError("Temporal aggregation not yet migrated.")

def build_distributional_aggregation() -> Aggregation:
    raise NotImplementedError("Distributional aggregation not yet migrated.")
