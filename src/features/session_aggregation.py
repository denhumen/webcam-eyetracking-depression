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
    ("fixation_count", "avg_fixation_count"),
    ("mean_fixation_duration_ms", "avg_fixation_duration_ms"),
    ("total_fixation_duration_ms", "avg_total_fixation_duration_ms"),
    ("fixation_rate_per_sec", "avg_fixation_rate"),
    ("scanpath_length", "avg_scanpath_length"),
    ("saccade_count", "avg_saccade_count"),
    ("saccade_rate_per_sec", "avg_saccade_rate"),
    ("mean_saccade_amplitude", "avg_saccade_amplitude"),
    ("blink_count", "avg_blink_count"),
    ("blink_rate_per_min", "avg_blink_rate"),
    ("gaze_transition_entropy", "avg_gaze_entropy"),
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
            exprs.append(F.mean(F.when(F.col("scene_valence_pair") == pair_name, F.col(scene_col).isNull().cast("double"))).alias(miss_alias))
            ttff_cols.extend([mean_alias, miss_alias])

    # 5. First fixation probability
    for pair_name, pair_suffix, valences in PAIRS:
        for v in valences:
            alias = f"first_fix_prob_{V_SHORT[v]}__{pair_suffix}"
            exprs.append(F.mean(F.when(F.col("scene_valence_pair") == pair_name, (F.col("first_fixation_valence") == v).cast("double"))).alias(alias))
            first_fix_cols.append(alias)

    return Aggregation(
        exprs=exprs,
        columns={
            "simple": simple_cols,
            "bias": bias_cols,
            "pair_split": pair_split_cols,
            "ttff": ttff_cols,
            "first_fix": first_fix_cols,
        },
    )

DIST_SIMPLE_METRICS = [
    "fixation_count",
    "mean_fixation_duration_ms",
    "total_fixation_duration_ms",
    "fixation_rate_per_sec",
    "scanpath_length",
    "saccade_count",
    "saccade_rate_per_sec",
    "mean_saccade_amplitude",
    "blink_count",
    "blink_rate_per_min",
    "gaze_transition_entropy",
    "transition_matrix_density",
    "first_fixation_duration_ms",
]

DIST_PER_VALENCE_TEMPLATES = [
    ("dwell", "dwell_time_ms_{v}"),
    ("fix_prop", "fixation_proportion_{v}"),
    ("revisit", "revisit_count_{v}"),
    ("dwell_500ms", "dwell_time_500ms_{v}"),
    ("ttff", "ttff_ms_{v}"),
]

def _quartile_exprs(scene_col, alias_prefix, filter_expr=None):
    """
    Return (q25, q50, q75, IQR=q75-q25) aggregation expressions.
    """
    col = F.col(scene_col) if filter_expr is None else F.when(filter_expr, F.col(scene_col))
    q25 = F.percentile_approx(col, 0.25).alias(f"{alias_prefix}_q25")
    q50 = F.percentile_approx(col, 0.50).alias(f"{alias_prefix}_q50")
    q75 = F.percentile_approx(col, 0.75).alias(f"{alias_prefix}_q75")
    return [q25, q50, q75]

def build_distributional_aggregation() -> Aggregation:
    exprs = []
    simple_cols, bias_cols, pair_split_cols, first_fix_cols = [], [], [], []

    # Simple metrics
    for scene_col in DIST_SIMPLE_METRICS:
        alias_prefix = scene_col
        exprs.extend(_quartile_exprs(scene_col, alias_prefix))
        for suffix in ("q25", "q50", "q75", "iqr"):
            simple_cols.append(f"{alias_prefix}_{suffix}")

    # Bias scores
    for pair_name, pair_suffix, _ in PAIRS:
        scene_col = f"bias_{pair_suffix}"
        alias_prefix = f"bias_{pair_suffix}"
        filter_expr = F.col("scene_valence_pair") == pair_name
        exprs.extend(_quartile_exprs(scene_col, alias_prefix, filter_expr))
        for suffix in ("q25", "q50", "q75", "iqr"):
            bias_cols.append(f"{alias_prefix}_{suffix}")

    # Valence metrics
    for out_prefix, template in DIST_PER_VALENCE_TEMPLATES:
        for pair_name, pair_suffix, valences in PAIRS:
            filter_expr = F.col("scene_valence_pair") == pair_name
            for v in valences:
                scene_col = template.format(v=v)
                alias_prefix = f"{out_prefix}_{V_SHORT[v]}__{pair_suffix}"
                exprs.extend(_quartile_exprs(scene_col, alias_prefix, filter_expr))
                for suffix in ("q25", "q50", "q75", "iqr"):
                    pair_split_cols.append(f"{alias_prefix}_{suffix}")

    # First fixation probability
    for pair_name, pair_suffix, valences in PAIRS:
        for v in valences:
            alias = f"first_fix_prob_{V_SHORT[v]}__{pair_suffix}"
            exprs.append(F.mean(F.when(F.col("scene_valence_pair") == pair_name, (F.col("first_fixation_valence") == v).cast("double"))).alias(alias))
            first_fix_cols.append(alias)

    return Aggregation(
        exprs=exprs,
        columns={
            "simple": simple_cols,
            "bias": bias_cols,
            "pair_split": pair_split_cols,
            "first_fix": first_fix_cols,
        },
    )

def add_iqr_columns(df, aggregation: Aggregation):
    """
    Add IQR (q75 - q25) columns to a Pandas DataFrame produced by applying a distributional aggregation.
    """
    seen_prefixes = set()
    for col_name in aggregation.all_columns:
        if col_name.endswith("_iqr"):
            prefix = col_name[:-4]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            df[f"{prefix}_iqr"] = df[f"{prefix}_q75"] - df[f"{prefix}_q25"]
    return df

TEMPORAL_SIMPLE_METRICS = DIST_SIMPLE_METRICS
TEMPORAL_PER_VALENCE_TEMPLATES = DIST_PER_VALENCE_TEMPLATES

def _half_mean(scene_col, is_early_col, half, filter_expr=None):
    """
    Mean of scene_col restricted to the given session half.
    """
    target = F.col(is_early_col) if half == "early" else ~F.col(is_early_col)
    if filter_expr is not None:
        target = target & filter_expr
    return F.mean(F.when(target, F.col(scene_col)))

def build_temporal_aggregation() -> Aggregation:
    """
    Build aggregation expressions for early/late session halves.
    """
    exprs = []
    simple_cols, bias_cols, pair_split_cols, first_fix_cols = [], [], [], []

    # Simple metrics
    for scene_col in TEMPORAL_SIMPLE_METRICS:
        for half in ("early", "late"):
            alias = f"{scene_col}_{half}"
            exprs.append(_half_mean(scene_col, "is_early", half).alias(alias))
            simple_cols.append(alias)
        simple_cols.append(f"{scene_col}_delta")

    # Bias scores
    for pair_name, pair_suffix, _ in PAIRS:
        scene_col = f"bias_{pair_suffix}"
        filter_expr = F.col("scene_valence_pair") == pair_name
        for half in ("early", "late"):
            alias = f"bias_{pair_suffix}_{half}"
            exprs.append(_half_mean(scene_col, "is_early", half, filter_expr).alias(alias))
            bias_cols.append(alias)
        bias_cols.append(f"bias_{pair_suffix}_delta")

    # Valence metrics
    for out_prefix, template in TEMPORAL_PER_VALENCE_TEMPLATES:
        for pair_name, pair_suffix, valences in PAIRS:
            filter_expr = F.col("scene_valence_pair") == pair_name
            for v in valences:
                scene_col = template.format(v=v)
                alias_prefix = f"{out_prefix}_{V_SHORT[v]}__{pair_suffix}"
                for half in ("early", "late"):
                    alias = f"{alias_prefix}_{half}"
                    exprs.append(_half_mean(scene_col, "is_early", half, filter_expr).alias(alias))
                    pair_split_cols.append(alias)
                pair_split_cols.append(f"{alias_prefix}_delta")

    # First fixation probability
    for pair_name, pair_suffix, valences in PAIRS:
        filter_expr = F.col("scene_valence_pair") == pair_name
        for v in valences:
            alias_prefix = f"first_fix_prob_{V_SHORT[v]}__{pair_suffix}"
            for half in ("early", "late"):
                target = F.col("is_early") if half == "early" else ~F.col("is_early")
                alias = f"{alias_prefix}_{half}"
                exprs.append(F.mean(F.when(target & filter_expr, (F.col("first_fixation_valence") == v).cast("double"))).alias(alias))
                first_fix_cols.append(alias)
            first_fix_cols.append(f"{alias_prefix}_delta")

    return Aggregation(
        exprs=exprs,
        columns={
            "simple": simple_cols,
            "bias": bias_cols,
            "pair_split": pair_split_cols,
            "first_fix": first_fix_cols,
        },
    )

def add_delta_columns(df, aggregation: Aggregation):
    """
    Add delta (late - early) columns to a Pandas DataFrame produced by a
    temporal aggregation. Call once after `.toPandas()`.
    """
    seen_prefixes = set()
    for col_name in aggregation.all_columns:
        if col_name.endswith("_delta"):
            prefix = col_name[:-6]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            df[f"{prefix}_delta"] = df[f"{prefix}_late"] - df[f"{prefix}_early"]
    return df
