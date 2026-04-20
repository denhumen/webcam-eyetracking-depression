"""
Shared pytest fixtures for constructing synthetic gaze data
"""

import numpy as np
import pandas as pd
import pytest

RAW_SCHEMA = ["TIMESTAMP", "TEV", "FEV", "FDUR", "FPOGX", "FPOGY", "RX", "RY", "BLINK", "IMAGE", "SCENE_INDEX"]

def _row(ts, *, fev=0, fdur=0.0, rx=0.5, ry=0.5, image="no_image", blink=False, scene=1):
    """
    Build one gaze-sample row
    """
    if blink:
        rx = ry = np.nan
    return {
        "TIMESTAMP": float(ts), "TEV": 1, "FEV": fev, "FDUR": fdur,
        "FPOGX": rx, "FPOGY": ry, "RX": rx, "RY": ry,
        "BLINK": blink, "IMAGE": image, "SCENE_INDEX": scene,
    }

def _make_fixation_rows(start_ts, duration_ms, image_id, sample_interval_ms=16.0, rx=0.3, ry=0.5):
    """
    Rows for one fixation: start (FEV=1), continues (FEV=2), end (FEV=3)
    """
    n = max(2, int(round(duration_ms / sample_interval_ms)))
    rows = []
    for i in range(n):
        ts = start_ts + i * sample_interval_ms
        fev = 1 if i == 0 else (3 if i == n - 1 else 2)
        fdur = float(duration_ms) if fev == 3 else 0.0
        rows.append(_row(ts, fev=fev, fdur=fdur, rx=rx, ry=ry, image=image_id))
    return rows

def _make_gap_rows(start_ts, duration_ms, sample_interval_ms=16.0):
    n = max(1, int(round(duration_ms / sample_interval_ms)))
    return [
        _row(start_ts + i * sample_interval_ms, rx=np.nan, ry=np.nan)
        for i in range(n)
    ]

@pytest.fixture
def empty_scene():
    return pd.DataFrame(columns=RAW_SCHEMA)

@pytest.fixture
def no_fixation_scene():
    return pd.DataFrame([_row(i * 50) for i in range(30)])

@pytest.fixture
def single_fixation_scene():
    rows = _make_fixation_rows(start_ts=0, duration_ms=200, image_id="img_A", rx=0.3, ry=0.6)
    return pd.DataFrame(rows)

@pytest.fixture
def two_image_scene():
    """
    Fixation sequence: img_A (200ms) -> gap (100ms) -> img_B (200ms)
    """
    rows = (
        _make_fixation_rows(0, 200, "img_A", rx=0.3, ry=0.6) + _make_gap_rows(210, 100) + _make_fixation_rows(320, 200, "img_B", rx=0.7, ry=0.6)
    )
    return pd.DataFrame(rows)

@pytest.fixture
def revisit_scene():
    """
    Sequence A -> B -> A for revisit counting
    """
    rows = (
        _make_fixation_rows(0, 150, "img_A", rx=0.3, ry=0.6) + _make_gap_rows(160, 50) + _make_fixation_rows(220, 150, "img_B", rx=0.7, ry=0.6) + _make_gap_rows(380, 50) + _make_fixation_rows(440, 150, "img_A", rx=0.3, ry=0.6)
    )
    return pd.DataFrame(rows)

@pytest.fixture
def blink_scene():
    """
    Two discrete blink events in an otherwise clean scene
    """
    rows = []
    rows += [_row(i * 50) for i in range(10)]
    rows += [_row(i * 50, blink=True) for i in range(10, 13)]
    rows += [_row(i * 50) for i in range(13, 18)]
    rows += [_row(i * 50, blink=True) for i in range(18, 20)]
    rows += [_row(i * 50) for i in range(20, 25)]
    return pd.DataFrame(rows)


@pytest.fixture
def stimulus_config_basic():
    return {
        "img_A": {"category": "stimulus", "labels": ["sad face"]},
        "img_B": {"category": "stimulus", "labels": ["happy face"]},
        "img_C": {"category": "neutral", "labels": []},
    }

@pytest.fixture
def fixations_df_simple():
    """
    Two fixations: 
    - img_A (0-200ms)
    - img_B (320-520ms)
    """
    return pd.DataFrame([
        {"start_timestamp": 0.0, "end_timestamp": 200.0, "duration_ms": 200.0, "fpog_x": 0.3, "fpog_y": 0.6, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 13},
        {"start_timestamp": 320.0, "end_timestamp": 520.0, "duration_ms": 200.0, "fpog_x": 0.7, "fpog_y": 0.6, "rx": 0.7, "ry": 0.6, "image": "img_B", "n_samples": 13},
    ])

@pytest.fixture
def fixations_df_single():
    return pd.DataFrame([{
        "start_timestamp": 0.0, "end_timestamp": 200.0, "duration_ms": 200.0, "fpog_x": 0.3, "fpog_y": 0.6, "rx": 0.3, "ry": 0.6, "image": "img_A", "n_samples": 13,
    }])

@pytest.fixture
def fixations_df_empty():
    return pd.DataFrame(columns=[
        "start_timestamp", "end_timestamp", "duration_ms", "fpog_x", "fpog_y", "rx", "ry", "image", "n_samples",
    ])
