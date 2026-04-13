"""
Data loading utilities
"""

import json
import os
from typing import Dict, List, Optional
import pandas as pd

def load_stimulus_config(json_path: str) -> Dict:
    """
    Load a single JSON stimulus config
    """
    with open(json_path, "r") as f:
        return json.load(f)

def load_forms(forms_path) -> pd.DataFrame:
    """
    Load forms.csv
    """
    df = pd.read_csv(forms_path)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    return df

def list_session_files(folder_path: str) -> List[str]:
    """
    List all CSV files paths for a given folder path
    """    
    files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".csv"):
            files.append(os.path.join(folder_path, fname))
    return sorted(files)


def session_id_from_path(filepath: str) -> str:
    """
    Get session_id from a CSV file
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def load_session(file_path: str) -> pd.DataFrame:
    """
    Load and clean single session CSV into a pandas DataFrame.    
    Rows with NaN SCENE_INDEX are dropped
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    df = df.dropna(subset=["SCENE_INDEX"])
    
    df["SCENE_INDEX"] = df["SCENE_INDEX"].astype(int)
    df["BLINK"] = df["BLINK"].astype(bool)
    df["FEV"] = pd.to_numeric(df["FEV"], errors="coerce")
    df["FDUR"] = pd.to_numeric(df["FDUR"], errors="coerce")
    df["FPOGX"] = pd.to_numeric(df["FPOGX"], errors="coerce")
    df["FPOGY"] = pd.to_numeric(df["FPOGY"], errors="coerce")
    
    return df