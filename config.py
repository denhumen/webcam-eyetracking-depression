from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports" / "figures"

# Volume paths
VOLUME_BASE = "/Volumes/workspace/anima/anima_volume/raw/dynamic_ab_research"
FORMS_PATH = f"{VOLUME_BASE}/forms.csv"

# Table names
TABLE_STIMULI = "anima.stimuli"
TABLE_FORMS = "anima.forms"
TABLE_SESSIONS = "anima.sessions"
TABLE_SCENE_METRICS = "anima.scene_metrics"

# Stimulus sets = configuration of the images shown to users, saved in json files 
# dict: folder_name : json_filename
STIMULUS_SETS = {
    "depression": "depression.json",
    "Depression_2": "Depression_2.json",
    "Depression_3": "Depression_3.json",
    "Depression_4": "Depression_4.json",
}

# Which folders to process in pipeline 04
FOLDERS_TO_PROCESS = ["depression", "Depression_2", "Depression_3", "Depression_4"]

# Valence classification keywords
NEGATIVE_KEYWORDS = {"sad face", "sad", "funeral", "suicide", "death", "breakup"}
POSITIVE_KEYWORDS = {"happy face", "positive", "happy", "wedding", "party", "pregnant"}

# Quality thresholds

# Session-level
SESSION_MIN_SAMPLES = 200
SESSION_MIN_DURATION_SEC = 30
SESSION_MAX_MISSING_PCT = 0.50
SESSION_MIN_SCENES = 10

# Trial-level (scene)
SCENE_MIN_SAMPLES = 20
SCENE_MIN_DURATION_MS = 1000
SCENE_MAX_MISSING_PCT = 0.40
SCENE_MAX_BLINK_PCT = 0.40
SCENE_MIN_FIXATIONS = 1
