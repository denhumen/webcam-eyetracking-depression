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
NEGATIVE_KEYWORDS = {"sad face", "funeral", "suicide", "death", "dep", "breakup"}
POSITIVE_KEYWORDS = {"happy face", "positive", "happy", "wedding", "party", "pregnant"}