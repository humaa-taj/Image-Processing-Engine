"""Project-wide paths and constants. Every script imports from here so nothing is hard-coded twice."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data (READ-ONLY, never write here)
DATA = ROOT / "data"
PET_DIR = DATA / "oxford-iiit-pet"
FS2K_DIR = DATA / "fs2k" / "FS2K"

# Derived outputs
ARTIFACTS = ROOT / "artifacts"
SPLITS = ARTIFACTS / "splits"
MANIFESTS = ARTIFACTS / "manifests"
CACHE = ARTIFACTS / "cache"          # gitignored (large)
OPTUNA_DIR = ARTIFACTS / "optuna"
RESULTS = ARTIFACTS / "results"
FIGURES = ROOT / "docs" / "figures"
CHECKPOINTS = ROOT / "models" / "checkpoints"   # gitignored
ONNX_DIR = ROOT / "models" / "onnx"             # committed
MLRUNS = ROOT / "mlruns"                        # gitignored

IMG_SIZE = 128
SEED = 42

# Input conditions for Tasks 1-3. The list index is the classifier label and
# also the branch order of the Task 3 gate: [identity, salt, blur, occlusion].
CONDITIONS = ["clean", "salt", "blur", "occlusion"]
CLEAN, SALT, BLUR, OCCLUSION = range(4)

# Training corruption ranges (assignment table)
SALT_P_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_RECTS_RANGE = (1, 3)
OCC_AREA_RANGE = (0.10, 0.35)

# Fixed test severities: level -> parameters (assignment text)
LEVELS = ["low", "medium", "high"]
TEST_SALT_P = {"low": 0.03, "medium": 0.08, "high": 0.15}
TEST_BLUR = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}
TEST_OCCLUSION = {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}  # (rectangles, area)

# FS2K
FS2K_STYLES = 3          # JSON field "style" in {0,1,2} = UI "Style 1/2/3"
FS2K_VAL_FRACTION = 0.15
