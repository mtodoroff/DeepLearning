from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
RANDOM_SEED = 42
