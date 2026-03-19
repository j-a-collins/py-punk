from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = ROOT / "notes"
TELEMETRY_DIR = ROOT / "telemetry"
RAW_TELEMETRY_DIR = TELEMETRY_DIR / "raw"
PROCESSED_TELEMETRY_DIR = TELEMETRY_DIR / "processed"


def ensure_project_dirs() -> None:
    """
    create project dirs used by the telemetry pipeline.

    returns
    -------
    none
    """
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
