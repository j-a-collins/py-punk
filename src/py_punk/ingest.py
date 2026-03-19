from pathlib import Path

import pandas as pd

from py_punk.paths import PROCESSED_TELEMETRY_DIR, ensure_project_dirs

REQUIRED_COLUMNS = [
    "run_id",
    "frame_idx",
    "t_wall_s",
    "t_run_s",
    "player_in_vehicle",
    "vehicle_id",
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "speed_mps",
    "yaw_rad",
    "pitch_rad",
    "roll_rad",
    "throttle_cmd",
    "brake_cmd",
    "steer_cmd",
    "manual_override",
    "collision_flag",
    "episode_done",
    "done_reason",
]


def _read_raw(path: Path) -> pd.DataFrame:
    """
    load raw telemetry file.

    parameters
    ----------
    path : Path
        path to a CSV, JSONL, parquet, or line-delimited JSON file.

    returns
    -------
    pd.DataFrame
        loaded telemetry frame data.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported telemetry format: {path}")


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """normalise telemetry column ordering and dtypes.

    parameters
    ----------
    df : pd.DataFrame
        raw telemetry frame data.

    returns
    -------
    pd.DataFrame
        normalised telemetry data.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    df["vehicle_id"] = df["vehicle_id"].astype("string")
    df["done_reason"] = df["done_reason"].fillna("").astype("string")
    df["run_id"] = df["run_id"].astype("string")

    int_cols = ["frame_idx"]
    float_cols = [
        "t_wall_s",
        "t_run_s",
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "speed_mps",
        "yaw_rad",
        "pitch_rad",
        "roll_rad",
        "throttle_cmd",
        "brake_cmd",
        "steer_cmd",
    ]
    bool_cols = [
        "player_in_vehicle",
        "manual_override",
        "collision_flag",
        "episode_done",
    ]

    for col in int_cols:
        df[col] = df[col].astype("int64")
    for col in float_cols:
        df[col] = df[col].astype("float64")
    for col in bool_cols:
        df[col] = df[col].astype("bool")

    df = df.sort_values(["run_id", "frame_idx"], kind="stable").reset_index(drop=True)
    return df


def ingest_telemetry(path: Path, output_name: str | None = None) -> Path:
    """
    convert a raw telemetry file into normalised parquet.

    parameters
    ----------
    path : Path
        raw telemetry file path.
    output_name : str | None, optional
        output parquet filename stem. if omitted, derived from the input filename.

    returns
    -------
    Path
        path to the processed parquet file.
    """
    ensure_project_dirs()
    df = _coerce_columns(_read_raw(path))
    stem = output_name or path.stem
    out_path = PROCESSED_TELEMETRY_DIR / f"{stem}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
