from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a telemetry dataframe.

    Attributes
    ----------
    ok : bool
        Whether validation passed.
    errors : list[str]
        Hard validation failures.
    warnings : list[str]
        Non-fatal issues worth inspecting.
    """

    ok: bool
    errors: list[str]
    warnings: list[str]


def _check_monotonic_per_run(df: pd.DataFrame, column: str) -> list[str]:
    """Check that a column is monotonically non-decreasing within each run.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry dataframe.
    column : str
        Column name to check.

    Returns
    -------
    list[str]
        Validation errors.
    """
    errors: list[str] = []
    for run_id, group in df.groupby("run_id", sort=False):
        values = group[column].to_numpy()
        if np.any(np.diff(values) < 0):
            errors.append(f"{column} is not monotonic for run_id={run_id}")
    return errors


def validate_telemetry_df(df: pd.DataFrame) -> ValidationResult:
    """Validate processed telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Processed telemetry dataframe.

    Returns
    -------
    ValidationResult
        Validation outcome.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        errors.append("telemetry dataframe is empty")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    required = {
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
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    if df["run_id"].isna().any():
        errors.append("run_id contains nulls")

    if (df["frame_idx"] < 0).any():
        errors.append("frame_idx contains negative values")

    if (df["t_run_s"] < 0).any():
        errors.append("t_run_s contains negative values")

    if (df["speed_mps"] < 0).any():
        errors.append("speed_mps contains negative values")

    if ((df["throttle_cmd"] < 0.0) | (df["throttle_cmd"] > 1.0)).any():
        errors.append("throttle_cmd is outside [0, 1]")

    if ((df["brake_cmd"] < 0.0) | (df["brake_cmd"] > 1.0)).any():
        errors.append("brake_cmd is outside [0, 1]")

    if ((df["steer_cmd"] < -1.0) | (df["steer_cmd"] > 1.0)).any():
        errors.append("steer_cmd is outside [-1, 1]")

    errors.extend(_check_monotonic_per_run(df, "frame_idx"))
    errors.extend(_check_monotonic_per_run(df, "t_run_s"))
    errors.extend(_check_monotonic_per_run(df, "t_wall_s"))

    dupes = df.duplicated(subset=["run_id", "frame_idx"])
    if dupes.any():
        errors.append(f"duplicate (run_id, frame_idx) rows found: {int(dupes.sum())}")

    terminal_rows = df.groupby("run_id", sort=False)["episode_done"].sum()
    bad_terminal = terminal_rows[terminal_rows > 1]
    if not bad_terminal.empty:
        errors.append(
            "multiple episode_done rows found for run_ids: "
            + ", ".join(map(str, bad_terminal.index.tolist()))
        )

    no_terminal = terminal_rows[terminal_rows == 0]
    if not no_terminal.empty:
        warnings.append(
            "no terminal row found for run_ids: "
            + ", ".join(map(str, no_terminal.index.tolist()))
        )

    if df["player_in_vehicle"].mean() < 0.95:
        warnings.append("player_in_vehicle is false for a noticeable fraction of rows")

    dt = df.groupby("run_id", sort=False)["t_run_s"].diff()
    positive_dt = dt[dt.notna()]
    if not positive_dt.empty:
        median_dt = float(positive_dt.median())
        if median_dt > 0.1:
            warnings.append(f"median timestep is large: {median_dt:.4f}s")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_telemetry_file(path: Path) -> ValidationResult:
    """Validate a processed telemetry parquet file.

    Parameters
    ----------
    path : Path
        Path to processed telemetry parquet.

    Returns
    -------
    ValidationResult
        Validation outcome.
    """
    df = pd.read_parquet(path)
    return validate_telemetry_df(df)
