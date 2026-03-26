from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _unwrap_angle(x: np.ndarray) -> np.ndarray:
    """Unwrap a radian angle sequence.

    Parameters
    ----------
    x : np.ndarray
        Angle sequence in radians.

    Returns
    -------
    np.ndarray
        Unwrapped angle sequence.
    """
    return np.unwrap(x)


def enrich_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Derive velocity, speed, and yaw from position traces.

    Parameters
    ----------
    df : pd.DataFrame
        Processed telemetry dataframe.

    Returns
    -------
    pd.DataFrame
        Enriched telemetry dataframe.
    """
    df = df.copy()
    parts: list[pd.DataFrame] = []

    for _, group in df.groupby("run_id", sort=False):
        group = group.sort_values("frame_idx", kind="stable").copy()

        dt = group["t_run_s"].diff().to_numpy()
        dx = group["pos_x"].diff().to_numpy()
        dy = group["pos_y"].diff().to_numpy()
        dz = group["pos_z"].diff().to_numpy()

        vel_x = np.zeros(len(group), dtype=np.float64)
        vel_y = np.zeros(len(group), dtype=np.float64)
        vel_z = np.zeros(len(group), dtype=np.float64)

        valid = np.isfinite(dt) & (dt > 0.0)
        vel_x[valid] = dx[valid] / dt[valid]
        vel_y[valid] = dy[valid] / dt[valid]
        vel_z[valid] = dz[valid] / dt[valid]

        speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

        yaw = np.zeros(len(group), dtype=np.float64)
        moving = speed > 1.0
        yaw[moving] = np.arctan2(vel_y[moving], vel_x[moving])

        if moving.any():
            idx = np.flatnonzero(moving)
            yaw[: idx[0]] = yaw[idx[0]]
            yaw = pd.Series(yaw).replace(0.0, np.nan).ffill().bfill().to_numpy()
            yaw = _unwrap_angle(yaw)

        group["vel_x"] = vel_x
        group["vel_y"] = vel_y
        group["vel_z"] = vel_z
        group["speed_mps"] = speed
        group["yaw_rad"] = yaw

        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def enrich_telemetry_file(path: Path) -> Path:
    """Enrich a processed telemetry parquet file in place.

    Parameters
    ----------
    path : Path
        Path to processed parquet.

    Returns
    -------
    Path
        Output parquet path.
    """
    df = pd.read_parquet(path)
    df = enrich_telemetry(df)
    df.to_parquet(path, index=False)
    return path
