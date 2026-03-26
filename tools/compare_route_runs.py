from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py_punk.paths import NOTES_DIR, PROCESSED_TELEMETRY_DIR


@dataclass(slots=True)
class RunStats:
    """Summary statistics for a single run.

    Attributes
    ----------
    name : str
        Run name derived from the file stem.
    n_rows : int
        Number of telemetry rows.
    duration_s : float
        Run duration in seconds.
    path_length_m : float
        Total travelled path length in metres.
    mean_speed_mps : float
        Mean speed in metres per second.
    max_speed_mps : float
        Maximum speed in metres per second.
    start_x : float
        Starting x position.
    start_y : float
        Starting y position.
    end_x : float
        Final x position.
    end_y : float
        Final y position.
    """

    name: str
    n_rows: int
    duration_s: float
    path_length_m: float
    mean_speed_mps: float
    max_speed_mps: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float


def load_runs(pattern: str = "route_a_run_*.parquet") -> list[tuple[str, pd.DataFrame]]:
    """Load processed runs matching a filename pattern.

    Parameters
    ----------
    pattern : str, optional
        Glob pattern under the processed telemetry directory.

    Returns
    -------
    list[tuple[str, pd.DataFrame]]
        List of run names and dataframes.
    """
    paths = sorted(PROCESSED_TELEMETRY_DIR.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    runs: list[tuple[str, pd.DataFrame]] = []
    for path in paths:
        df = pd.read_parquet(path).sort_values("frame_idx", kind="stable").reset_index(drop=True)
        runs.append((path.stem, df))
    return runs


def cumulative_path_length(df: pd.DataFrame) -> np.ndarray:
    """Compute cumulative xy path length.

    Parameters
    ----------
    df : pd.DataFrame
        Run telemetry.

    Returns
    -------
    np.ndarray
        Cumulative path length in metres.
    """
    dx = df["pos_x"].diff().to_numpy()
    dy = df["pos_y"].diff().to_numpy()
    ds = np.sqrt(np.nan_to_num(dx) ** 2 + np.nan_to_num(dy) ** 2)
    return np.cumsum(ds)


def compute_stats(name: str, df: pd.DataFrame) -> RunStats:
    """Compute summary statistics for a run.

    Parameters
    ----------
    name : str
        Run name.
    df : pd.DataFrame
        Run telemetry.

    Returns
    -------
    RunStats
        Run summary.
    """
    s = cumulative_path_length(df)
    return RunStats(
        name=name,
        n_rows=len(df),
        duration_s=float(df["t_run_s"].iloc[-1]),
        path_length_m=float(s[-1]),
        mean_speed_mps=float(df["speed_mps"].mean()),
        max_speed_mps=float(df["speed_mps"].max()),
        start_x=float(df["pos_x"].iloc[0]),
        start_y=float(df["pos_y"].iloc[0]),
        end_x=float(df["pos_x"].iloc[-1]),
        end_y=float(df["pos_y"].iloc[-1]),
    )


def interpolate_speed_vs_progress(df: pd.DataFrame, n_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate speed on normalised path progress.

    Parameters
    ----------
    df : pd.DataFrame
        Run telemetry.
    n_points : int, optional
        Number of interpolation points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        normalised progress and interpolated speed.
    """
    s = cumulative_path_length(df)
    total = float(s[-1])
    if total <= 0.0:
        progress = np.linspace(0.0, 1.0, n_points)
        return progress, np.zeros_like(progress)

    progress_raw = s / total
    speed = df["speed_mps"].to_numpy()

    keep = np.r_[True, np.diff(progress_raw) > 0.0]
    progress_raw = progress_raw[keep]
    speed = speed[keep]

    progress = np.linspace(0.0, 1.0, n_points)
    speed_interp = np.interp(progress, progress_raw, speed)
    return progress, speed_interp


def print_summary_table(stats: list[RunStats]) -> pd.DataFrame:
    """Print a compact summary table.

    Parameters
    ----------
    stats : list[RunStats]
        Per-run statistics.

    Returns
    -------
    pd.DataFrame
        Summary dataframe.
    """
    df = pd.DataFrame([asdict(s) for s in stats])
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    durations = df["duration_s"].to_numpy()
    lengths = df["path_length_m"].to_numpy()
    print()
    print(f"duration range (s): {durations.min():.3f} -> {durations.max():.3f}")
    print(f"path length range (m): {lengths.min():.3f} -> {lengths.max():.3f}")
    return df


def ensure_notes_assets_dir() -> Path:
    """Create the notes asset directory.

    Returns
    -------
    Path
        Asset directory path.
    """
    out_dir = NOTES_DIR / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_trajectories(runs: list[tuple[str, pd.DataFrame]], out_dir: Path) -> Path:
    """Plot xy trajectories for all runs and save the figure.

    Parameters
    ----------
    runs : list[tuple[str, pd.DataFrame]]
        Run names and dataframes.
    out_dir : Path
        Output directory.

    Returns
    -------
    Path
        Saved figure path.
    """
    out_path = out_dir / "route_a_trajectories.png"

    plt.figure(figsize=(8, 8))
    for name, df in runs:
        plt.plot(df["pos_x"], df["pos_y"], label=name)
    plt.xlabel("pos_x")
    plt.ylabel("pos_y")
    plt.title("Route A trajectories")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return out_path


def plot_speed_vs_progress(runs: list[tuple[str, pd.DataFrame]], out_dir: Path) -> Path:
    """Plot speed against normalised route progress and save the figure.

    Parameters
    ----------
    runs : list[tuple[str, pd.DataFrame]]
        Run names and dataframes.
    out_dir : Path
        Output directory.

    Returns
    -------
    Path
        Saved figure path.
    """
    out_path = out_dir / "route_a_speed_vs_progress.png"

    plt.figure(figsize=(8, 4))
    for name, df in runs:
        progress, speed = interpolate_speed_vs_progress(df)
        plt.plot(progress, speed, label=name)
    plt.xlabel("normalised progress")
    plt.ylabel("speed_mps")
    plt.title("Speed vs route progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    return out_path


def main() -> None:
    """Compare repeated runs on the same route.

    Returns
    -------
    None
    """
    runs = load_runs()
    stats = [compute_stats(name, df) for name, df in runs]
    print_summary_table(stats)

    out_dir = ensure_notes_assets_dir()
    traj_path = plot_trajectories(runs, out_dir)
    speed_path = plot_speed_vs_progress(runs, out_dir)

    print()
    print(f"saved: {traj_path}")
    print(f"saved: {speed_path}")


if __name__ == "__main__":
    main()