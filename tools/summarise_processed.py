from __future__ import annotations

import pandas as pd

from py_punk.paths import PROCESSED_TELEMETRY_DIR


def main() -> None:
    """Print a compact summary of a processed telemetry run.

    Returns
    -------
    None
    """
    path = PROCESSED_TELEMETRY_DIR / "dummy_run_001.parquet"
    df = pd.read_parquet(path)

    dt = df["t_run_s"].diff()
    dt = dt[dt.notna()]

    print(f"rows: {len(df)}")
    print(f"runs: {df['run_id'].nunique()}")
    print(f"duration_s: {df['t_run_s'].iloc[-1]:.3f}")
    print(f"median_dt_s: {dt.median():.6f}")
    print(f"mean_speed_mps: {df['speed_mps'].mean():.3f}")
    print(f"max_speed_mps: {df['speed_mps'].max():.3f}")
    print(f"terminal_rows: {int(df['episode_done'].sum())}")
    print(f"collisions: {int(df['collision_flag'].sum())}")


if __name__ == "__main__":
    main()
