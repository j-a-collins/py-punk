from pathlib import Path

import numpy as np
import pandas as pd

from py_punk.paths import RAW_TELEMETRY_DIR, ensure_project_dirs


def main() -> None:
    """Generate a dummy telemetry run for pipeline testing.

    Returns
    -------
    None
    """
    ensure_project_dirs()

    n = 600
    dt = 1.0 / 60.0
    t = np.arange(n, dtype=np.float64) * dt
    speed = np.clip(12.0 * (1.0 - np.exp(-0.8 * t)), 0.0, None)
    yaw = 0.02 * np.sin(0.5 * t)
    steer = 0.1 * np.sin(0.7 * t)
    throttle = np.where(t < 4.0, 0.6, 0.2)
    brake = np.zeros_like(t)

    x = np.cumsum(speed * np.cos(yaw) * dt)
    y = np.cumsum(speed * np.sin(yaw) * dt)
    z = np.zeros_like(t)

    vel_x = speed * np.cos(yaw)
    vel_y = speed * np.sin(yaw)
    vel_z = np.zeros_like(t)

    df = pd.DataFrame(
        {
            "run_id": "dummy_run_001",
            "frame_idx": np.arange(n, dtype=np.int64),
            "t_wall_s": 1_700_000_000.0 + t,
            "t_run_s": t,
            "player_in_vehicle": True,
            "vehicle_id": "test_vehicle",
            "pos_x": x,
            "pos_y": y,
            "pos_z": z,
            "vel_x": vel_x,
            "vel_y": vel_y,
            "vel_z": vel_z,
            "speed_mps": speed,
            "yaw_rad": yaw,
            "pitch_rad": 0.0,
            "roll_rad": 0.0,
            "throttle_cmd": throttle,
            "brake_cmd": brake,
            "steer_cmd": steer,
            "manual_override": True,
            "collision_flag": False,
            "episode_done": False,
            "done_reason": "",
        }
    )

    df.loc[n - 1, "episode_done"] = True
    df.loc[n - 1, "done_reason"] = "timeout"

    out_path = RAW_TELEMETRY_DIR / "dummy_run_001.csv"
    df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()