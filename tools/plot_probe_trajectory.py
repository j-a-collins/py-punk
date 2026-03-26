import matplotlib.pyplot as plt
import pandas as pd

from py_punk.paths import PROCESSED_TELEMETRY_DIR


def main() -> None:
    """Plot xy trajectory and derived speed.

    Returns
    -------
    None
    """
    df = pd.read_parquet(PROCESSED_TELEMETRY_DIR / "telemetry_probe.parquet")

    plt.figure(figsize=(8, 8))
    plt.plot(df["pos_x"], df["pos_y"])
    plt.xlabel("pos_x")
    plt.ylabel("pos_y")
    plt.title("Probe trajectory")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(df["t_run_s"], df["speed_mps"])
    plt.xlabel("t_run_s")
    plt.ylabel("speed_mps")
    plt.title("Derived speed over time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
