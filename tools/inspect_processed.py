import pandas as pd

from py_punk.paths import PROCESSED_TELEMETRY_DIR


def main() -> None:
    """Print a quick summary of the processed telemetry dataset.

    Returns
    -------
    None
    """
    path = PROCESSED_TELEMETRY_DIR / "dummy_run_001.parquet"
    df = pd.read_parquet(path)

    print(df.head())
    print()
    print(df.dtypes)
    print()
    print(df[["speed_mps", "steer_cmd"]].describe())


if __name__ == "__main__":
    main()
