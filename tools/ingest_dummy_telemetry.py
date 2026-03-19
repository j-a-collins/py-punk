from py_punk.ingest import ingest_telemetry
from py_punk.paths import RAW_TELEMETRY_DIR


def main() -> None:
    """Ingest the dummy telemetry run into processed parquet.

    Returns
    -------
    None
    """
    out = ingest_telemetry(RAW_TELEMETRY_DIR / "dummy_run_001.csv")
    print(out)


if __name__ == "__main__":
    main()