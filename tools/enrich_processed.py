from py_punk.enrich import enrich_telemetry_file
from py_punk.paths import PROCESSED_TELEMETRY_DIR


def main() -> None:
    """Enrich the probe telemetry parquet file.

    Returns
    -------
    None
    """
    path = PROCESSED_TELEMETRY_DIR / "telemetry_probe.parquet"
    print(enrich_telemetry_file(path))


if __name__ == "__main__":
    main()
