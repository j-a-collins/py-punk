from __future__ import annotations

import sys

from py_punk.paths import PROCESSED_TELEMETRY_DIR
from py_punk.validate import validate_telemetry_file


def main() -> int:
    """Validate the processed dummy telemetry file.

    Returns
    -------
    int
        Process exit code.
    """
    path = PROCESSED_TELEMETRY_DIR / "dummy_run_001.parquet"
    result = validate_telemetry_file(path)

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print()

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
