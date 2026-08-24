import csv
import logging
from pathlib import Path
from typing import Any

from schemas import DatasetRowChurn

logger = logging.getLogger(__name__)


def _normalize_csv_row(row: dict[str, str]) -> dict[str, str | None]:
    return {
        name: stripped if (stripped := value.strip()) else None
        for name, value in row.items()
    }


def load_dataset(file_path: Path) -> list[DatasetRowChurn]:
    try:
        file = file_path.open("r", newline="", encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Dataset file not found: %s", file_path)
        return []

    with file:
        dataset = [
            DatasetRowChurn.model_validate(_normalize_csv_row(row))
            for row in csv.DictReader(file)
        ]
    logger.info("Dataset loaded: path=%s rows=%d", file_path, len(dataset))
    return dataset


def info_dataset(dataset: list[DatasetRowChurn]) -> dict[str, Any]:
    zeros = sum(row.churn == 0 for row in dataset)
    ones = sum(row.churn == 1 for row in dataset)
    return {
        "num_rows": len(dataset),
        "num_cols": len(DatasetRowChurn.model_fields),
        "cols": list(DatasetRowChurn.model_fields),
        "churn": {"0": zeros, "1": ones},
    }
