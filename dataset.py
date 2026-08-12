import csv
from schemas import DatasetRowChurn
from pathlib import Path
from typing import Any


def load_dataset(file_path: Path) -> list[DatasetRowChurn]:
    with open(file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = [DatasetRowChurn.model_validate(row) for row in reader]
    return rows


def info_dataset(dataset: list[DatasetRowChurn]) -> dict[str, Any]:
    zeros = 0
    ones = 0
    for row in dataset:
        if row.churn == 1:
            ones += 1
        elif row.churn == 0:
            zeros += 1
    return {
        "num_rows": len(dataset),
        "num_cols": len(DatasetRowChurn.model_fields),
        "cols": list(DatasetRowChurn.model_fields),
        "churn": {
            "0": zeros,
            "1": ones,
        },
    }
