import json
from pathlib import Path

from schemas import TrainingHistoryRecord

TRAINING_HISTORY_FILEPATH = Path("artifacts/models/training_history.json")


def load_training_history() -> list[TrainingHistoryRecord]:
    if not TRAINING_HISTORY_FILEPATH.exists():
        return []

    raw_history = json.loads(TRAINING_HISTORY_FILEPATH.read_text(encoding="utf-8"))
    return [TrainingHistoryRecord.model_validate(item) for item in raw_history]


def append_training_record(record: TrainingHistoryRecord) -> None:
    history = load_training_history()
    history.append(record)

    TRAINING_HISTORY_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = TRAINING_HISTORY_FILEPATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in history],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(TRAINING_HISTORY_FILEPATH)
