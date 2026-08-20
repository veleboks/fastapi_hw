import json
from pathlib import Path
from threading import Lock

from core.config import TRAINING_HISTORY_FILEPATH
from schemas import ModelTypeChurn, TrainingHistoryRecord


class JsonTrainingHistoryRepository:
    def __init__(self, path: Path = TRAINING_HISTORY_FILEPATH):
        self.path = path
        self._lock = Lock()

    def load(self) -> list[TrainingHistoryRecord]:
        if not self.path.exists():
            return []
        raw_history = json.loads(self.path.read_text(encoding="utf-8"))
        return [TrainingHistoryRecord.model_validate(item) for item in raw_history]

    def append(self, record: TrainingHistoryRecord) -> None:
        with self._lock:
            history = self.load()
            history.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in history],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(self.path)

    def recent(
        self, model_type: ModelTypeChurn | None = None, limit: int = 10
    ) -> list[TrainingHistoryRecord]:
        history = self.load()
        if model_type is not None:
            history = [record for record in history if record.model_type == model_type]
        return list(reversed(history))[:limit]
