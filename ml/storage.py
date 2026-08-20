from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from core.config import CACHED_MODEL_FILEPATH
from schemas import TrainingConfigChurn


@dataclass
class ModelMetadata:
    config: TrainingConfigChurn | None = None
    trained: bool = False
    trained_at: datetime | None = None
    metrics: dict[str, float] | None = None


@dataclass
class ModelBundle:
    model: Any | None = None
    metadata: ModelMetadata = field(default_factory=ModelMetadata)


class JoblibModelRepository:
    def __init__(self, path: Path = CACHED_MODEL_FILEPATH):
        self.path = path

    def load(self) -> ModelBundle:
        if not self.path.exists():
            return ModelBundle()
        return joblib.load(self.path)

    def save(self, bundle: ModelBundle) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.path)
