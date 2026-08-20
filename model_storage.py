from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from schemas import TrainingConfigChurn

CACHED_MODEL_FILEPATH = Path("artifacts/models/cached_model.joblib")


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


def load_cached_model_bundle() -> ModelBundle:
    if not CACHED_MODEL_FILEPATH.exists():
        return ModelBundle()

    return joblib.load(CACHED_MODEL_FILEPATH)


def dump_model_bundle(bundle: ModelBundle):
    joblib.dump(bundle, CACHED_MODEL_FILEPATH)
