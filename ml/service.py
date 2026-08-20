from dataclasses import dataclass

from ml.history import JsonTrainingHistoryRepository
from ml.inference import predict_batch, predict_single
from ml.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PreparedDataset,
)
from ml.storage import JoblibModelRepository, ModelBundle
from ml.training import ModelTrainer
from schemas import (
    DatasetRowChurn,
    FeatureVectorChurn,
    MetricsHistoryResponse,
    ModelTypeChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
    TrainingHistoryRecord,
)


@dataclass
class ChurnModelService:
    dataset: list[DatasetRowChurn]
    split: PreparedDataset | None
    model_repository: JoblibModelRepository
    history_repository: JsonTrainingHistoryRepository
    trainer: ModelTrainer
    model_bundle: ModelBundle

    @classmethod
    def create(
        cls,
        dataset: list[DatasetRowChurn],
        split: PreparedDataset | None,
    ) -> "ChurnModelService":
        repository = JoblibModelRepository()
        return cls(
            dataset=dataset,
            split=split,
            model_repository=repository,
            history_repository=JsonTrainingHistoryRepository(),
            trainer=ModelTrainer(),
            model_bundle=repository.load(),
        )

    def train(self, config: TrainingConfigChurn) -> dict[str, float]:
        if not self.dataset or self.split is None:
            raise ValueError("Dataset is empty or unavailable")
        bundle = self.trainer.train(config, self.split)
        self.model_repository.save(bundle)
        self.model_bundle = bundle
        metadata = bundle.metadata
        if metadata.trained_at is None:
            raise ValueError("Model metadata is incomplete")
        self.history_repository.append(
            TrainingHistoryRecord(
                timestamp=metadata.trained_at,
                model_type=config.model_type,
                hyperparameters=config.hyperparameters,
                metrics=metadata.metrics or {},
            )
        )
        return metadata.metrics or {}

    def predict(
        self, features: FeatureVectorChurn | list[FeatureVectorChurn]
    ) -> PredictionResponseChurn | list[PredictionResponseChurn]:
        if not self.model_bundle.metadata.trained:
            raise ValueError("Model is not trained")
        if isinstance(features, list):
            return predict_batch(self.model_bundle, features)
        return predict_single(self.model_bundle, features)

    def metrics(
        self, model_type: ModelTypeChurn | None, limit: int
    ) -> MetricsHistoryResponse:
        history = self.history_repository.recent(model_type, limit)
        return MetricsHistoryResponse(
            latest=history[0] if history else None,
            history=history,
        )

    def schema(self) -> dict[str, list[dict[str, str]]]:
        def describe(columns: list[str]) -> list[dict[str, str]]:
            return [
                {
                    "name": name,
                    "type": FeatureVectorChurn.model_fields[name].annotation.__name__,
                }
                for name in columns
            ]

        return {
            "numeric": describe(NUMERIC_COLUMNS),
            "categorical": describe(CATEGORICAL_COLUMNS),
        }
