from core.config import DATASET_FILEPATH
from ml.dataset import load_dataset
from ml.preprocessing import prepare_dataset
from ml.service import ChurnModelService


def create_model_service() -> ChurnModelService:
    dataset = load_dataset(DATASET_FILEPATH)
    split = prepare_dataset(dataset) if dataset else None
    return ChurnModelService.create(dataset, split)
