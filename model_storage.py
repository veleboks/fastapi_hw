"""Backward-compatible imports for models serialized before the refactor."""

from ml.storage import JoblibModelRepository, ModelBundle, ModelMetadata

__all__ = ["JoblibModelRepository", "ModelBundle", "ModelMetadata"]
