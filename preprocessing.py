from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from schemas import DatasetRowChurn


NUMERIC_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_COLUMNS = ["region", "device_type", "payment_method"]
TARGET_COLUMN = "churn"


@dataclass
class PreparedDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]

    def split_info(self) -> dict[str, Any]:
        return {
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "train_churn_distribution": self.y_train.value_counts()
            .sort_index()
            .to_dict(),
            "test_churn_distribution": self.y_test.value_counts()
            .sort_index()
            .to_dict(),
        }


def prepare_dataset(
    dataset: list[DatasetRowChurn],
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedDataset:
    frame = pd.DataFrame([row.model_dump() for row in dataset])
    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # При DataFrame/Series на входе это именно DataFrame/Series, но stubs
    # scikit-learn описывают возвращаемые значения шире.
    X_train = cast(pd.DataFrame, X_train).copy()
    X_test = cast(pd.DataFrame, X_test).copy()
    y_train = cast(pd.Series, y_train)
    y_test = cast(pd.Series, y_test)

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    X_train[NUMERIC_COLUMNS] = numeric_imputer.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = numeric_imputer.transform(X_test[NUMERIC_COLUMNS])
    X_train[CATEGORICAL_COLUMNS] = categorical_imputer.fit_transform(
        X_train[CATEGORICAL_COLUMNS]
    )
    X_test[CATEGORICAL_COLUMNS] = categorical_imputer.transform(
        X_test[CATEGORICAL_COLUMNS]
    )

    return PreparedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
