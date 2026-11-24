"""Model training and prediction helpers for the house price app."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


FEATURES = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "lot area",
    "number of floors",
    "condition of the house",
    "grade of the house",
    "Area of the basement",
    "Built Year",
]
TARGET = "Price"


def train_model(data: pd.DataFrame):
    """Train a model and return it with holdout-set evaluation metrics."""
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )

    training_data = data[required_columns].copy()
    for column in required_columns:
        training_data[column] = pd.to_numeric(training_data[column], errors="coerce")
    training_data = training_data.dropna(subset=[TARGET])

    if len(training_data) < 10:
        raise ValueError("The training CSV must contain at least 10 rows with Price.")

    features = training_data[FEATURES]
    target = training_data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                FEATURES,
            )
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    # Refit on all available rows before the app uses the model for predictions.
    model.fit(features, target)
    return model, metrics
