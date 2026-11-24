from pathlib import Path

import pandas as pd
import streamlit as st

from model import FEATURES, train_model


DATA_PATH = Path(__file__).parent / "data" / "train.csv"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write(
    "Enter a few property details to estimate its sale price. "
    "This demo uses the House Price India dataset."
)


@st.cache_resource
def load_model(csv_path: str, modified_time: float):
    data = pd.read_csv(csv_path)
    return train_model(data)


if not DATA_PATH.exists():
    st.warning("Download the Kaggle training CSV and save it as `data/train.csv`.")
    st.stop()

try:
    model, metrics = load_model(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except (ValueError, pd.errors.ParserError, OSError) as error:
    st.error(f"Could not load or train the model: {error}")
    st.stop()

st.subheader("Model performance")
metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Mean absolute error", f"${metrics['mae']:,.0f}")
metric_col2.metric("R² score", f"{metrics['r2']:.3f}")

st.subheader("Property details")
defaults = {
    "number of bedrooms": 3,
    "number of bathrooms": 2.0,
    "living area": 1800,
    "lot area": 5000,
    "number of floors": 1.0,
    "condition of the house": 3,
    "grade of the house": 7,
    "Area of the basement": 0,
    "Built Year": 2000,
}
labels = {
    "number of bedrooms": "Bedrooms",
    "number of bathrooms": "Bathrooms",
    "living area": "Living area",
    "lot area": "Lot area",
    "number of floors": "Floors",
    "condition of the house": "Condition (1–5)",
    "grade of the house": "Grade (1–13)",
    "Area of the basement": "Basement area",
    "Built Year": "Year built",
}

with st.form("prediction_form"):
    columns = st.columns(2)
    values = {}
    for index, feature in enumerate(FEATURES):
        with columns[index % 2]:
            values[feature] = st.number_input(
                labels[feature],
                min_value=0.0 if isinstance(defaults[feature], float) else 0,
                value=defaults[feature],
                step=0.5 if isinstance(defaults[feature], float) else 1,
            )
    submitted = st.form_submit_button("Estimate sale price", type="primary")

if submitted:
    input_data = pd.DataFrame([values], columns=FEATURES)
    estimated_price = model.predict(input_data)[0]
    st.success(f"Estimated price: **{estimated_price:,.0f}**")
    st.caption("Price is shown in the same units as the dataset.")
