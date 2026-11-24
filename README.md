# House Price Predictor

A small house sale-price prediction project with a Python/scikit-learn model and a Streamlit interface. It uses the **House Price India** CSV dataset.

## 1. Set up Python

Python 3.9 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Add the dataset

Download and extract the House Price India dataset. Copy `House Price India.csv` into this project's `data` folder and rename it `train.csv`:

```text
tns/
├── app.py
└── data/
    └── train.csv
```

The app reads the CSV with `pandas.read_csv` from `data/train.csv`. To check the file can be opened, run this from the project folder:

```powershell
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').head())"
```

The model predicts `Price` using bedrooms, bathrooms, living and lot area, floors, house condition and grade, basement area, and year built. It displays predictions in the same units as the dataset.

The app uses the `SalePrice` target and these numeric columns: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`, `FullBath`, `BedroomAbvGr`, and `LotArea`.

## 3. Run the app

From the project folder:

```powershell
python -m streamlit run app.py
```

The app trains a Random Forest model from `data/train.csv`, shows holdout-set MAE and R², and lets you enter property details to get an estimated price.

## Project files

- `app.py` — Streamlit frontend
- `model.py` — data checks, preprocessing, training, and evaluation
- `data/train.csv` — Kaggle training data you download manually (not committed)
