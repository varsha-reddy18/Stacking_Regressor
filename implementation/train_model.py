import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "housing.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop(
    "Price",
    axis=1
)

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

estimators = [

    (
        "rf",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    ),

    (
        "gb",
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
    )

]

model = StackingRegressor(

    estimators=estimators,

    final_estimator=LinearRegression(),

    cv=5

)

model.fit(
    X_train,
    y_train
)
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "stacking_regressor.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

print("Model Saved Successfully")