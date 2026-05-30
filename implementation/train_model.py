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

df = pd.read_csv(
    "../data/housing.csv"
)

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

joblib.dump(
    model,
    "../models/stacking_regressor.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("Model Saved Successfully")