import polars as pl
from datetime import timedelta
from datetime import datetime
from dataclasses import dataclass
from xgboost import XGBRegressor
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error,
    median_absolute_error,
    root_mean_squared_error,
)

target_col = "passage"
date_col = "timestamp"
cat_col = "station"

df = (
    pl.read_parquet("data/Xy.parquet")
    .filter(pl.col("timestamp") >= datetime(2023, 1, 1, 0, 0, 0))
    .drop("line_name")
)


@dataclass
class Range:
    start: datetime
    end: datetime


test_split = [
    Range(
        start=datetime(2024, 6, 30, 23, 0) - timedelta(days=120),
        end=datetime(2024, 6, 30, 23, 0) - timedelta(days=90),
    ),
    Range(
        start=datetime(2024, 6, 30, 23, 0) - timedelta(days=90),
        end=datetime(2024, 6, 30, 23, 0) - timedelta(days=60),
    ),
    Range(
        start=datetime(2024, 6, 30, 23, 0) - timedelta(days=60),
        end=datetime(2024, 6, 30, 23, 0) - timedelta(days=30),
    ),
    Range(
        start=datetime(2024, 6, 30, 23, 0) - timedelta(days=30),
        end=datetime(2024, 6, 30, 23, 0),
    ),
]


def train_xgb(Xy, params, log1p: bool = True, weighted: bool = True) -> XGBRegressor:
    xgb = XGBRegressor(
        random_state=42,
        enable_categorical=True,
        verbosity=0,
        eval_metric="mae",
        tree_method="hist",
        device="cuda",
        **params,
    )

    X = Xy.drop([date_col, target_col]).to_pandas()
    X[cat_col] = X[cat_col].astype("category")

    if log1p:
        y = Xy.select(pl.col(target_col).log1p()).to_numpy().reshape(-1)
    else:
        y = Xy.select(target_col).to_numpy().reshape(-1)

    if weighted:
        xgb.fit(X, y, sample_weight=1 / y)
    else:
        xgb.fit(X, y)

    return xgb


def predict_xgb(Xy, xgb: XGBRegressor, log1p: bool = True):
    X = Xy.drop([date_col, target_col]).to_pandas()
    X[cat_col] = X[cat_col].astype("category")

    if log1p:
        y_pred = np.expm1(xgb.predict(X))
    else:
        y_pred = xgb.predict(X)

    return y_pred


def cv(df: pl.DataFrame, params, log1p: bool = True) -> float:
    df = df.with_columns(
        [
            df[col].cast(pl.Float32)
            for col, dtype in zip(df.columns, df.dtypes)
            if dtype == pl.Float64
        ]
    )

    xgb = XGBRegressor(
        random_state=42,
        enable_categorical=True,
        verbosity=0,
        eval_metric="mae",
        device="cuda",
        **params,
    )

    mape = []
    mae = []
    mae_w = []
    rmse = []

    for split in tqdm(test_split, desc="Processing test splits"):
        Xy_train = df.filter(pl.col(date_col) <= split.start)
        Xy_test = df.filter(pl.col(date_col) > split.start).filter(
            pl.col(date_col) <= split.end
        )

        tqdm.write(f"Train: {len(Xy_train)} / Test:  {len(Xy_test)}")

        xgb = train_xgb(Xy_train, params, log1p)

        y_pred = predict_xgb(Xy_test, xgb, log1p)

        y = Xy_test.select(target_col).to_numpy().reshape(-1)

        mape.append(mean_absolute_percentage_error(y, y_pred))
        mae.append(median_absolute_error(y, y_pred))
        mae_w.append(median_absolute_error(y, y_pred, sample_weight=1 / y))
        rmse.append(root_mean_squared_error(y, y_pred))

    print("MAE: {:.4f} {}".format(np.array(mae).mean(), np.array(mae)))
    print("MAE-w: {:.4f} {}".format(np.array(mae_w).mean(), np.array(mae_w)))
    print("MAPE: {:.4f} {}".format(np.array(mape).mean(), np.array(mape)))
    print("RMSE: {:.4f}".format(np.array(rmse).mean()))

    return np.array(mae_w).mean()


LOG1P = True

param = {
    "max_depth": 5,
    "learning_rate": 0.02,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 3,
    "objective": "reg:absoluteerror",
}

cv(df.drop_nulls(target_col), param, log1p=LOG1P)
