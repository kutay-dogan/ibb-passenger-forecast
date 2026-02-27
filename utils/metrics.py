import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    root_mean_squared_error,
)


def get_all_metrics(results: pl.DataFrame, cat_col, target_col):

    def calculate_metrics(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        y = df[target_col].to_numpy()
        y_pred = df["prediction"].to_numpy()

        return pl.Series(
            [
                {
                    "rmse": root_mean_squared_error(y, y_pred),
                    "medae": median_absolute_error(y, y_pred),
                    "mae": mean_absolute_error(y, y_pred),
                    "mape": mean_absolute_percentage_error(y, y_pred),
                }
            ]
        )

    by_all = (
        (
            results.group_by("split", cat_col)
            .agg(
                pl.struct(target_col, "prediction")
                .map_batches(calculate_metrics)
                .alias("metrics"),
            )
            .with_columns(pl.col("metrics").list.first())
            .unnest("metrics")
        )
        .sort(["split", cat_col])
        .with_columns(pl.selectors.float().cast(pl.Float32))
    )

    by_cat = (
        (
            results.group_by(cat_col)
            .agg(
                pl.struct(target_col, "prediction")
                .map_batches(calculate_metrics)
                .alias("metrics"),
            )
            .with_columns(pl.col("metrics").list.first())
            .unnest("metrics")
        )
        .sort(cat_col)
        .with_columns(pl.selectors.float().cast(pl.Float32))
    )

    return by_all, by_cat
