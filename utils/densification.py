import polars as pl


def get_dense(
    df: pl.LazyFrame,
    target_col: str,
    date_col: str,
    cat_cols: list[str],
    interval: str = "1h",
):

    bounds = df.select(
        [pl.col(date_col).min().alias("min"), pl.col(date_col).max().alias("max")]
    ).collect()

    date_range_df = pl.LazyFrame(
        {
            date_col: pl.datetime_range(
                bounds.get_column("min")[0],
                bounds.get_column("max")[0],
                interval=interval,
                eager=True,
            )
        }
    )

    return (
        date_range_df.join(df.select(cat_cols).unique(), how="cross")
        .join(df, on=[date_col, *cat_cols], how="left")
        .with_columns(pl.col(target_col).is_null().cast(pl.Int8).alias("was_null"))
    )
