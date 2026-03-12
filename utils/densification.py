import polars as pl


def get_dense(
    df: pl.DataFrame,
    target_col: str,
    date_col: str,
    cat_cols: list[str],
    interval: str = "1h",
):

    return (
        df.group_by(cat_cols)
        .agg(
            pl.datetime_range(
                pl.col(date_col).min(), pl.col(date_col).max(), interval=interval
            ).alias(date_col)
        )
        .explode(date_col)
        .join(df, on=[date_col, *cat_cols], how="left")
        .with_columns(pl.col(target_col).is_null().cast(pl.Int8).alias("was_null"))
    )
