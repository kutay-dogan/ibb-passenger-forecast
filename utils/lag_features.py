import duckdb as ddb
import polars as pl


def add_lag_features(df, target_col, cat_cols, date_col, lags, lag_unit="DAY"):
    cat_col_str = ", ".join(col for col in cat_cols)
    constants = f"{date_col}, {target_col}, {cat_col_str},\n"

    select_clauses = []
    cte_select_clauses = []

    select_clauses.append(constants)
    cte_select_clauses.append("a.*,\n")

    select_clauses.append(
        "".join(
            f"{date_col} + INTERVAL {lag} {lag_unit} AS {date_col}_future_{lag},\n"
            for lag in lags
        )
    )
    select_clauses_str = "".join(select_clauses)

    join_condition_str = "\n".join(
        f"""LEFT JOIN future_table b_{lag} 
        ON a.{date_col} = b_{lag}.{date_col}_future_{lag} AND {
            "AND ".join([f"a.{cat} = b_{lag}.{cat} " for cat in cat_cols])
        }"""
        for lag in lags
    )

    cte_select_clauses.append(
        ",\n".join(f"b_{lag}.{target_col} AS {target_col}_lag_{lag}" for lag in lags)
    )
    cte_select_clauses_str = "".join(cte_select_clauses)

    return (
        ddb.sql(
            f"""
    WITH future_table AS (
        SELECT
            {select_clauses_str}
        FROM 
            df
    )

    SELECT
        {cte_select_clauses_str}
    FROM
        df a
    {join_condition_str}
    """
        )
        .pl(lazy=True)
        .with_columns(pl.selectors.numeric().cast(pl.Float32))
    )
