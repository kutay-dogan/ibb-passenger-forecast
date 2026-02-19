import duckdb as ddb
import polars as pl


def add_stat_features(df, target_col, date_col, cat_cols, intervals):
    cat_col_str = ", ".join(col for col in cat_cols)
    cat_join_str = " AND ".join(f"a.{col} = b.{col}" for col in cat_cols)
    select_clauses = []
    window_clauses = []

    select_clauses.append(f"{target_col}, {date_col}, {cat_col_str},")
    for interval in intervals:
        suffix = interval.lower().replace(" ", "_")
        window_name = f"w_{suffix}"

        features = f"""
            AVG({target_col}) OVER {window_name} AS avg_{target_col}_{suffix},
            MIN({target_col}) OVER {window_name} AS min_{target_col}_{suffix},
            MAX({target_col}) OVER {window_name} AS max_{target_col}_{suffix},
            QUANTILE({target_col}, 0.25) OVER {window_name} AS q25_{target_col}_{suffix},
            QUANTILE({target_col}, 0.50) OVER {window_name} AS q50_{target_col}_{suffix},
            QUANTILE({target_col}, 0.75) OVER {window_name} AS q75_{target_col}_{suffix},
            STDDEV_SAMP({target_col}) OVER {window_name} AS std_{target_col}_{suffix},
            SKEWNESS({target_col}) OVER {window_name} AS skew_{target_col}_{suffix},
            KURTOSIS({target_col}) OVER {window_name} AS kurt_{target_col}_{suffix},
            EXP(AVG(LN({target_col}+1)) OVER {window_name}) -1 AS geomean_{target_col}_{suffix},
            SUM({target_col}) OVER {window_name} AS sum_{target_col}_{suffix},
            SUM({target_col}*{target_col}) OVER {window_name} AS abs_energy_{target_col}_{suffix},
            REGR_SLOPE({target_col}, EPOCH({date_col})) OVER {window_name} AS slope_{target_col}_{suffix},
        """

        select_clauses.append(features)

        window_def = f"""
        {window_name} AS (
            PARTITION BY {cat_col_str}
            ORDER BY {date_col}
            RANGE BETWEEN INTERVAL {interval} PRECEDING AND CURRENT ROW
        )"""
        window_clauses.append(window_def)

    full_select_str = "\n    ".join(select_clauses)
    full_window_str = ",\n".join(window_clauses)
    final_query = f"""
        WITH features AS (
            SELECT 
                {full_select_str}
            FROM df
            WINDOW 
                {full_window_str}
            ORDER BY {date_col}, {cat_col_str}
        )

        SELECT
            a.*,
            b.* EXCLUDE({date_col}, {target_col}, {cat_col_str})
        FROM
            df a
        LEFT JOIN
            features b
        ON
            b.{date_col} + INTERVAL 30 DAYS = a.{date_col}
            AND {cat_join_str}
        ORDER BY a.{date_col}
        """

    return (
        ddb.sql(final_query)
        .pl(lazy=True)
        .with_columns(pl.selectors.numeric().cast(pl.Float32))
    )

