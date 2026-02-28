import duckdb as ddb
import polars as pl
from holidays import Turkey


class Holiday:
    def __init__(self, start=2021, end=2025):
        self.holidays = Turkey(years=[start, end])

    def __call__(self, date) -> int:
        return int(date in self.holidays)


ddb.create_function("is_holiday", Holiday())


def add_date_features(df, date_col):
    date_df = ddb.sql(f"""
    SELECT 
        day,
        is_holiday::BOOLEAN AS is_holiday,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW)::INT AS hc_in_last_7_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN INTERVAL 5 DAYS PRECEDING AND CURRENT ROW)::INT AS hc_in_last_6_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN INTERVAL 4 DAYS PRECEDING AND CURRENT ROW)::INT AS hc_in_last_5_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN INTERVAL 3 DAYS PRECEDING AND CURRENT ROW)::INT AS hc_in_last_4_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN INTERVAL 2 DAYS PRECEDING AND CURRENT ROW)::INT AS hc_in_last_3_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN INTERVAL 1 DAYS PRECEDING AND CURRENT ROW)::INT AS hc_in_last_2_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN CURRENT ROW AND INTERVAL 6 DAYS FOLLOWING)::INT AS hc_in_next_7_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN CURRENT ROW AND INTERVAL 5 DAYS FOLLOWING)::INT AS hc_in_next_6_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN CURRENT ROW AND INTERVAL 4 DAYS FOLLOWING)::INT AS hc_in_next_5_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN CURRENT ROW AND INTERVAL 3 DAYS FOLLOWING)::INT AS hc_in_next_4_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN CURRENT ROW AND INTERVAL 2 DAYS FOLLOWING)::INT AS hc_in_next_3_days,
        SUM(is_holiday) OVER (ORDER BY day RANGE BETWEEN CURRENT ROW AND INTERVAL 1 DAYS FOLLOWING)::INT AS hc_in_next_2_days,
        SIN(2*PI()*moy/12) AS month_sin,
        COS(2*PI()*moy/12) AS month_cos,
        SIN(2*PI()*dow/7) AS dow_sin,
        COS(2*PI()*dow/7) AS dow_cos,
        SIN(2*PI()*dom/total_days_in_month) AS dom_sin,
        COS(2*PI()*dom/total_days_in_month) AS dom_cos,
        SIN(2*PI()*woy/total_weeks_in_year) AS woy_sin,
        COS(2*PI()*woy/total_weeks_in_year) AS woy_cos,
        SIN(2*PI()*doy/total_days_in_year) AS doy_sin,
        COS(2*PI()*doy/total_days_in_year) AS doy_cos,
        CASE WHEN dow IN (6, 7) THEN 1 ELSE 0 END::INT AS is_weekend
        
    FROM (
        SELECT 
            DISTINCT {date_col}::DATE AS day,
            ISOYEAR({date_col}) AS year,
            DAYOFYEAR({date_col}) AS doy,
            WEEKOFYEAR({date_col}) AS woy,
            MONTH({date_col}) AS moy,
            ISODOW({date_col}) AS dow,
            DAY({date_col}) AS dom,
            DAY(LAST_DAY({date_col}::DATE)) AS total_days_in_month,
            -- december 28 guarantees last week of the year
            WEEKOFYEAR(MAKE_DATE(ISOYEAR({date_col}::DATE)::INT, 12, 28)) AS total_weeks_in_year,
            DAYOFYEAR(MAKE_DATE(YEAR({date_col}::DATE)::INT, 12, 31)) AS total_days_in_year,
            IS_HOLIDAY({date_col}::DATE) AS is_holiday
        FROM 
            df
    )
    """).pl()

    return (
        ddb.sql(f"""
        SELECT 
            t.*,
            SIN(2*PI()*HOUR(t.{date_col})/24) AS hour_sin,
            COS(2*PI()*HOUR(t.{date_col})/24) AS hour_cos,
            SIN(2*PI()*MINUTE(t.{date_col})/60) AS minute_sin,
            COS(2*PI()*MINUTE(t.{date_col})/60) AS minute_cos,
            d.* EXCLUDE(day)
        FROM 
            df t
        LEFT JOIN
            date_df d
        ON
            (t.{date_col}::DATE = d.day)
    """)
        .pl(lazy=True)
        .with_columns(pl.selectors.numeric().cast(pl.Float32))
    )
