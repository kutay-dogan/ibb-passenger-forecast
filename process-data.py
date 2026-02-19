import polars as pl
from stat_features import add_stat_features
from lag_features import add_lag_features


df = (
    pl.scan_parquet("data/hourly_transportation.parquet")
    .filter(pl.col("road_type") == "RAYLI")
    .drop(
        "road_type",
        "number_of_passenger",
        "town",
        "line",
        "product_kind",
        "transfer_type",
        "transport_type_id",
        "transaction_type_desc",
    )
    .rename({"station_poi_desc_cd": "station"})
)


df = df.with_columns(
    pl.when(pl.col("transition_hour") < 10)
    .then(pl.concat_str([pl.lit("0"), pl.col("transition_hour")]))
    .otherwise(pl.col("transition_hour"))
    .alias("transition_hour")
)

df = df.with_columns(
    pl.concat_str([pl.col("transition_hour"), pl.lit("-00")]).alias("transition_hour")
)


df = df.with_columns(
    pl.concat_str([pl.col("transition_date"), pl.col("transition_hour")], separator="-")
    .str.to_datetime(format="%Y-%m-%d-%H-%M")
    .alias("timestamp")
).drop(["transition_date", "transition_hour"])

dropped_lines = ["NOSTRAM", "T3", "KADIKOY-EMN", "34A"]

df = df.filter([pl.col("line_name") != line for line in dropped_lines])
df = df.filter(pl.col("station").is_not_null())

# filter m4 line only (baseline)

m4_mapping = {
    "KARTAL (BATI)": "KARTAL",
    "KARTAL (DOGU)": "KARTAL",
    "YAKACIK (DOGU)": "YAKACIK",
    "YAKACIK (BATI)": "YAKACIK",
    "HASTANE (DOGU/ADLIYE)": "HASTANE",
    "HASTANE (BATI)": "HASTANE",
    "TAVSANTEPE (BATI)": "TAVŞANTEPE",
    "TAVSANTEPE (DOGU)": "TAVŞANTEPE",
    "BOSTANCI (DOGU)": "BOSTANCI",
    "BOSTANCI (BATI)": "BOSTANCI",
    "PENDIK (BATI)": "PENDIK",
    "PENDIK (DOGU)": "PENDIK",
    "KADIKOY (BATI)": "KADIKOY",
    "KADIKOY (DOGU)": "KADIKOY",
    "ACIBADEM (DOGU)": "ACIBADEM",
    "ACIBADEM (BATI)": "ACIBADEM",
    "SABIHA GOKCEN HAVALIMANI": "SABIHA GOKCEN",
    "M4 KURTKOY": "KURTKOY",
}

df = df.filter(pl.col("line_name") == "M4").with_columns(
    pl.col("station").replace(m4_mapping).alias("station")
)
df = df.group_by("line_name", "station", "timestamp").agg(
    pl.sum("number_of_passage").alias("passage")
)


cat_cols = ["line_name", "station"]
date_col = "timestamp"
target_col = "passage"
lags = [30, 31, 32, 33, 35, 37, 40, 42, 49, 56, 63, 70]

df = add_stat_features(
    df,
    "passage",
    "timestamp",
    ["line_name", "station"],
    intervals=["1 day", "1 week", "1 month", "3 months"],
).collect()

df = add_lag_features(df, target_col, cat_cols, date_col, lags)
df.sink_parquet("data/Xy.parquet")
