import polars as pl
from utils.stat_features import add_stat_features
from utils.lag_features import add_lag_features

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

lines = ["M1", "M2", "M4", "T1", "MARMARAY"]

df = df.filter(pl.col("line_name").is_in(lines))
df = df.filter(pl.col("station").is_not_null())

df = df.with_columns(
    pl.col("station")
    .str.replace_many(
        [
            " (GUNEY)",
            " (KUZEY)",
            " (BATI)",
            " (DOGU)",
            " GUNEY",
            " KUZEY",
            " BATI",
            " DOGU",
        ],
        [""],
    )
    .alias("station")
)

mapping = {
    "HASTANE (DOGU/ADLIYE)": "HASTANE",
    "SABIHA GOKCEN HAVALIMANI": "SABIHA GOKCEN",
    "M4 KURTKOY": "KURTKOY",
    "SEYRANTEPE 3 STAD GIRISI": "SEYRANTEPE",
    "SEYRANTEPE 1": "SEYRANTEPE",
    "SEYRANTEPE 2": "SEYRANTEPE",
    "4 LEVENT 2": "4 LEVENT",
    "LEVENT 2": "LEVENT",
    "SISLI 2": "SISLI",
    "OSMANBEY 2": "OSMANBEY",
    "OTOGAR 1": "OTOGAR",
    "AKSARAY 1": "AKSARAY",
    "SIRKECI-4": "SIRKECI",
    "SIRKECI-3": "SIRKECI",
    "SIRKECI-2": "SIRKECI",
    "SIRKECI-1": "SIRKECI",
    "BAKIRKOY-1": "BAKIRKOY",
    "BAKIRKOY-2": "BAKIRKOY",
    "USKUDAR-1": "USKUDAR",
    "USKUDAR-2": "USKUDAR",
    "USKUDAR-3": "USKUDAR",
    "GEBZE-1": "GEBZE",
    "GEBZE-2": "GEBZE",
    "KUCUKYALI-1": "KUCUKYALI",
    "KUCUKYALI-2": "KUCUKYALI",
    "YENIKAPI-1": "YENIKAPI",
    "YENIKAPI-2": "YENIKAPI",
    "YENIKAPI-3": "YENIKAPI",
    "TERSANE-1": "TERSANE",
    "TERSANE-2": "TERSANE",
    "BOSTANCI-1": "BOSTANCI",
    "BOSTANCI-2": "BOSTANCI",
    "CEVIZLI-1": "CEVIZLI",
    "CEVIZLI-2": "CEVIZLI",
    "EMINONU 2": "EMINONU",
    "ZEYTINBURNU 2": "ZEYTINBURNU",
    "KABATAS 2": "KABATAS",
}

df = df.with_columns(pl.col("station").replace(mapping).alias("station"))
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
