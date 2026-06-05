# utils/pipeline.py

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler,
    StringIndexer, OneHotEncoder
)
from pyspark.ml.pipeline import Transformer
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window


# Custom Transformers
# Must be defined BEFORE build_pipeline() which references them


class TemporalFeatureTransformer(Transformer):
    """
    Extracts time-based features from a timestamp column.
    Stateless — safe for both batch and streaming.
    """

    def _transform(self, df: DataFrame) -> DataFrame:
        return df \
            .withColumn("hour_of_day",  F.hour("datetime")) \
            .withColumn("day_of_week",  F.dayofweek("datetime")) \
            .withColumn("is_weekend",   F.when(F.dayofweek("datetime").isin([1, 7]), 1).otherwise(0)) \
            .withColumn("month",        F.month("datetime")) \
            .withColumn("day_of_month", F.dayofmonth("datetime"))


class HolidayFlagTransformer(Transformer):
    """
    Adds a binary is_holiday flag for Portuguese public holidays.
    Stateless — safe for both batch and streaming.

    NOTE: Holiday list is hardcoded for the observed date range.
    In production this would be loaded from a holiday calendar lookup
    table and joined, rather than hardcoded.
    """

    HOLIDAYS = [
        "2022-11-01", "2022-12-01", "2022-12-08", "2022-12-25",
        "2023-01-01", "2023-04-07", "2023-04-09", "2023-04-25",
        "2023-05-01", "2023-06-08", "2023-06-10", "2023-08-15",
        "2023-10-05", "2023-11-01",
    ]

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "is_holiday",
            F.when(
                F.date_format("datetime", "yyyy-MM-dd").isin(self.HOLIDAYS), 1
            ).otherwise(0)
        )


class LagFeatureTransformer(Transformer):
    def _transform(self, df):
        window = Window.partitionBy("municipality").orderBy("datetime")

        return (
            df
            # lag_168h — same hour exactly one week ago (168 hourly steps back)
            .withColumn("lag_168h", F.lag("total_active_energy_kwh", 168).over(window))
            # lag_24h — same hour yesterday (24 hourly steps back)
            .withColumn("lag_24h",  F.lag("total_active_energy_kwh", 24).over(window))
            # lag_1h — previous hour
        )
    
class NullLagDropTransformer(Transformer):
    """
    Drops rows where lag features are null (cold-start period).

    Only used during training — never in streaming inference where
    incoming records cannot be dropped.
    """

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.filter(
            F.col("lag_168h").isNotNull()
        )

