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


# ============================================================================
# Custom Transformers
# Must be defined BEFORE build_pipeline() which references them
# ============================================================================

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
    """
    Adds lag and rolling average features using Window functions,
    partitioned by municipality.

    context_df is passed as raw data (same format as the input to the
    full pipeline). We apply TemporalFeatureTransformer and
    HolidayFlagTransformer to it internally before the union so that
    both DataFrames have matching schemas.
    """

    def __init__(self, context_df=None):
        super().__init__()
        self.context_df = context_df

    def _transform(self, df: DataFrame) -> DataFrame:

        if self.context_df is not None:
            # ---------------------------------------------------------------------------
            # Apply the same stateless transformers to context_df so its schema
            # matches df (which has already passed through Temporal + Holiday stages).
            # This is the fix for NUM_COLUMNS_MISMATCH — context_df enters the pipeline
            # as raw data but df is already feature-enriched at this stage.
            # ---------------------------------------------------------------------------
            context_enriched = TemporalFeatureTransformer()._transform(
                                    HolidayFlagTransformer()._transform(
                                        self.context_df
                                    )
                               )
            context_tagged = context_enriched.withColumn("_is_context", F.lit(True))
            df_tagged      = df.withColumn("_is_context", F.lit(False))
            working_df     = context_tagged.union(df_tagged)
        else:
            working_df = df.withColumn("_is_context", F.lit(False))

        time_window        = Window.partitionBy("municipality").orderBy("datetime")
        rolling_window_24h = time_window.rowsBetween(-24, -1)  # exclude current row
        rolling_window_7d  = time_window.rowsBetween(-168, -1) # exclude current row

        result = working_df \
            .withColumn("lag_168h",
                F.lag("total_active_energy_kwh", 24).over(time_window)) 

        return result \
            .filter(F.col("_is_context") == False) \
            .drop("_is_context")


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

