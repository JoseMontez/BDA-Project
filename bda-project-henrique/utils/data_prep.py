import geopandas as gpd
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def load_and_reproject(filepath: str, crs: int = 3763) -> gpd.GeoDataFrame:
    """
    Loads a polygon file and reprojects it to a given CRS.

    Parameters:
        filepath : path to the polygon file (.geojson, .shp, etc.)
        crs      : EPSG code for the target CRS (default: 3763 = Portugal TM06)
                   A projected CRS is required for accurate area calculations.

    Returns:
        GeoDataFrame reprojected to the target CRS.

    NOTE: GeoPandas is not big-data-safe. Acceptable here because polygon 
    files are small static lookups (~300 municipalities, ~2800 postal codes).
    """
    gdf = gpd.read_file(filepath)
    return gdf.to_crs(epsg=crs)


def compute_intersection_weights(
    postal_gdf: gpd.GeoDataFrame,
    municipality_gdf: gpd.GeoDataFrame,
    postal_col: str = "CP4",
    municipality_col: str = "Concelho"
) -> pd.DataFrame:
    """
    Computes the fractional area contribution of each municipality 
    to each postal code polygon via spatial intersection.

    For each (postal_code, municipality) pair, the weight represents:
        weight = intersection_area / total_postal_code_area

    Weights for a given postal code sum to 1.0, allowing proportional
    distribution of any postal-code-level metric (e.g. energy consumption)
    across municipalities.
    """
    # 1. Compute postal code areas on a isolated copy
    postal_base = postal_gdf.copy()
    postal_base["postal_area"] = postal_base.geometry.area

    # 2. Spatial intersection (keeps only Polygon/MultiPolygon results)
    intersection = gpd.overlay(postal_gdf, municipality_gdf, how="intersection", keep_geom_type=True)
    intersection["intersection_area"] = intersection.geometry.area

    # 3. Explicitly merge the baseline 'postal_area' back in using the identifier column
    intersection = intersection.merge(
        postal_base[[postal_col, "postal_area"]],
        on=postal_col,
        how="left"
    )

    # 4. Compute fractional weight safely
    intersection["weight"] = intersection["intersection_area"] / intersection["postal_area"]

    weights_df = intersection[[postal_col, municipality_col, "weight"]].copy()

    # Sanity check
    weight_sums = weights_df.groupby(postal_col)["weight"].sum()
    bad = weight_sums[abs(weight_sums - 1.0) > 0.01]
    if not bad.empty:
        print(f" Warning: {len(bad)} postal codes have weights that don't sum to 1.0")
        print(bad)

    return weights_df



def distribute_consumption(
    eredes_df: DataFrame,
    weights_df: DataFrame,
    consumption_col: str = "consumption_kwh",
    postal_col: str = "zip_code",
    weight_postal_col: str = "CodigoPostal",
    municipality_col: str = "Concelho"
) -> DataFrame:
    """
    Distributes postal-code-level energy consumption across municipalities
    using precomputed spatial intersection weights.

    For each hourly record:
        municipality_consumption = consumption_kwh * weight

    This is big-data-safe — it is a join + multiplication applied
    row-wise across distributed partitions with no sorting or collecting.

    Parameters:
        eredes_df          : Spark DataFrame with hourly consumption per postal code
        weights_df         : Spark DataFrame with (postal_code, municipality, weight)
        consumption_col    : column name for consumption values
        postal_col         : postal code column in eredes_df
        weight_postal_col  : postal code column in weights_df
        municipality_col   : municipality column in weights_df

    Returns:
        Spark DataFrame with consumption distributed across municipalities.
    """
    distributed = eredes_df.join(
        weights_df,
        on=eredes_df[postal_col] == weights_df[weight_postal_col],
        how="left"
    ).drop(weights_df[weight_postal_col]) \
     .withColumn(
        f"municipality_{consumption_col}",
        F.col(consumption_col) * F.col("weight")
    )
    
    
    # One job calculate both totals 
    totals = eredes_df.agg(F.sum(consumption_col).alias("original")) \
    .crossJoin(
        distributed.agg(F.sum(f"municipality_{consumption_col}").alias("distributed"))
    ).first()

    diff_pct = abs(totals["original"] - totals["distributed"]) / totals["original"] * 100
    print(f"Original total consumption:    {totals.original:,.1f} kWh")
    print(f"Distributed total consumption: {totals.distributed:,.1f} kWh")
    print(f"Difference: {diff_pct:.4f}%  ({'OK' if diff_pct < 0.1 else ' Check weights'})")
    return distributed