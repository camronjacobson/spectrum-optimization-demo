# bea_mapper.py

import geopandas as gpd
import pandas as pd

def load_bea_shapes(geojson_path):
    """
    Load BEA polygons from a GeoJSON file.

    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file exported from QGIS.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    bea_shapes = gpd.read_file(geojson_path)
    return bea_shapes


def merge_bea_data(bea_shapes, csv_path, join_field="BEA_CODE"):
    """
    Merge BEA polygons with your custom CSV data.

    Parameters
    ----------
    bea_shapes : geopandas.GeoDataFrame
        GeoDataFrame of BEA polygons.
    csv_path : str
        Path to CSV with BEA data.
    join_field : str
        Column name to join on (e.g. "BEA_CODE").

    Returns
    -------
    geopandas.GeoDataFrame
        Merged data.
    """
    csv_data = pd.read_csv(csv_path)
    merged = bea_shapes.merge(csv_data, on=join_field, how="inner")
    return merged


def save_merged_geojson(merged_gdf, output_path):
    """
    Save merged GeoDataFrame as GeoJSON.

    Parameters
    ----------
    merged_gdf : geopandas.GeoDataFrame
    output_path : str
    """
    merged_gdf.to_file(output_path, driver="GeoJSON")


def save_merged_csv(merged_gdf, output_path):
    """
    Save merged GeoDataFrame as CSV with WKT geometry.

    Parameters
    ----------
    merged_gdf : geopandas.GeoDataFrame
    output_path : str
    """
    merged_gdf["geometry_wkt"] = merged_gdf["geometry"].apply(lambda g: g.wkt)
    merged_gdf.drop(columns="geometry").to_csv(output_path, index=False)

