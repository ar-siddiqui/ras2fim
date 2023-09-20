import argparse
import errno
import os
import shutil
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon

import shared_variables as sv
from shared_functions import get_changelog_version, progress_bar_handler


warnings.simplefilter(action="ignore", category=FutureWarning)


# -------------------------------------------------
def produce_geocurves(feature_id, huc, rating_curve, depth_grid_list, version, geocurves_dir, polys_dir):
    """
    For a single feature_id, the function produces a version of a RAS2FIM rating curve
    which includes the geometry of the extent for each stage/flow.

    Args:
        feature_id (str): National Water Model feature_id.
        huc (str): Derived from the directory names of the 05_hec_ras outputs which are organized by HUC12.
        rating_curve(str): The path to the feature_id-specific rating curve generated by RAS2FIM.
        depth_grid_list (list): A list of paths to the depth grids generated by RAS2FIM for the feature_id.
        version (str): The version number.
        output_folder (str): Path to output folder where geo version of rating curve will be written.
        polys_dir (str or Nonetype): Can be a path to a folder where polygons will be written, or None.

    """
    try:
        depth_grid = ""

        # Read rating curve for feature_id
        rating_curve_df = pd.read_csv(rating_curve)

        proj_crs = pyproj.CRS.from_string(sv.DEFAULT_RASTER_OUTPUT_CRS)

        # Loop through depth grids and store up geometries to collate into a single rating curve.
        iteration = 0
        for depth_grid in depth_grid_list:
            # Interpolate flow from given stage.
            stage_mm = float(os.path.split(depth_grid)[1].split("-")[1].strip(".tif"))

            with rasterio.open(depth_grid) as src:
                # Open inundation_raster using rasterio.
                image = src.read(1)

                # Use numpy.where operation to reclassify depth_array on the condition
                # that the pixel values are > 0.
                reclass_inundation_array = np.where((image > 0) & (image != src.nodata), 1, 0).astype("uint8")

                # if the array only has values of zero, then skip it (aka.. no heights above surface)
                if np.min(reclass_inundation_array) == 0 and np.max(reclass_inundation_array) == 0:
                    continue

                # Aggregate shapes
                results = (
                    {"properties": {"extent": 1}, "geometry": s}
                    for i, (s, v) in enumerate(
                        shapes(
                            reclass_inundation_array,
                            mask=reclass_inundation_array > 0,
                            transform=src.transform,
                        )
                    )
                )

                l_results = list(results)

                # Convert list of shapes to polygon, then dissolve
                extent_poly = gpd.GeoDataFrame.from_features(l_results, crs=proj_crs)

                try:
                    extent_poly_diss = extent_poly.dissolve(by="extent")
                    # extent_poly_diss = extent_poly.dissolve()
                    extent_poly_diss["geometry"] = [
                        MultiPolygon([feature]) if type(feature) == Polygon else feature
                        for feature in extent_poly_diss["geometry"]
                    ]

                except AttributeError as ae:
                    # TODO why does this happen? I suspect bad geometry. Small extent?
                    # TODO: We should log this when the logging system comes online.
                    print("^^^^^^^^^^^^^^^^^^")
                    print("Error...")
                    print(f"  huc is {huc}; feature_id = {feature_id}; depth_grid is {depth_grid}")
                    print(f"  Details: {ae}")
                    print("^^^^^^^^^^^^^^^^^^")
                    continue

                # -- Add more attributes -- #
                extent_poly_diss["version"] = version
                extent_poly_diss["feature_id"] = feature_id
                extent_poly_diss["stage_mm_join"] = stage_mm
                if polys_dir is not None:
                    inundation_polygon_path = os.path.join(
                        polys_dir, feature_id + "_" + huc + "_" + str(int(stage_mm)) + "_mm" + ".gpkg"
                    )
                    extent_poly_diss["filename"] = os.path.split(inundation_polygon_path)[1]

                if iteration < 1:  # Initialize the rolling huc_rating_curve_geo
                    feature_id_rating_curve_geo = pd.merge(
                        rating_curve_df,
                        extent_poly_diss,
                        left_on="stage_mm",
                        right_on="stage_mm_join",
                        how="right",
                    )
                else:
                    rating_curve_geo_df = pd.merge(
                        rating_curve_df,
                        extent_poly_diss,
                        left_on="stage_mm",
                        right_on="stage_mm_join",
                        how="right",
                    )
                    feature_id_rating_curve_geo = pd.concat(
                        [feature_id_rating_curve_geo, rating_curve_geo_df]
                    )

                # Produce polygon version of flood extent if directed by user
                if polys_dir is not None:
                    extent_poly_diss["stage_m"] = stage_mm / 1000.0
                    extent_poly_diss = extent_poly_diss.drop(columns=["stage_mm_join"])
                    extent_poly_diss["version"] = version
                    extent_poly_diss.to_file(inundation_polygon_path, driver="GPKG")

                iteration += 1

        if feature_id_rating_curve_geo is not None:
            feature_id_rating_curve_geo.to_csv(
                os.path.join(geocurves_dir, feature_id + "_" + huc + "_rating_curve_geo.csv")
            )

    except Exception as ex:
        print("*******************")
        print("Error:")
        print(f"  huc is {huc}; feature_id = {feature_id}; depth_grid is {depth_grid}")
        errMsg = str(ex) + " \n   " + traceback.format_exc()
        print(errMsg)


# -------------------------------------------------
def manage_geo_rating_curves_production(
    ras2fim_output_dir, version, job_number, output_folder, overwrite, produce_polys
):
    """
    This function sets up the multiprocessed generation of geo version of feature_id-specific rating curves.

    Args:
        ras2fim_output_dir (str): Path to top-level directory storing RAS2FIM outputs for a given run.
        version (str): Version number for RAS2FIM version that produced outputs.
        job_number (int): The number of jobs to use when parallel processing feature_ids.
        output_folder (str): The path to the output folder where geo rating curves will be written.
    """
    print()
    overall_start_time = datetime.now()
    dt_string = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print(f"Started: {dt_string}")

    # Check job numbers and raise error if necessary
    total_cpus_available = os.cpu_count() - 1
    if job_number > total_cpus_available:
        raise ValueError(
            "The job number, {}, "
            "exceeds your machine's available CPU count minus one ({}). "
            "Please lower the job_number.".format(job_number, total_cpus_available)
        )

    # Set up output folders. (final outputs folder now created early in the ras2fim.py lifecycle)
    if not os.path.exists(ras2fim_output_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ras2fim_output_dir)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Make geocurves_dir and polys dir
    geocurve_dirs = []
    geocurves_dir = os.path.join(output_folder, "geocurves")
    geocurve_dirs.append(geocurves_dir)

    polys_dir = os.path.join(output_folder, "polys")
    geocurve_dirs.append(polys_dir)

    for gc_dir in geocurve_dirs:
        if os.path.exists(gc_dir) and not overwrite:
            print(
                "The output directory, "
                + gc_dir
                + ", already exists. Use the overwrite flag (-o) to overwrite."
            )
            quit()

        if os.path.exists(gc_dir):
            shutil.rmtree(gc_dir)

    # Either way.. we are makign a new geocurve folder. e.g. If it is overwrite, we deleted
    #  before replacing it so we don't have left over garbage
    os.makedirs(geocurves_dir)

    if produce_polys:
        os.makedirs(polys_dir)
    else:
        polys_dir = None

    # Check version arg input.
    if os.path.isfile(version):
        version = get_changelog_version(version)
        print("Version found: " + version)

    # Define paths outputs
    simplified_depth_grid_parent_dir = os.path.join(
        ras2fim_output_dir, sv.R2F_OUTPUT_DIR_METRIC, sv.R2F_OUTPUT_DIR_SIMPLIFIED_GRIDS
    )
    rating_curve_parent_dir = os.path.join(
        ras2fim_output_dir, sv.R2F_OUTPUT_DIR_METRIC, sv.R2F_OUTPUT_DIR_METRIC_RATING_CURVES
    )

    # Create dictionary of files to process
    proc_dictionary = {}
    local_dir_list = os.listdir(simplified_depth_grid_parent_dir)
    for huc in local_dir_list:
        full_huc_path = os.path.join(simplified_depth_grid_parent_dir, huc)
        if not os.path.isdir(full_huc_path):
            continue
        feature_id_list = os.listdir(full_huc_path)
        for feature_id in feature_id_list:
            feature_id_depth_grid_dir = os.path.join(simplified_depth_grid_parent_dir, huc, feature_id)
            feature_id_rating_curve_path = os.path.join(
                rating_curve_parent_dir, huc, feature_id, feature_id + "_rating_curve.csv"
            )
            try:
                depth_grid_list = os.listdir(feature_id_depth_grid_dir)
            except FileNotFoundError:
                continue
            full_path_depth_grid_list = []
            for depth_grid in depth_grid_list:
                # filter out any files that is not a tif (ie.. tif.aux.xml (from QGIS))
                if depth_grid.endswith(".tif"):
                    full_path_depth_grid_list.append(os.path.join(feature_id_depth_grid_dir, depth_grid))
            proc_dictionary.update(
                {
                    feature_id: {
                        "huc": huc,
                        "rating_curve": feature_id_rating_curve_path,
                        "depth_grids": full_path_depth_grid_list,
                    }
                }
            )

    with ProcessPoolExecutor(max_workers=job_number) as executor:
        executor_dict = {}

        for feature_id in proc_dictionary:
            produce_gc_args = {
                "feature_id": feature_id,
                "huc": proc_dictionary[feature_id]["huc"],
                "rating_curve": proc_dictionary[feature_id]["rating_curve"],
                "depth_grid_list": proc_dictionary[feature_id]["depth_grids"],
                "version": version,
                "geocurves_dir": geocurves_dir,
                "polys_dir": polys_dir,
            }

            try:
                future = executor.submit(produce_geocurves, **produce_gc_args)
                executor_dict[future] = feature_id
            except Exception as ex:
                print(f"*** {ex}")
                traceback.print_exc()
                sys.exit(1)

        # Send the executor to the progress bar and wait for all MS tasks to finish
        progress_bar_handler(executor_dict, True, f"Creating geocurves with {job_number} workers")

    # Calculate duration
    print()
    end_time = datetime.now()
    dt_string = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print(f"Ended: {dt_string}")
    time_duration = end_time - overall_start_time
    print(f"Duration: {str(time_duration).split('.')[0]}")
    print()


# -------------------------------------------------
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Produce Geo Rating Curves for RAS2FIM")
    parser.add_argument(
        "-f", "--ras2fim_output_dir", help="Path to directory containing RAS2FIM outputs", required=True
    )
    parser.add_argument(
        "-v",
        "--version",
        help="RAS2FIM Version number, or supply path to repo Changelog",
        required=False,
        default="Unspecified",
    )
    parser.add_argument(
        "-j", "--job_number", help="Number of processes to use", required=False, default=1, type=int
    )
    parser.add_argument(
        "-t", "--output_folder", help="Target: Where the output folder will be", required=True
    )
    parser.add_argument("-o", "--overwrite", help="Overwrite files", required=False, action="store_true")
    parser.add_argument(
        "-p",
        "--produce_polys",
        help="Produce polygons in addition to geocurves.",
        required=False,
        default=False,
        action="store_true",
    )

    args = vars(parser.parse_args())
    manage_geo_rating_curves_production(**args)
