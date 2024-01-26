import argparse
import csv
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from timeit import default_timer as timer

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv


pd.options.mode.copy_on_write = True  # Enable copy-on-write


@dataclass
class SiteProcessingRecord:
    """Holds processing record for each site."""

    site: str
    start_time: datetime
    status: str = ""
    error: str = ""
    message: str = ""
    end_time: datetime = field(default_factory=datetime.now)

    def update_on_error(self, error_type: str, error_message: str):
        self.end_time = datetime.now()
        self.error = error_type
        self.status = "failed"
        self.message = error_message

    def update_on_success(self):
        self.end_time = datetime.now()
        self.status = "success"


def identify_best_branch_catchments(huc8_outputs_dir, subset_fim_gdf):
    # Open branch_polygons and check for overlap with subset_fim_gdf
    branch_polygons = os.path.join(huc8_outputs_dir, 'branch_polygons.gpkg')
    branch_polygons_gdf = gpd.read_file(branch_polygons)
    branch_polygons_gdf = branch_polygons_gdf.to_crs(subset_fim_gdf.crs)
    joined_gdf = branch_polygons_gdf.sjoin(subset_fim_gdf, how='left')
    not_null_rows = joined_gdf['USGSID'].notnull()
    subset_joined_gdf = joined_gdf[not_null_rows]
    branches_of_interest = list(subset_joined_gdf.levpa_id.unique())

    # Get path to branches directory and create paths to all branch catchment in list
    branch_path_list = []
    branches_dir = os.path.join(huc8_outputs_dir, 'branches')
    for branch in branches_of_interest:
        branch_catchments = os.path.join(
            branches_dir, branch, f'gw_catchments_reaches_filtered_addedAttributes_crosswalked_{branch}.gpkg'
        )
        branch_path_list.append(branch_catchments)

    del branch_polygons_gdf, joined_gdf, not_null_rows, subset_joined_gdf, branches_of_interest

    return branch_path_list


def get_union(catchments_gdf, subset_fim_gdf, site_stage):
    # Subset subset_fim_gdf to only stage of interes
    stage_subset_fim_gdf = subset_fim_gdf.loc[subset_fim_gdf.STAGE == site_stage]

    # Reproject fim_gdf to match NWM catchments
    stage_subset_fim_gdf = stage_subset_fim_gdf.to_crs(catchments_gdf.crs)

    # Dissolve all geometries?
    stage_subset_fim_gdf['dissolve'] = 1
    stage_subset_fim_gdf = stage_subset_fim_gdf.dissolve(by="dissolve")
    # Cut dissolved geometry to align with catchment breakpoints and associate feature_ids (union)
    union = gpd.overlay(stage_subset_fim_gdf, catchments_gdf)

    # # Drop unnecessary columns
    # columns_to_keep = ['fid', 'STAGE', 'ELEV', 'QCFS', 'feature_id', 'order_', 'discharge']
    # columns_to_drop = [col for col in union.columns if col not in columns_to_keep]
    # union.drop(columns=columns_to_drop, inplace=True)

    return union


def translate_site(
    site, geometry, usgs_rc_df, output_dir, usgs_gages_gdf, usgs_gdb, level_path_parent_dir, log_level, log_folder: str
) -> SiteProcessingRecord:
    start_time = datetime.now()

    logger = setup_logging(log_level, f'{log_folder}/{site}')
    processing_record = SiteProcessingRecord(site=site, start_time=start_time)
    try:
        try:
            int(site)
        except ValueError as e:
            logger.error(f"{site}: {str(e)}")
            processing_record.update_on_error("InvalidSiteName", str(e))
            return processing_record

        # Subset usgs_rc_df to only gage of interest
        site_usgs_rc_df = usgs_rc_df.loc[usgs_rc_df.location_id == int(site)]

        # Exit if site-specific rating curve doesn't exist in provided file
        if site_usgs_rc_df.empty:
            logger.error(f"{site}: Missing rating curve")
            processing_record.update_on_error("MissingRatingCurve", "")
            return processing_record

        # Create output directory site
        site_dir = os.path.join(output_dir, site)
        if not os.path.exists(site_dir):
            os.mkdir(site_dir)

        # Create directories for temporary files.
        branch_parent_dir = os.path.join(site_dir, 'branches')
        if not os.path.exists(branch_parent_dir):
            os.mkdir(branch_parent_dir)

        # Load USGS FIM Library geopackage
        logger.info("Loading USGS FIM library for site " + site + "...")
        usgs_lib_start = timer()
        usgs_fim_gdf = gpd.read_file(usgs_gdb, layer='fim_flood_extents', mask=geometry)
        logger.info(f"{site} loaded in {round((timer() - usgs_lib_start)/60, 2)} minutes.")

        # Determine HUC8  TODO would be faster if FIM library had HUC8 attribute
        try:
            huc12 = usgs_gages_gdf.loc[usgs_gages_gdf.SITE_NO == site].huc12.values[0]
            huc8 = huc12[:8]
        except IndexError as e:
            logger.error(f"{site}: {str(e)}")
            processing_record.update_on_error("CannotGetHUC", str(e))
            return processing_record

        # Subset the entire usgs_fim_gdf library to only one site at a time
        subset_fim_gdf = usgs_fim_gdf.loc[usgs_fim_gdf.USGSID == site]

        # Remove rows with missing geometry  TODO LOG
        subset_fim_gdf = subset_fim_gdf.loc[subset_fim_gdf.geometry != None]
        # Identify which level path is best for the site
        huc8_outputs_dir = os.path.join(level_path_parent_dir, huc8)
        if os.path.exists(huc8_outputs_dir):
            branch_path_list = identify_best_branch_catchments(huc8_outputs_dir, subset_fim_gdf)
        else:
            logger.error(f"{site}: Missing branch data")
            processing_record.update_on_error("MissingBranchData", f"Expected {huc8_outputs_dir} to exist")
            shutil.rmtree(site_dir)
            return processing_record

        # Get list of unique stage values
        site_stages = list(subset_fim_gdf.STAGE.unique())
        if len(site_stages) == 0:
            logger.error(f"{site}: No stage values")
            processing_record.update_on_error("NoStageValues", "")
            return processing_record

        # Loop through different catchments, do the below processing, then check for best geometrical match
        branch_id_list, candidate_layers = [], []
        catchments_path_list, feature_count_list = [], []

        # Identify the first map in the stack.
        first_site_stage = site_stages[0]

        # Loop through all relevant branches and perform unions
        for catchments in branch_path_list:
            branch_id = os.path.split(catchments)[1].split('_')[-1].replace('.gpkg', '')
            branch_id_list.append(branch_id)
            branch_output_dir = os.path.join(branch_parent_dir, branch_id)
            if not os.path.exists(branch_output_dir):
                os.mkdir(branch_output_dir)

            # Load catchment geopackage
            if os.path.exists(catchments):
                catchments_gdf = gpd.read_file(catchments)
            else:
                logger.error(f"{site}: Missing catchments {catchments}")
                processing_record.update_on_error("MissingCatchmentsData", f"Expected {catchments} to exist")
                return processing_record

            # Once the union with the highest count is known, perform union again with only that branch
            feature_count = len(get_union(catchments_gdf, subset_fim_gdf, first_site_stage))
            catchments_path_list.append(catchments)
            feature_count_list.append(feature_count)

            del catchments_gdf

        # Create new union with best catchment layer
        # Select best match of all the generated FIM/branch unions
        logger.info("Producing union for " + site + "...")
        if len(feature_count_list) == 0:
            logger.error(f"{site}: No feature count values")
            processing_record.update_on_error("NoFeatureCountValues", "")
            return processing_record

        max_index = feature_count_list.index(max(feature_count_list))
        best_match_catchments_gdf = gpd.read_file(catchments_path_list[max_index])

        # Merge all relevant unions for the "best match" union
        iteration = 1
        for site_stage in site_stages:
            if iteration == 1:
                union = get_union(best_match_catchments_gdf, subset_fim_gdf, site_stage)
                union['discharge_cfs'] = round(
                    (np.interp([site_stage], site_usgs_rc_df['stage'], site_usgs_rc_df['flow'])[0]), 3
                )
            else:
                union_to_append = get_union(best_match_catchments_gdf, subset_fim_gdf, site_stage)
                union_to_append['discharge_cfs'] = round(
                    (np.interp([site_stage], site_usgs_rc_df['stage'], site_usgs_rc_df['flow'])[0]), 3
                )

                union = pd.concat([union, union_to_append])

            iteration += 1
        # Save as geopackage (temp) 5793592_HUC_120903010404_rating_curve_geo.csv
        # TODO write individual geopackages in the same format as RAS2FIM (Ask Carson for latest released data)

        if union.empty == True:
            logger.error(f"{site}: Union empty")
            processing_record.update_on_error("UnionEmpty", "")
            return processing_record

        # Clean up geodataframe to match ras2fim schema
        union.rename(columns={'STAGE': 'stage_ft', 'ELEV': 'wse_ft', 'QCFS': 'orig_cfs'}, inplace=True)
        union['discharge_cms'] = round((union['discharge_cfs'] * 0.0283168), 3)
        union['stage_m'] = round(union['stage_ft'] * 0.3048, 3)  # Convert feet to meters
        union['version'] = 'usgs_fim'  # Assuming a constant value
        union['stage_mm'] = round(union['stage_m'] * 100, 3)
        union['wse_m'] = round(union['wse_ft'] * 0.3048, 3)
        union['valid'] = union.is_valid
        columns_to_keep = [
            'discharge_cms',
            'discharge_cfs',
            'stage_m',
            'version',
            'stage_ft',
            'wse_ft',
            'geometry',
            'feature_id',
            'stage_mm',
            'wse_m',
            'valid',
        ]
        union_subset = union[columns_to_keep]

        # union_subset.to_file(output_shapefile, driver='GPKG')

        # Convert to Web Mercator
        union_subset = union_subset.to_crs('EPSG:3857')
        # union_subset['geometry'] = union_subset['geometry'].simplify(1)

        final_geocurves_dir = os.path.join(output_dir, 'geocurves')
        final_geom_dir = os.path.join(output_dir, 'geocurve_polys')

        # Subset to each feature_id
        feature_id_list = list(union_subset.feature_id.unique())
        for feature_id_item in feature_id_list:
            feature_id_subset = union_subset.loc[union_subset.feature_id == feature_id_item]

            feature_id_subset['filename'] = feature_id_subset.apply(
                lambda row: f"{final_geom_dir}/{int(feature_id_item)}_HUC_{huc12}_{int(row['stage_mm'])}_mm.gpkg",
                axis=1,
            )

            # for idx, row in feature_id_subset.iterrows():
            #     # Construct the path string for each row using its specific 'stage_mm' value
            #     path_str = final_geom_dir + '/' + str(int(feature_id_item)) + '_HUC_' + huc12 + '_' + str(int(row['stage_mm'])) + '_mm.gpkg'

            #     # Directly assign the constructed string to the 'path' column for the current row
            #     feature_id_subset.loc[idx, 'path'] = path_str

            # Write polygons using path
            unique_path_list = list(feature_id_subset.filename.unique())
            for unique_path in unique_path_list:
                unique_path_subset = feature_id_subset.loc[feature_id_subset.filename == unique_path]
                unique_path_subset.to_file(unique_path)

            # Write final CSV for feature_id
            feature_id_output_csv = os.path.join(
                final_geocurves_dir, str(int(feature_id_item)) + '_HUC_' + huc12 + '_rating_curve_geo.csv'
            )
            feature_id_subset.to_csv(feature_id_output_csv)

        logger.info(f"{site} completed in {datetime.now() - start_time}")
        processing_record.update_on_success()
        return processing_record

    except Exception as e:
        logger.error(f"{site}: {str(e)}")
        processing_record.update_on_error("UnknownError", str(e))
        return processing_record


def load_datasets(usgs_rating_curves, usgs_gages_gpkg, usgs_gdb, logger):
    logger.info("Loading USGS rating curves...")
    usgs_rc_df = pd.read_csv(usgs_rating_curves)
    logger.info(f"Number of unique rating curve sites in the CSV: {usgs_rc_df['location_id'].nunique()}")

    logger.info("Loading USGS gages...")
    usgs_gages_gdf = gpd.read_file(usgs_gages_gpkg)
    logger.info(f"Number of usgs gages in GPKG: {usgs_gages_gdf.shape[0]}")

    logger.info("Loading FIM domains...")
    fim_domain_gdf_pre_diss = gpd.read_file(usgs_gdb, layer='fim_model_extent')
    fim_domain_gdf_pre_diss['usgs_id'] = fim_domain_gdf_pre_diss['USGSID']
    fim_domain_gdf = fim_domain_gdf_pre_diss.dissolve(by="usgs_id")
    fim_domain_gdf = fim_domain_gdf[['geometry', 'MULTI_SITE']]
    logger.info(f"Number of unique FIM domains: {fim_domain_gdf.shape[0]}")

    return usgs_rc_df, usgs_gages_gdf, fim_domain_gdf


def setup_directories(output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    final_geocurves_dir = os.path.join(output_dir, 'geocurves')
    final_geom_dir = os.path.join(output_dir, 'geocurve_polys')
    for dir in [final_geocurves_dir, final_geom_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    return


def setup_logging(log_level: int, name: str) -> logging.Logger:
    if not isinstance(log_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create a new logger for each site
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Configure file handler
    file_handler = logging.FileHandler(f'{name}.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    )

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prototype capability to reformat USGS inundation maps to geocurves.")
    parser.add_argument("-d", "--usgs_gdb", required=True, type=str, help="Path to USGS FIMs GDB.")
    parser.add_argument("-o", "--output_dir", required=True, type=str, help="Directory path for output geocurves.")
    parser.add_argument(
        "-c", "--level_path_parent_dir", required=True, type=str, help="Path to HAND FIM4 parent directory."
    )
    parser.add_argument("-rc", "--usgs_rating_curves", required=True, type=str, help="Path to rating curves CSV.")
    parser.add_argument("-g", "--usgs_gages_gpkg", required=True, type=str, help="Path to usgs_gages.gpkg.")
    parser.add_argument(
        "-pp", "--parallel_processes_count", default=None, type=int, help="Number of sites to process simultaneously."
    )
    parser.add_argument(
        "-ll", "--log_level", default="INFO", type=str, help="Set the logging level (e.g., INFO, DEBUG)."
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_arguments()

    run_time = datetime.now()
    run_time_str = run_time.strftime("%Y_%m_%d_%H_%M_%S")

    os.makedirs(run_time_str)
    log_level = getattr(logging, args.log_level.upper(), None)
    logger = setup_logging(log_level, f"{run_time_str}/main")

    setup_directories(args.output_dir)

    # Setup site processing records CSV
    with open(f'{run_time_str}/sites.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["site", "status", "error", "message", "start_time", "end_time"])

    # Read datasets and load in memory
    usgs_rc_df, usgs_gages_gdf, fim_domain_gdf = load_datasets(
        args.usgs_rating_curves, args.usgs_gages_gpkg, args.usgs_gdb, logger
    )
    logger.info(f"Datasets loaded in {datetime.now() - run_time}")

    # Run translation process for each FIM Domain
    logger.info(f"Executing individual sites...")
    with ProcessPoolExecutor(max_workers=args.parallel_processes_count) as executor:
        futures = [
            executor.submit(
                translate_site,
                index,
                row['geometry'],
                usgs_rc_df,
                args.output_dir,
                usgs_gages_gdf,
                args.usgs_gdb,
                args.level_path_parent_dir,
                log_level,
                run_time_str,
            )
            for index, row in fim_domain_gdf.iterrows()
        ]

    with open(f'{run_time_str}/sites.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for future in as_completed(futures):
            record = future.result()
            writer.writerow(
                [record.site, record.status, record.error, record.message, record.start_time, record.end_time]
            )

    logger.info(f"Completed in {datetime.now() - run_time}")


if __name__ == '__main__':
    main()
