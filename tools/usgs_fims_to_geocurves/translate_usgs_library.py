import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from timeit import default_timer as timer

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv


logging.captureWarnings(True)
pd.options.mode.copy_on_write = True  # Enable copy-on-write


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


def translate_site(site, geometry, usgs_rc_df, output_dir, usgs_gages_gdf, usgs_gdb, level_path_parent_dir):
    site_start = timer()

    # Subset usgs_rc_df to only gage of interest
    site_usgs_rc_df = usgs_rc_df.loc[usgs_rc_df.location_id == int(site)]

    # Exit if site-specific rating curve doesn't exist in provided file
    if site_usgs_rc_df.empty:
        print("Missing RC for " + site)
        return

    try:
        int(site)
    except ValueError as e:
        logging.error(e)
        return

    # Create output directory site
    site_dir = os.path.join(output_dir, site)
    if not os.path.exists(site_dir):
        os.mkdir(site_dir)

    # Create directories for temporary files.
    branch_parent_dir = os.path.join(site_dir, 'branches')
    if not os.path.exists(branch_parent_dir):
        os.mkdir(branch_parent_dir)

    # Create final directory.
    final_dir = os.path.join(site_dir, 'final')
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)

    # Load USGS FIM Library geopackage
    logging.info("Loading USGS FIM library for site " + site + "...")
    usgs_lib_start = timer()
    usgs_fim_gdf = gpd.read_file(usgs_gdb, layer='fim_flood_extents', mask=geometry)
    print(site + f" loaded in {round((timer() - usgs_lib_start)/60, 2)} minutes.")

    # Determine HUC8  TODO would be faster if FIM library had HUC8 attribute
    try:
        huc12 = usgs_gages_gdf.loc[usgs_gages_gdf.SITE_NO == site].huc12.values[0]
        huc8 = huc12[:8]
    except IndexError as e:
        logging.error(e)
        return  # TODO log, why?

    # Subset the entire usgs_fim_gdf library to only one site at a time
    subset_fim_gdf = usgs_fim_gdf.loc[usgs_fim_gdf.USGSID == site]

    # Remove rows with missing geometry  TODO LOG
    subset_fim_gdf = subset_fim_gdf.loc[subset_fim_gdf.geometry != None]
    # Identify which level path is best for the site
    huc8_outputs_dir = os.path.join(level_path_parent_dir, huc8)
    if os.path.exists(huc8_outputs_dir):
        branch_path_list = identify_best_branch_catchments(huc8_outputs_dir, subset_fim_gdf)
    else:
        print("Missing branch data, expected: " + huc8_outputs_dir)
        shutil.rmtree(site_dir)
        return

    # Get list of unique stage values
    site_stages = list(subset_fim_gdf.STAGE.unique())
    if len(site_stages) == 0:
        return

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
            return  # TODO log

        # Once the union with the highest count is known, perform union again with only that branch
        feature_count = len(get_union(catchments_gdf, subset_fim_gdf, first_site_stage))
        catchments_path_list.append(catchments)
        feature_count_list.append(feature_count)

        del catchments_gdf

    # Create new union with best catchment layer
    # Select best match of all the generated FIM/branch unions
    print("Producing union for " + site + "...")
    if len(feature_count_list) == 0:
        return
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
        print("empty")
        return

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

    # output_shapefile = os.path.join(final_dir, str(site) + '_' + branch_id + '.gpkg')
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
            lambda row: f"{final_geom_dir}/{int(feature_id_item)}_HUC_{huc12}_{int(row['stage_mm'])}_mm.gpkg", axis=1
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

    # del union

    print(site + f" completed in {round((timer() - site_start)/60, 2)} minutes.")
    print()


def load_datasets(usgs_rating_curves, usgs_gages_gpkg, usgs_gdb):
    logging.info("Loading USGS rating curves...")
    usgs_rc_df = pd.read_csv(usgs_rating_curves)
    logging.info(f"Number of unique rating curve sites in the CSV: {usgs_rc_df['location_id'].nunique()}")

    logging.info("Loading USGS gages...")
    usgs_gages_gdf = gpd.read_file(usgs_gages_gpkg)
    logging.info(f"Number of usgs gages in GPKG: {usgs_gages_gdf.shape[0]}")

    logging.info("Loading FIM domains...")
    fim_domain_gdf_pre_diss = gpd.read_file(usgs_gdb, layer='fim_model_extent')
    fim_domain_gdf_pre_diss['usgs_id'] = fim_domain_gdf_pre_diss['USGSID']
    fim_domain_gdf = fim_domain_gdf_pre_diss.dissolve(by="usgs_id")
    fim_domain_gdf = fim_domain_gdf[['geometry', 'MULTI_SITE']]
    logging.info(f"Number of unique FIM domains: {fim_domain_gdf.shape[0]}")

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
    parser.add_argument("-j", "--job_number", required=True, type=int, help="Number of jobs to use.")
    parser.add_argument(
        "-ll", "--log_level", default="INFO", type=str, help="Set the logging level (e.g., INFO, DEBUG)."
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_arguments()

    # Set up logging
    log_level_int = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level_int, int):
        raise ValueError(f'Invalid log level: { args.log_level}')
    logging.basicConfig(
        filename=f'usgs_fim_to_geocurves.log',
        level=log_level_int,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Setup directories
    setup_directories(args.output_dir)

    # Read datasets and load in memory
    start = timer()
    usgs_rc_df, usgs_gages_gdf, fim_domain_gdf = load_datasets(
        args.usgs_rating_curves, args.usgs_gages_gpkg, args.usgs_gdb
    )
    logging.info(f"Datasets loaded in {round((timer() - start)/60, 2)} minutes.")

    # Run translation process for each FIM Domain
    if args.job_number > 1:
        with ProcessPoolExecutor(max_workers=args.job_number) as executor:
            for _, row in fim_domain_gdf.iterrows():
                executor.submit(
                    translate_site,
                    index,
                    row['geometry'],
                    usgs_rc_df,
                    args.output_dir,
                    usgs_gages_gdf,
                    args.usgs_gdb,
                    args.level_path_parent_dir,
                )
    else:
        for index, row in fim_domain_gdf.iterrows():
            translate_site(
                index,
                row['geometry'],
                usgs_rc_df,
                args.output_dir,
                usgs_gages_gdf,
                args.usgs_gdb,
                args.level_path_parent_dir,
            )
            exit()
    logging.info(f"Completed in {round((timer() - start)/60, 2)} minutes.")


if __name__ == '__main__':
    main()
