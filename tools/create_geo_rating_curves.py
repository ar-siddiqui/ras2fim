import os
import argparse
import numpy as np
from datetime import datetime
import pandas as pd
import geopandas as gpd
import rasterio
import errno
from rasterio.features import shapes
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
VIZ_PROJECTION ='PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]'
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon


def produce_geo_rating_curves(feature_id, huc, rating_curve, depth_grid_list, version, output_folder):


    # Read rating curve for feature_id
    rating_curve_df = pd.read_csv(rating_curve)

    # Loop through depth grids and store up geometries to collate into a single rating curve.
    iteration = 0
    for depth_grid in depth_grid_list:
        # Open depth_grid using rasterio.
        depth_src = rasterio.open(depth_grid)
        depth_array = depth_src.read(1)

        # Use numpy.where operation to reclassify depth_array on the condition that the pixel values are > 0.
        reclass_depth_array = np.where((depth_array>0) & (depth_array != depth_src.nodata), 1, 0).astype('uint8')

        # Save resulting array to new tif with appropriate name.
        is_all_zero = np.all((reclass_depth_array == 0))

        # Only continue if there are actually pixel values above zero.
        if is_all_zero:
            continue

        # Interpolate flow from given stage.
        stage_ft = float(os.path.split(depth_grid)[1].split('-')[1].strip('.tif'))/10
        # interpolated_flow_cfs = np.interp(stage_ft,rating_curve_df.loc[:,'AvgDepth(ft)'],rating_curve_df.loc[:,'Flow(cfs)'])

        # Aggregate shapes
        results = ({'properties': {'extent': 1}, 'geometry': s} for i, (s, v) in enumerate(shapes(reclass_depth_array, mask=reclass_depth_array>0)))

        # Convert list of shapes to polygon, then dissolve
        extent_poly = gpd.GeoDataFrame.from_features(list(results), crs=VIZ_PROJECTION)
        extent_poly_diss = extent_poly.dissolve(by='extent')
        extent_poly_diss = extent_poly_diss.to_crs(VIZ_PROJECTION)
        extent_poly_diss["geometry"] = [MultiPolygon([feature]) if type(feature) == Polygon else feature for feature in extent_poly_diss["geometry"]]

        # -- Add more attributes -- #
        extent_poly_diss['version'] = version
        extent_poly_diss['feature_id'] = feature_id
        extent_poly_diss['stage_ft'] = stage_ft

        # TODO add remainder of ras2fim attributes from catchment layer

        if iteration < 1:  # Initialize the rolling huc_rating_curve_geo
            feature_id_rating_curve_geo = pd.merge(rating_curve_df, extent_poly_diss, left_on='AvgDepth(ft)', right_on='stage_ft', how='right')
        else:
            rating_curve_geo_df = pd.merge(rating_curve_df, extent_poly_diss, left_on='AvgDepth(ft)', right_on='stage_ft', how='right')
            feature_id_rating_curve_geo = pd.concat([feature_id_rating_curve_geo, rating_curve_geo_df])
        iteration += 1 

    feature_id_rating_curve_geo.to_csv(os.path.join(output_folder, feature_id + '_' + huc + '_rating_curve_geo.csv'))


def manage_geo_rating_curves_production(ras2fim_output_dir, version, job_number, output_folder, overwrite):
    overall_start_time = datetime.now()
    dt_string = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print (f"Started: {dt_string}")
        
    # Check job numbers and raise error if necessary
    total_cpus_available = os.cpu_count() - 1
    if job_number > total_cpus_available:
        raise ValueError('The job number, {}, '\
                          'exceeds your machine\'s available CPU count minus one ({}). '\
                          'Please lower the job_number.'.format(job_number, total_cpus_available))
    
    if not os.path.exists(ras2fim_output_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ras2fim_output_dir)
    hec_ras_output_path = os.path.join(ras2fim_output_dir, '05_hecras_output')  # Get path to step 5 outputs
    
    # Set up multiprocessing
    dictionary = {}
    local_dir_list = os.listdir(hec_ras_output_path)
    for huc in local_dir_list:
        full_huc_path = os.path.join(hec_ras_output_path, huc)
        if not os.path.isdir(full_huc_path):
            continue
        feature_id_list = os.listdir(full_huc_path)
        for feature_id in feature_id_list:
            full_feature_id_path = os.path.join(full_huc_path, feature_id)
            feature_id_list = os.listdir(full_huc_path)
            for feature_id in feature_id_list:
                full_feature_id_path = os.path.join(full_huc_path, feature_id)
                depth_grid_dir = os.path.join(full_feature_id_path, 'Depth_Grid')
                rating_curve_path = os.path.join(full_feature_id_path, 'Rating_Curve', feature_id + '_rating_curve.csv')
                try:
                    depth_grid_list = os.listdir(depth_grid_dir)
                except FileNotFoundError:
                    continue
                full_path_depth_grid_list = []
                for depth_grid in depth_grid_list:
                    full_path_depth_grid_list.append(os.path.join(depth_grid_dir, depth_grid))
                dictionary.update({feature_id: {'huc': huc, 'rating_curve': rating_curve_path, 'depth_grids': full_path_depth_grid_list}})
                
    print("Multiprocessing " + str(len(dictionary)) + " feature_ids using " + str(job_number) + " jobs...")
    with ProcessPoolExecutor(max_workers=job_number) as executor:
        for feature_id in dictionary:
            executor.submit(produce_geo_rating_curves, feature_id, dictionary[feature_id]['huc'], 
                            dictionary[feature_id]['rating_curve'], dictionary[feature_id]['depth_grids'], 
                            version, output_folder)
            
#            produce_geo_rating_curves(feature_id, dictionary[feature_id]['huc'], dictionary[feature_id]['rating_curve'], 
#                                      dictionary[feature_id]['depth_grids'], version, output_folder)
            
    # Calculate duration
    end_time = datetime.now()
    dt_string = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print (f"Ended: {dt_string}")
    time_duration = end_time - overall_start_time
    print(f"Duration: {str(time_duration).split('.')[0]}")
    print()


if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Produce Geo Rating Curves for RAS2FIM')
    parser.add_argument('-f', '--ras2fim_output_dir', help='Path to directory containing RAS2FIM outputs',
                        required=True)
    parser.add_argument('-v', '--version', help='RAS2FIM Version number',
                        required=True)
    parser.add_argument('-j','--job_number',help='Number of processes to use', required=False, default=1, type=int)
#    parser.add_argument('jf', '--job_number_feature_id',help='Number of concurrent feature_ids to process)
    parser.add_argument('-t', '--output_folder', help = 'Target: Where the output folder will be', required = False)
    parser.add_argument('-o','--overwrite', help='Overwrite files', required=False, action="store_true")
    
    args = vars(parser.parse_args())
    manage_geo_rating_curves_production(**args)