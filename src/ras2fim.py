# This is the main orchistration script for "ras2fim".  It attempts to convert
# geospatial 1D HEC-RAS models into a set of flood innundation mapping (fim)
# library of rasters with a cooresponding syntetic rating curve for a
# correponding National Water Model (NWM) reach segment.
#
# This script needs seven (7) other scripts to complete the process
# [create_shapes_from_hecras, conflate_hecras_to_nwm, get_usgs_dem_from_shape,
# convert_tif_to_ras_hdf5, create_fim_rasters, worker_fim_raster,
# simplify_fim_rasters]
#
# This is built to run on a Windows machine and requires that HEC-RAS v6.0
# be installed prior to execution.
#
# Created by: Andy Carter, PE
# Last revised - 2021.09.09
#
# Main code for ras2fim
# Uses the 'ras2fim' conda environment


from create_shapes_from_hecras import fn_create_shapes_from_hecras
from conflate_hecras_to_nwm import fn_conflate_hecras_to_nwm
from get_usgs_dem_from_shape import fn_get_usgs_dem_from_shape
from convert_tif_to_ras_hdf5 import fn_convert_tif_to_ras_hdf5
from create_fim_rasters import fn_create_fim_rasters
from simplify_fim_rasters import fn_simplify_fim_rasters

import argparse
import os

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def fn_run_ras2fim(str_huc8_arg,
                   str_ras_path_arg,
                   str_out_arg,
                   str_crs_arg,
                   b_is_feet,
                   str_nation_arg,
                   str_hec_path):
    
    print(" ")
    print("+=================================================================+")
    print("|          RUN RAS2FIM FOR A HEC-RAS 1-D DATASET (HUC8)           |")
    print("|     Created by Andy Carter, PE of the National Water Center     |")
    print("+-----------------------------------------------------------------+")
    
    print("  ---(r) HUC 8 WATERSHED: " + str(str_huc8_arg))
    print("  ---(i) PATH TO HEC-RAS: " + str(str_ras_path_arg))
    print("  ---(o) OUTPUT DIRECTORY: " + str(str_out_arg))
    print("  ---(p) PROJECTION OF HEC-RAS MODELS: " + str(str_crs_arg))
    print("  ---(v) VERTICAL IN FEET: " + str(b_is_feet))    
    print("  ---(n) PATH TO NATIONAL DATASETS: " + str(str_nation_arg))     
    print("  ---(r) PATH TO HEC-RAS v6.0: " + str(str_hec_path))
    
    print("===================================================================")
    print(" ")
    
    # create an output folder
    # TODO - 2021.09.07
    # check if this folder exists and has HEC-RAS files
    # Do you want to overwrite the previous output if exists?
    
    if not os.path.exists(str_out_arg):
        os.mkdir(str_out_arg)
    
    # ---- Step 1: create_shapes_from_hecras ----
    # create a folder for the shapefiles from hec-ras
    str_shapes_from_hecras_dir = os.path.join(str_out_arg, "shapes_from_hecras") 
    if not os.path.exists(str_shapes_from_hecras_dir):
        os.mkdir(str_shapes_from_hecras_dir)
    
    # run the first script (create_shapes_from_hecras)
    fn_create_shapes_from_hecras(str_ras_path_arg,
                                 str_shapes_from_hecras_dir,
                                 str_crs_arg)
    # -------------------------------------------

    # ------ Step 2: conflate_hecras_to_nwm -----    
    # do whatever is needed to create folders and determine variables
    str_shapes_from_conflation_dir = os.path.join(str_out_arg, "shapes_from_conflation")
    if not os.path.exists(str_shapes_from_conflation_dir):
        os.mkdir(str_shapes_from_conflation_dir)
    
    # run the second script (conflate_hecras_to_nwm)
    fn_conflate_hecras_to_nwm(str_huc8_arg, 
                              str_shapes_from_hecras_dir, 
                              str_shapes_from_conflation_dir,
                              str_nation_arg)
    # -------------------------------------------

    # ------ Step 3: get_usgs_dem_from_shape ----    
    # create folder

    str_area_shp_name = str_huc8_arg + "_huc_12_ar.shp"
    str_input_path = os.path.join(str_shapes_from_conflation_dir, str_area_shp_name)
    
    # create output folder
    str_terrain_from_usgs_dir = os.path.join(str_out_arg, "terrain_from_usgs")
    if not os.path.exists(str_terrain_from_usgs_dir):
        os.mkdir(str_terrain_from_usgs_dir)
    
    # *** variables set - raster terrain harvesting ***
    # ==============================================
    int_res = 3 # resolution of the downloaded terrain (meters)
    int_buffer = 300 # buffer distance for each watershed shp
    int_tile = 1500 # tile size requested from USGS WCS
    # ==============================================
    
    # field name is from the National watershed boundary dataset (WBD)
    str_field_name = "HUC_12"
    
    # run the third script (conflate_hecras_to_nwm)
    fn_get_usgs_dem_from_shape(str_input_path,
                               str_terrain_from_usgs_dir,
                               int_res,
                               int_buffer,
                               int_tile,
                               b_is_feet,
                               str_field_name)
    # -------------------------------------------

    # ------ Step 4: convert_tif_to_ras_hdf5 ----- 
    # 
    
    # folder of tifs created in third script (get_usgs_dem_from_shape)
    # str_terrain_from_usgs_dir
    
    # create a converted terrain folder
    str_hecras_terrain_dir = os.path.join(str_out_arg, "hecras_terrain")
    if not os.path.exists(str_hecras_terrain_dir):
        os.mkdir(str_hecras_terrain_dir)
        
    str_area_prj_name = str_huc8_arg + "_huc_12_ar.prj"
    str_projection_path = os.path.join(str_shapes_from_conflation_dir, str_area_prj_name)
    

    fn_convert_tif_to_ras_hdf5(str_hec_path,
                               str_terrain_from_usgs_dir,
                               str_hecras_terrain_dir,
                               str_projection_path,
                               b_is_feet)
    # -------------------------------------------
    
    # ------ Step 5: create_fim_rasters ----- 
    
    # folder of tifs created in third script (get_usgs_dem_from_shape)
    # str_terrain_from_usgs_dir
    
    # create a converted terrain folder
    str_fim_out_dir = os.path.join(str_out_arg, "hecras_output")
    if not os.path.exists(str_fim_out_dir):
        os.mkdir(str_fim_out_dir)
    
    # path to standard input (PlanStandardText01.txt, PlanStandardText02.txt, ProjectStandardText01.txt )
    str_std_input_path = os.getcwd() # assumed as in directory executing script
    
    # *** variables set - raster terrain harvesting ***
    # ==============================================
    flt_interval = 0.2 # vertical step of average depth (recommended - 0.2m and 0.5ft)
    flt_out_resolution = 3 # output depth raster resolution - meters
    # ==============================================
    
    fn_create_fim_rasters(str_huc8_arg,
                          str_shapes_from_conflation_dir,
                          str_fim_out_dir,
                          str_projection_path,
                          str_hecras_terrain_dir,
                          str_std_input_path,
                          flt_interval,
                          flt_out_resolution)
    # -------------------------------------------
    
    # ------ Step 6: simplify fim rasters -----
    # ==============================================
    flt_resolution_depth_grid = 3
    str_output_crs = "EPSG:3857"
    # ==============================================
    
    fn_simplify_fim_rasters(str_fim_out_dir,
                            flt_resolution_depth_grid,
                            str_output_crs)

    
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='========== RUN RAS2FIM FOR A HEC-RAS 1-D DATASET (HUC8) ==========')
    
    parser.add_argument('-w',
                        dest = "str_huc8_arg",
                        help='REQUIRED: HUC-8 watershed that is being evaluated: Example: 10170204',
                        required=True,
                        metavar='STRING',
                        type=str)
    
    parser.add_argument('-i',
                        dest = "str_ras_path_arg",
                        help=r'REQUIRED: path containing the HEC-RAS files: Example C:\HEC\ras_folder',
                        required=True,
                        metavar='DIR',
                        type=str)

    parser.add_argument('-o',
                        dest = "str_out_arg",
                        help=r'REQUIRED: path to write all the outputs: Example C:\HEC\output_folder',
                        required=True,
                        metavar='DIR',
                        type=str)
    
    parser.add_argument('-p',
                        dest = "str_crs_arg",
                        help=r'REQUIRED: projection of HEC-RAS models: Example EPSG:26915',
                        required=True,
                        metavar='STRING',
                        type=str)
    
    parser.add_argument('-v',
                        dest = "b_is_feet",
                        help='REQUIRED: create vertical data in feet: Default=True',
                        required=True,
                        default=True,
                        metavar='T/F',
                        type=str2bool)
    
    parser.add_argument('-n',
                        dest = "str_nation_arg",
                        help=r'REQUIRED: path to national datasets: Example: E:\X-NWS\X-National_Datasets',
                        required=True,
                        metavar='DIR',
                        type=str)
    
    parser.add_argument('-r',
                        dest = "str_hec_path",
                        help=r'REQUIRED: path to HEC-RAS 6.0: Example: "C:\Program Files (x86)\HEC\HEC-RAS\6.0" (wrap in quotes)',
                        required=True,
                        metavar='DIR',
                        type=str)

    args = vars(parser.parse_args())
    
    str_huc8_arg = args['str_huc8_arg']
    str_ras_path_arg = args['str_ras_path_arg']
    str_out_arg = args['str_out_arg']
    str_crs_arg = args['str_crs_arg']
    b_is_feet = args['b_is_feet']
    str_nation_arg = args['str_nation_arg']
    str_hec_path = args['str_hec_path']
    
    fn_run_ras2fim(str_huc8_arg, str_ras_path_arg, str_out_arg, str_crs_arg, b_is_feet, str_nation_arg, str_hec_path)