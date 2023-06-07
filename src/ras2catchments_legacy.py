    
import shutil
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.merge import merge
import rasterio
import rasterio.mask
import re
import fiona
import fiona.transform
from shapely.geometry import Polygon
from rasterio import features
import glob
import time
import datetime as dt
from osgeo import gdal
import keepachangelog
import argparse

#
# I'd like to move this conversion to be a separate external utility,
# but for now I think it's fine here because it's related to getting
# everything in the right format for Hydrovis.
#
def convert_to_metric(ras2rem_dir, huc):
    src_path = os.path.join(ras2rem_dir,'rating_curve.csv')
    df = pd.read_csv(src_path)

    # convert to metric if needed
    if 'stage_m' not in df.columns: # if no meters, then only Imperial units are in the file
        df["stage_m"] = ["{:0.2f}".format(h * 0.3048) for h in df["stage_ft"]]
        df["discharge_cms"] = ["{:0.2f}".format(q * 0.0283168) for q in df["discharge_cfs"]]
        df.to_csv(os.path.join(src_path),index=False)

        # REM will also be in Imperial units if the incoming SRC was
        rem_path = os.path.join(ras2rem_dir,'rem.tif')
        with rasterio.open(rem_path) as src:
            raster = src.read()
            raster = np.multiply(raster, np.where(raster != 65535,0.3048,1)) # keep no-data value as-is
            output_meta = src.meta.copy()
        with rasterio.open(rem_path,'w',**output_meta,compress="LZW") as dest:
            dest.write(raster)

    return

#
# Scrapes the top changelog version (most recent version listed in our repo)
#
def get_changelog_version(changelog_path):
    changelog = keepachangelog.to_dict(changelog_path)
    return list(changelog.keys())[0]

# This function creates a geopackage of Feature IDs from the raster that
# matches the Depth Grid processing extents ("maxments") from ras2fim Step 5.
def vectorize(path,changelog_path,catalog_path,src_path):
    with rasterio.open(path) as src:
        rast = src.read()

    #
    # SETUP DATA FROM FEATURE ID RASTER
    #
    shapes = features.shapes(rast,transform=src.transform)

    geoms = []
    vals = []
    for shp,val in shapes:
      geoms.append(Polygon(shp["coordinates"][0]))
      vals.append(val) # feature IDs

    # Dataframe of Feature ID values to complement the geometries of the polygons derived from raster
    df = pd.DataFrame(vals)
    df.columns = ["feature_id"]
    df["HydroID"] = vals
    
    # We pass in (potentially) many different depths for each Feature ID, so dissolve them together
    gdf = gpd.GeoDataFrame(df,crs=src.crs,geometry=geoms)
    gdf = gdf.dissolve('feature_id') # ensure that we just have one big polygon per feature ID

    #
    # MODEL CATALOG DATA
    #
    model_df = pd.read_csv(catalog_path)
    
    # Join GeoDataFrame to model catalog that has some relevant metadata; keep only relevant fields that match our feature IDs
    gdf = gdf.merge(model_df,how="left",left_on="feature_id",right_on="nhdplus_comid")
    columns_to_keep = ["geometry","HydroID","source","last_modified","model_name"]
    gdf = gdf[columns_to_keep]
    gdf['last_modified'] = gdf['last_modified'].fillna(-1)
    try:
        gdf['last_modified'] = [dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in gdf['last_modified']]
    except Exception as e:
        print("Model catalog date issue. Moving on.")
        print(e)


    #
    # SRC DATA
    #
    src_df = pd.read_csv(src_path)
    
    # Get min and max stage and discharge for each Feature ID, then create fields for stage and flow mins and maxs
    agg_df = src_df.groupby(["feature_id"]).agg({'stage_m': ['min', 'max'], 'discharge_cms': ['min','max']})
    
    agg_df['start_stage'] = agg_df['stage_m']['min']
    agg_df['end_stage'] = agg_df['stage_m']['max']
    agg_df['start_flow'] = agg_df['discharge_cms']['min']
    agg_df['end_flow'] = agg_df['discharge_cms']['max']

    # Flatten from MultiIndex and keep relevant columns
    agg_df.columns = ["_".join(c).rstrip('_') for c in agg_df.columns.to_flat_index()]
    agg_df = agg_df[['start_stage','end_stage','start_flow','end_flow']]

    # Join SRC data to polygon GeoDataFrame
    gdf = gdf.merge(agg_df,how="left",left_on="HydroID",right_on="feature_id")
    
    #
    # Placeholders for other metadata
    #
    gdf['ras_model_type'] = ['1D'] * len(gdf)
    gdf['model_date'] = ["Unknown"] * len(gdf) # where would we find this in the model data?
    gdf['flow_match_score'] = [0] * len(gdf)
    gdf['terrain_match_score'] = [0] * len(gdf)
    gdf['other_notes'] = [""] * len(gdf)

    gdf['ras2fim_version'] = [get_changelog_version(changelog_path)] * len(gdf)

    # ... then we should have all the metadata we need on a per-catchment basis 

    print("** Writing out updated catchments geopackage **")
    gdf.to_file(path.replace(".tif","_meta.gpkg"),driver="GPKG",layer="catchments")
    return

# This function creates a raster AND geopackage of Feature IDs that correspond to the extent
# of the Depth Grids (and subsequent REMs that also match the Depth Grids)
def make_catchments(input_dir,output_dir,nwm_feature_id_polygons,temp_folder):
    start_time = time.time()    
    all_tif_files=list(Path(input_dir).rglob('*/Depth_Grid/*.tif'))
    feature_id_values = list(map(lambda var:str(var).split(".tif")[0].split("-")[-2], all_tif_files))

    all_feature_ids = list(set(feature_id_values))
    all_feature_ids = [int(str(x).split('\\')[-1].split('-')[0]) for x in all_feature_ids]
    print(f"Looking through {all_feature_ids}")

    # This file will be written out to contain a subset of polygons that correspond to the feature IDs of interest
    temp_sub_polygons = os.path.join(temp_folder,"subset_polys.shp")
    
    # Creating a subset of polygons that only contain the feature IDs of interest.
    # This step is technically optional, but it will speed up processing immensely so that we don't
    # have to touch the full multi-million NWM catchments dataset more than once, if ever
    with fiona.open(nwm_feature_id_polygons) as shapefile:
        print(".. getting all relevant shapes")
        with fiona.open(temp_sub_polygons,"w",driver=shapefile.driver, \
                        schema=shapefile.schema,crs=shapefile.crs) as dest:
            for x in shapefile:
                if x["properties"]["FEATUREID"] in all_feature_ids:
                    print(f"...... writing {x['properties']['FEATUREID']}")
                    dest.write(x)

    print("TIME to create subset polygons: " +str(time.time()-start_time))
    
    raster_to_mosaic = []
    # Traverse through all feature IDs that we found depth grid folders for earlier
    for feature_id in all_feature_ids:
        # Pull all relevant depth grid tiffs for this feature ID
        this_feature_id_tif_files = list(Path(input_dir).rglob(f'*/Depth_Grid/{feature_id}-*.tif'))
        print(this_feature_id_tif_files)
        max_rem = -1
        max_rem_file = ""
        # Look through all depth grid tiffs associated with this feature ID and find the max depth value and file
        for p in this_feature_id_tif_files:
            print(p)
            rem = int(str(p).split('\\')[-1].split('-')[1].rstrip('.tif')) # get rem value from filename
            print(f"Looking at {rem} for {feature_id}... max_rem was {max_rem} before this")
            if(rem > max_rem):
                max_rem = rem
                max_rem_file = p

        raster = rasterio.open(max_rem_file)

        # The following file will be written to contain a projected version of the subset of polygons, to make sure
        # we match our raster projections.
        # Create a matching projection version of the subset of possible catchments; only need to do this the first time
        # because each subsequent run can use this same file
        temp_proj_polygons = os.path.join(temp_folder,"temp_sub_proj_polys.shp") # reproj
        if not os.path.exists(temp_proj_polygons):
            print("Projecting Feature ID polygons from NWM Catchments")
            print(raster.crs)
            df = gpd.read_file(temp_sub_polygons)
            print(df.crs)
            df = df.to_crs(raster.crs)
            df.to_file(temp_proj_polygons)

        # Get the polygon associated with this current feature ID
        with fiona.open(temp_proj_polygons) as subset_shapefile:
            shapes = [feature["geometry"] for feature in subset_shapefile if feature['properties']['FEATUREID']==int(feature_id)]
        
        # ... apply polygon as mask to initial raster. Will only pull one feature ID at a time. 
        # Clips depth grid to catchment boundary to create "maxment"
        masked_raster, transform = rasterio.mask.mask(raster, shapes, nodata=raster.nodata, filled=True, invert=False)

        # Force datatypes to be 32-bit integers
        long_raster_vals = np.array(masked_raster,dtype=np.int32)
        long_raster_vals[long_raster_vals != raster.nodata] = np.int32(feature_id)
            
        # write 32-bit integer versions of our feature ID data, to ensure datatyping
        feature_id_rem_path = os.path.join(output_dir,"temp_{}_{}.tif".format(max_rem,feature_id))
        output_meta = raster.meta.copy()
        output_meta.update({"dtype":np.int32})
        with rasterio.open(feature_id_rem_path, "w", **output_meta) as m:
            m.write(long_raster_vals)

        # append particular feature_id + depth value path to list
        raster_to_mosaic.append(feature_id_rem_path)

    # Merge all the feature IDs' max depths together
    print("Merging all max depths")
    print(raster_to_mosaic)
    mosaic, output = merge(list(map(rasterio.open,raster_to_mosaic)), method="min")

    # Setup the metadata for our raster
    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
         "height": mosaic.shape[1],
         "width": mosaic.shape[2],
         "transform": output,
         "dtype":np.int32,
         }
    )
    # Write out the feature_id raster file, which is one of our required outputs
    feature_id_path = os.path.join(output_dir,"feature_id.tif")
    with rasterio.open(feature_id_path, "w", **output_meta,compress="LZW") as m:
        m.write(mosaic)
        print(f"** Writing final feature_id mosaic to {feature_id_path}")

    # Clean up intermediate files
    for p in raster_to_mosaic:
        os.remove(p)

    print(f"Removing {temp_sub_polygons} and {temp_proj_polygons}")
    temp_sub_poly_files = glob.glob(temp_sub_polygons.replace('.shp','.*'))
    temp_proj_poly_files = glob.glob(temp_proj_polygons.replace('.shp','.*'))

    # Remove other related files that shapefiles require
    for f in temp_sub_poly_files:
        os.remove(f)
    for f in temp_proj_poly_files:
        os.remove(f)

    print("TIME to finish raster creation: " +str(time.time()-start_time))
    
    return feature_id_path


def main_call(huc_num, root_dir, nwm_catchments_shapefile,
                    temp_folder, changelog_path, catalog_path):

    input_dir= os.path.join(root_dir,"05_hecras_output")
    ras2rem_dir = os.path.join(root_dir,"06_ras2rem")
    output_dir= os.path.join(root_dir,"08_eff_catchment_output")
    
    # converts SRC and REM to metric if needed based on the existing SRC columns
    convert_to_metric(ras2rem_dir, huc_num)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
    feature_id_path = make_catchments(input_dir,output_dir,nwm_catchments_shapefile,temp_folder)

    print("*** Vectorizing Feature IDs and creating metadata")
    vectorize(feature_id_path,changelog_path,catalog_path,
              os.path.join(ras2rem_dir,'rating_curve.csv'))

    return    

if __name__=="__main__":
    # delete this environment variable because the updated library we need
    # is included in the rasterio wheel
    try:
        del os.environ["PROJ_LIB"]
    except Exception as e:
        print(e)
        
    parser = argparse.ArgumentParser(description='========== Create catchments for specified existing output_ras2fim_models folder ==========')

    parser.add_argument('-w',
                        dest = "huc_num",
                        help = 'REQUIRED: HUC-8 watershed that is being evaluated: Example: 10170204',
                        required = True,
                        type = str)

    parser.add_argument('-p',
                        dest = "root_dir",
                        help = r'REQUIRED: Full path that contains results folders of a ras2fim run.',
                        required = True,
                        type = str) 

    parser.add_argument('-n',
                        dest = "nwm_catchments_shapefile",
                        help = r'REQUIRED: path to NWM catchments: Example: E:\X-NWS\X-National_Datasets\nwm_catchments.shp.',
                        required = True,
                        type = str)

    parser.add_argument('-t',
                        dest = "temp_folder",
                        help = r'REQUIRED: path to temporary working folder. Example: C:/users/<user_name>/Documents/temp.',
                        required = True,
                        type = str)
    
    parser.add_argument('-c',
                        dest = "changelog_path",
                        help = r'REQUIRED: path to changelog file in this repo',
                        required = True,
                        type = str)

    parser.add_argument('-m',
                        dest = "catalog_path",
                        help = r'REQUIRED: path to model catalog csv file downloaded from S3',
                        required = True,
                        type = str)

    args = vars(parser.parse_args())
    # These paths should be rewired to point to the following:
    # Input directory - results from ras2fim Step 5 that contain depth grids
    # Output directory - desired output directory for feature ID files (currently points to ras2rem outputs)

    # Example run:
    # python working_ras2catchments_efficient.py -p "C:\Users\rdp-user\Documents\colorado" -n "C:\Users\rdp-user\Documents\nwm_catchments_clip\nwm_catchments_tx.shp" -t "c:\users\rdp-user\documents" -c "C:\Users\rdp-user\Projects\code_lkeys\ras2fim\doc\CHANGELOG.md" -m "c:/users/rdp-user/downloads/model_catalog.csv" -w 12090301

    main_call(**args)
    
