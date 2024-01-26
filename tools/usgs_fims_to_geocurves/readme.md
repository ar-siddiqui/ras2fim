# Instructions

1. Download required data
2. Create a `.env` file in this folder with `PROJ_LIB="D:\\Users\\abdul.siddiqui\\AppData\\Local\\anaconda3\\envs\\ras2fim\\Lib\\site-packages\\pyproj\\proj_dir\\share\\proj"`
3. Setup and activate `ras2fim` conda environment using [INSTALL](doc/INSTALL.md)
4. Run `pip install --upgrade geopandas`
5. Sample Usage:
```bash
python translate_usgs_library.py -d ~/workbench/data/usgs_to_geocurves/usgs/fim2.gdb -o ~/workbench/projects/usgs_to_geocurves/outputs -c ~/workbench/data/usgs_to_geocurves/level_paths -rc ~/workbench/data/usgs_to_geocurves/usgs/usgs_rating_curves.csv -g ~/workbench/data/usgs_to_geocurves/usgs/fim_sites_huc12.gpkg -pp 4 -ll INFO
```
