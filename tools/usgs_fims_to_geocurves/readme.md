# Instructions

1. Download required data
2. Create a `.env` file in this folder with `PROJ_LIB="D:\\Users\\abdul.siddiqui\\AppData\\Local\\anaconda3\\envs\\ras2fim\\Lib\\site-packages\\pyproj\\proj_dir\\share\\proj"`
3. Setup and activate `ras2fim` conda environment using [INSTALL](doc/INSTALL.md)
4. Run `pip install --upgrade geopandas`
5. Run `translate_usgs_library.py`. Example:
```bash
python translate_usgs_library.py -d ~/workbench/data/usgs_to_geocurves/usgs/fim2.gdb -o ~/workbench/projects/usgs_to_geocurves/outputs/$(git rev-parse --short HEAD) -c ~/workbench/data/usgs_to_geocurves/level_paths -rc ~/workbench/data/usgs_to_geocurves/usgs/usgs_rating_curves.csv -g ~/workbench/data/usgs_to_geocurves/usgs/fim_sites_huc12.gpkg -pp 4 -ll INFO
```
6. Run `gpkgs_to_csv.py`. Example
```bash
python gpkgs_to_csv.py -i "/d/Users/abdul.siddiqui/workbench/projects/usgs_to_geocurves/outputs/d28a2c4/polys" -o "/d/Users/abdul.siddiqui/workbench/projects/usgs_to_geocurves/outputs/d28a2c4/geocurves"
```