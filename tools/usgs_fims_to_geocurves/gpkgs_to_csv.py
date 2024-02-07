#!/usr/bin/env python

"""
This script merges multiple GeoPackage (GPKG) files into CSV files. It is designed to process a collection of GPKG files,
group them based on a common prefix in their filenames, and then merge each group into a single CSV file. The script
utilizes the 'ogr2ogr' tool from the GDAL library for the conversion process.

The script expects each GPKG filename to follow a naming convention where the files intended to be grouped together
share a common prefix. This prefix is used to name the output CSV files.

Usage:
    The script is run from the command line, accepting two mandatory arguments:
    - `-i` or `--input_folder`: The directory path containing the GPKG files to be processed.
    - `-o` or `--output_folder`: The directory path where the output CSV files will be saved.

Example:
    `python gpkgs_to_csv.py -i "/d/Users/abdul.siddiqui/workbench/projects/usgs_to_geocurves/outputs/d28a2c4/polys" \
        -o "/d/Users/abdul.siddiqui/workbench/projects/usgs_to_geocurves/outputs/d28a2c4/geocurves"`

Requirements:
    - GDAL's ogr2ogr command-line tool must be installed and accessible in the system's PATH.
"""

import argparse
import os
import subprocess
from collections import defaultdict


def merge_gpkgs(input_folder, output_folder):
    # Dictionary to hold file groupings
    file_groups = defaultdict(list)

    # Iterate over files in the input folder and group them
    for file in os.listdir(input_folder):
        if file.endswith('.gpkg'):
            group_key = '_'.join(file.split('_')[:-2])
            file_groups[group_key].append(os.path.join(input_folder, file))

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each group and merge into a single CSV using ogr2ogr
    for group_key, files in file_groups.items():
        output_csv = os.path.join(output_folder, f'{group_key}_rating_curve_geo.csv')
        for idx, file in enumerate(files):
            mode = '-overwrite' if idx == 0 else '-append'
            command = ['ogr2ogr', '-f', 'CSV', mode, output_csv, file, '-nln', f'{group_key}_rating_curve_geo']
            command += (
                [
                    '-lco',
                    'GEOMETRY=AS_WKT',  # Export geometry as WKT,
                    '-lco',
                    'GEOMETRY_NAME=geometry',
                    '-lco',
                    'CREATE_CSVT=YES',
                ]
                if idx == 0
                else []
            )

            subprocess.run(command, check=True)
        print(f'Merged data saved to {output_csv}')


def main():
    parser = argparse.ArgumentParser(description="Script to merge GPKG files into CSVs.")
    parser.add_argument("-i", "--input_folder", required=True, type=str, help="Directory containing GPKG files.")
    parser.add_argument("-o", "--output_folder", required=True, type=str, help="Directory to store output CSV files.")
    args = parser.parse_args()

    # Call the merge function
    merge_gpkgs(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()
