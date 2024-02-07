#!/bin/bash

# This script applies a specified SQL query to each GPKG file in a given directory.
# It use the TABLE_NAME key word as place holder for the table name, which it will get
# from file name. It counts any errors and reports them at the end.
# Usage:
#   ./script.sh /path/to/gpkg/folder "SQL query with TABLE_NAME as placeholder for the table name"
# Example:
#   ./query_gpkgs.sh "D:\Users\abdul.siddiqui\workbench\projects\usgs_to_geocurves\outputs\d40bbae\geocurve_polys" \
#      "UPDATE TABLE_NAME SET filename = REPLACE(filename, \
#      'D:/Users/abdul.siddiqui/workbench/projects/usgs_to_geocurves/outputs_referenced\geocurve_polys/', '')"


FOLDER_PATH="$1"  # Path to the folder containing GPKG files
SQL_TEMPLATE="$2" # SQL query template with TABLE_NAME as placeholder
ERROR_COUNT=0     # Counter for errors

# Check if folder path and SQL query template are provided
if [ -z "$FOLDER_PATH" ] || [ -z "$SQL_TEMPLATE" ]; then
    echo "Error: Folder path or SQL query template not provided."
    echo "Usage: ./script.sh /path/to/gpkg/folder 'SQL query with TABLE_NAME as placeholder'"
    exit 1
fi

# Loop through each GPKG file in the folder and apply the SQL query
for FILE in "$FOLDER_PATH"/*.gpkg; do
    if [ -f "$FILE" ]; then
        # Extract file name without extension to use as table name
        TABLE_NAME=$(basename "$FILE" .gpkg)

        # Replace TABLE_NAME placeholder with actual table name in SQL query
        SQL_QUERY=${SQL_TEMPLATE//TABLE_NAME/\"$TABLE_NAME\"}

        if ! ogrinfo "$FILE" -sql "$SQL_QUERY"; then
            echo "Error occurred in $FILE"
            ((ERROR_COUNT++))
        fi
    fi
done

# Report error count
if [ $ERROR_COUNT -gt 0 ]; then
    echo "Completed with $ERROR_COUNT errors."
else
    echo "Processing complete with no errors."
fi
