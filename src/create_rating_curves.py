# Creates depth grids and SRC for feature-ids in HUC8
# Uses the 'ras2fim' conda environment

# -------------------------------------------------
import argparse
import os
import traceback
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import pandas as pd

# import ras2fim_logger
import shared_variables as sv


# Global Variables
RLOG = sv.R2F_LOG  # the non mp version


# -------------------------------------------------
# Ploting synthetic rating curves
def plot_src(
    str_feature_id, list_int_step_flows, list_step_wse, str_rating_path_to_create, str_file_name, model_unit
):
    fig = plt.figure()
    fig.patch.set_facecolor("gainsboro")
    fig.suptitle("FEATURE ID: " + str_feature_id, fontsize=18, fontweight="bold")

    ax = plt.gca()
    today = date.today()

    ax.text(
        0.98,
        0.04,
        "Created: " + str(today),
        verticalalignment="bottom",
        horizontalalignment="right",
        backgroundcolor="w",
        transform=ax.transAxes,
        fontsize=6,
        style="italic",
    )

    ax.text(
        0.98,
        0.09,
        "Computed from HEC-RAS models",
        verticalalignment="bottom",
        horizontalalignment="right",
        backgroundcolor="w",
        transform=ax.transAxes,
        fontsize=6,
        style="italic",
    )

    ax.text(
        0.98,
        0.14,
        "NOAA - Office of Water Prediction",
        verticalalignment="bottom",
        horizontalalignment="right",
        backgroundcolor="w",
        transform=ax.transAxes,
        fontsize=6,
        style="italic",
    )

    plt.plot(list_int_step_flows, list_step_wse)  # creates the line
    plt.plot(list_int_step_flows, list_step_wse, "bd")
    # adding blue diamond points on line

    ax.get_xaxis().set_major_formatter(tick.FuncFormatter(lambda x, p: format(int(x), ",")))

    plt.xticks(rotation=90)

    if model_unit == "meter":
        plt.ylabel("Average Depth (m)")
        plt.xlabel("Discharge (m^3/s)")
    else:
        plt.ylabel("Average Depth (ft)")
        plt.xlabel("Discharge (ft^3/s)")

    plt.grid(True)

    str_rating_image_path = os.path.join(str_rating_path_to_create, str_file_name)
    plt.savefig(str_rating_image_path, dpi=300, bbox_inches="tight")

    plt.cla()
    plt.close("all")


# -------------------------------------------------
def cast_to_int(x):
    if str(x).endswith("*"):
        x = x[:-1]
    x = int(float(x))
    return x


# -------------------------------------------------
def fn_create_rating_curves(huc8, path_unit_folder):
    int_number_of_steps = 76
    model_unit = 'feet'

    RLOG.lprint("")
    RLOG.lprint("+=================================================================+")
    RLOG.notice("|               CREATING SYNTHETIC RATING CURVES                  |")
    RLOG.lprint("+-----------------------------------------------------------------+")

    # Reading data_summary from step 2
    str_path_to_fid_xs = os.path.join(
        path_unit_folder, sv.R2F_OUTPUT_DIR_SHAPES_FROM_CONF, f"{huc8}_stream_qc_fid_xs.csv"
    )

    fid_xs_huc8 = pd.read_csv(str_path_to_fid_xs)

    path_conflated_models_splt = [path.split("\\") for path in list(fid_xs_huc8['ras_path'])]
    conflated_model_names = [names[-2] for names in path_conflated_models_splt]

    # -------------------------------------------------
    # Reading model_catalog to add model_ids to data_summary
    path_model_catalog2 = os.path.join(path_unit_folder, f"OWP_ras_models_catalog_{huc8}.csv")

    model_catalog = pd.read_csv(path_model_catalog2)
    models_name_id = pd.concat([model_catalog["final_name_key"], model_catalog["model_id"]], axis=1)

    final_name_key = list(models_name_id["final_name_key"])

    # -------------------------------------------------
    # Assigning model_ids to data_summary
    # -------------------------------------------------
    conflated_model_names_id = []
    for nms in conflated_model_names:
        indx = final_name_key.index(nms)

        name_id = list(models_name_id.iloc[indx])

        conflated_model_names_id.append(name_id)

    conflated_model_names_id_df = pd.DataFrame(
        conflated_model_names_id, columns=["final_name_key", "model_id"]
    )

    df_fid_xs_huc8 = pd.concat([fid_xs_huc8, conflated_model_names_id_df], axis=1)

    # -------------------------------------------------
    # Reading all_x_sections_info (results from step5) for all conflated ras streams
    # Determing paths to the step 5 results
    path_to_step5 = os.path.join(path_unit_folder, sv.R2F_OUTPUT_DIR_HECRAS_OUTPUT)
    created_ras_models_folders = os.listdir(path_to_step5)

    path_to_all_x_sections_info = []
    for folders in created_ras_models_folders:
        path_to_all_xs_info = os.path.join(path_to_step5, folders, f"all_x_sections_info_{folders}.csv")

        path_to_all_x_sections_info.append(path_to_all_xs_info)

    # -------------------------------------------------
    # Assigning feature_ids from df_fid_xs_huc8 to all_x_sections_info
    # -------------------------------------------------
    # Creating a for loop going through all all_x_sections_info
    # for each confalted stream (step 5 results)
    for infoind in range(len(path_to_all_x_sections_info)):
        mid_x_sections_info = pd.read_csv(path_to_all_x_sections_info[infoind])  #
        mid_x_sections_info = mid_x_sections_info.rename(columns={'fid_xs': 'mid_xs', 'modelid': 'model_id'})

        model_id = mid_x_sections_info['model_id'][0]

        df_fid_xs_mid = pd.DataFrame()

        for mid in range(len(df_fid_xs_huc8['model_id'])):
            if df_fid_xs_huc8['model_id'][mid] == model_id:
                mid_info = df_fid_xs_huc8.iloc[mid][
                    ['feature_id', 'river', 'model_id', 'us_xs', 'ds_xs', 'peak_flow']
                ]

                df_fid_xs_mid = pd.concat([df_fid_xs_mid, mid_info], axis=1)

        df_fid_xs_mid = df_fid_xs_mid.T
        df_fid_xs_mid = df_fid_xs_mid.sort_values(by=['us_xs'], ascending=False)
        df_fid_xs_mid.index = range(len(df_fid_xs_mid))

        # Discussed in our ras2fim meeting (2023-12-28). Conclusion:
        # Inclusion of upstreams XS that are not part of the nwm
        # feature_ids in average depth per feature_if.

        df_XS_name = pd.DataFrame(mid_x_sections_info['Xsection_name'].apply(cast_to_int))

        # TODO: will be removed in the next PR
        # if type(mid_x_sections_info['Xsection_name'][0]) == str:

        #     df_XS_name0 = pd.DataFrame(mid_x_sections_info['Xsection_name'])
        #     xs_name = []
        #     for indx_xs in range(len(df_XS_name0)):

        #         if df_XS_name0['Xsection_name'][indx_xs][-1] == "*":

        #             xs_name.append(np.int64(float(df_XS_name0['Xsection_name'][indx_xs][0:-1])))

        #         else: xs_name.append(np.int64(df_XS_name0['Xsection_name'][indx_xs]))

        #     df_XS_name = pd.DataFrame(xs_name, columns = ['Xsection_name'])

        # else:
        #     df_XS_name = pd.DataFrame(mid_x_sections_info['Xsection_name'])

        maxind = 0  # df_fid_xs_mid["us_xs"].astype(float).idxmax()
        mid_fid = {
            (indxh, rowh['Xsection_name']): [maxind, df_fid_xs_mid['feature_id'][maxind]]
            for indxh, rowh in df_XS_name.iterrows()
        }

        for indxh, rowh in df_XS_name.iterrows():
            for indxh1, rowh1 in df_fid_xs_mid.iterrows():
                # print(indxh1, rowh1['ds_xs'], rowh1['us_xs'])
                if rowh1['ds_xs'] <= rowh['Xsection_name'] <= rowh1['us_xs']:
                    mid_fid[(indxh, rowh['Xsection_name'])] = [indxh1, df_fid_xs_mid['feature_id'][indxh1]]

        df_mid_fid = pd.DataFrame(mid_fid).T
        df_mid_fid.index = range(len(df_mid_fid))
        df_mid_fid.columns = ['fidindx', 'feature_id']

        mid_x_sections_info = mid_x_sections_info.rename(columns={'Unnamed: 0': 'xs_counter'})
        mid_x_sections_info_fid = pd.concat(
            [
                mid_x_sections_info[["mid_xs", "model_id", "xs_counter"]],
                df_XS_name["Xsection_name"],
                mid_x_sections_info[["wse", "discharge"]],
                df_mid_fid,
            ],
            axis=1,
        )

        mid_xs_info_fid = mid_x_sections_info_fid[
            ['model_id', 'feature_id', 'xs_counter', 'Xsection_name', 'wse', 'discharge']
        ]

        # -------------------------------------------------
        # Create profile names (numbers) and add it to the mid_xs_info_fid
        xs_counter = 1 + mid_x_sections_info_fid['xs_counter'].max()

        # profile_names_col = pd.DataFrame(
        #     [profile_names[i//xs_counter] for i in range(len(profile_names)*xs_counter)],
        #     columns = ['profile_name'])

        profile_num = [ns for ns in range(int_number_of_steps)]

        profile_num_col = pd.DataFrame(
            [profile_num[i // xs_counter] for i in range(len(profile_num) * xs_counter)],
            columns=['profile_num'],
        )

        mid_xs_info_fid = pd.concat([mid_xs_info_fid, profile_num_col], axis=1)

        # -------------------------------------------------
        # Grouped and averaged by 'profile_num', 'feature_id'
        mid_xs_info_fid_avr = mid_xs_info_fid.groupby(['profile_num', 'feature_id']).mean()

        fid_ind = mid_xs_info_fid_avr.index.get_level_values('feature_id').drop_duplicates()

        mid_xs_info_fid_1st = mid_xs_info_fid.groupby(['profile_num', 'feature_id']).first()
        mid_xs_info_fid_lst = mid_xs_info_fid.groupby(['profile_num', 'feature_id']).last()

        for fids in fid_ind:
            list_int_step_flows = list(
                mid_xs_info_fid_avr.iloc[mid_xs_info_fid_avr.index.get_level_values('feature_id') == fids][
                    'discharge'
                ]
            )
            list_step_wse = list(
                mid_xs_info_fid_avr.iloc[mid_xs_info_fid_avr.index.get_level_values('feature_id') == fids][
                    'wse'
                ]
            )
            str_feature_id = str(fids)

            fid_mid_x_sections_info_avr = mid_xs_info_fid_avr.iloc[
                mid_xs_info_fid_avr.index.get_level_values('feature_id') == fids
            ]  # .astype(int)
            fid_mid_x_sections_info_1st = mid_xs_info_fid_1st.iloc[
                mid_xs_info_fid_1st.index.get_level_values('feature_id') == fids
            ]
            fid_mid_x_sections_info_lst = mid_xs_info_fid_lst.iloc[
                mid_xs_info_fid_lst.index.get_level_values('feature_id') == fids
            ]
            xs_us_fid = fid_mid_x_sections_info_1st['Xsection_name']
            xs_us_fid = pd.DataFrame(xs_us_fid).rename(columns={'Xsection_name': 'xs_us'})
            xs_ds_fid = fid_mid_x_sections_info_lst['Xsection_name']
            xs_ds_fid = pd.DataFrame(xs_ds_fid).rename(columns={'Xsection_name': 'xs_ds'})

            fid_mid_x_sections_info_src = fid_mid_x_sections_info_avr[['model_id', 'wse', 'discharge']]
            fid_mid_x_sections_info_src = pd.concat(
                [fid_mid_x_sections_info_src, xs_us_fid, xs_ds_fid], axis=1
            )

            str_file_name = str_feature_id + "_rating_curve.png"

            # Create a Rating Curve folder
            str_rating_path_to_create = os.path.join(
                path_unit_folder,
                sv.R2F_OUTPUT_DIR_CREATE_RATING_CURVES,
                created_ras_models_folders[infoind],
                "Rating_Curve",
            )
            os.makedirs(str_rating_path_to_create, exist_ok=True)

            # -------------------------------------------------
            # Saving all cross sections info per feature_id
            # TODO: add meter units of feet and meter
            x_sections_info_fid = mid_x_sections_info_fid[mid_x_sections_info_fid['feature_id'] == fids]
            path_to_all_xs_info_fid = os.path.join(str_rating_path_to_create, f"all_xs_info_fid_{fids}.csv")

            discharge = pd.DataFrame(x_sections_info_fid['discharge'].astype(int), columns=['discharge'])
            wse = pd.DataFrame(x_sections_info_fid['wse'].astype(int), columns=['wse'])
            discharge_wse = pd.concat([discharge, wse], axis=1)

            x_sections_info_fid = x_sections_info_fid.drop(['discharge', 'wse'], axis=1)
            x_sections_info_fid = pd.concat([x_sections_info_fid, discharge_wse], axis=1)
            x_sections_info_fid.to_csv(path_to_all_xs_info_fid)

            # -------------------------------------------------
            # Plotting and saving synthetic rating curves
            str_xsection_path = os.path.join(str_rating_path_to_create, f"mean_xs_info_src_{fids}.csv")

            discharge2 = pd.DataFrame(
                fid_mid_x_sections_info_src['discharge'].astype(int), columns=['discharge']
            )
            wse2 = pd.DataFrame(fid_mid_x_sections_info_src['wse'].astype(int), columns=['wse'])
            discharge_wse2 = pd.concat([discharge2, wse2], axis=1)

            fid_mid_x_sections_info_src = fid_mid_x_sections_info_src.drop(['discharge', 'wse'], axis=1)
            fid_mid_x_sections_info_src = pd.concat([fid_mid_x_sections_info_src, discharge_wse2], axis=1)

            fid_mid_x_sections_info_src.to_csv(str_xsection_path, index=True)

            plot_src(
                str_feature_id,
                list_int_step_flows,
                list_step_wse,
                str_rating_path_to_create,
                str_file_name,
                model_unit,
            )
            #

    RLOG.success("Complete")


# -------------------------------------------------
if __name__ == "__main__":
    # Sample:
    # python create_rating_curves.py -w 12090301
    # -p 'C:\\ras2fimv2.0\\ras2fim_v2_output_12090301'

    parser = argparse.ArgumentParser(description="== CREATES SYNTHETIC RATING CURVES FOR FEATURE-IDS ==")

    parser.add_argument(
        "-w",
        dest="huc8",
        help="REQUIRED: HUC-8 watershed that is being evaluated: Example: 12090301",
        required=True,
        metavar="STRING",
        type=str,
    )

    parser.add_argument(
        "-p",
        dest="path_unit_folder",
        help=r"REQUIRED: Directory containing model catalog for HUC8:  Example: D:\12090301_2277_20231214",
        required=True,
        metavar="DIR",
        type=str,
    )

    args = vars(parser.parse_args())

    huc8 = args["huc8"]
    path_unit_folder = args["path_unit_folder"]

    log_file_folder = os.path.join(path_unit_folder, "logs")
    try:
        # Catch all exceptions through the script if it came
        # from command line.
        # Note.. this code block is only needed here if you are calling from command line.
        # Otherwise, the script calling one of the functions in here is assumed
        # to have setup the logger.

        # creates the log file name as the script name
        script_file_name = os.path.basename(__file__).split('.')[0]
        # Assumes RLOG has been added as a global var.
        RLOG.setup(os.path.join(log_file_folder, script_file_name + ".log"))

        # call main program
        fn_create_rating_curves(huc8, path_unit_folder)

    except Exception:
        RLOG.critical(traceback.format_exc())
