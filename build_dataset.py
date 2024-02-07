"""Build the features dataset from the raw antenna data."""
import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import scipy.stats

import load
import features


def build_antenna_dataset(
    data_path: pathlib.Path,
    file_out: pathlib.Path,
    nbins: int,
    antenna_len_str: str,
) -> None:
    # --------------
    # Load data for each video into separate dataframes
    (data_path / ".DS_Store").unlink(missing_ok=True)
    file_out.parent.mkdir(parents=True, exist_ok=True)
    data_paths = list(data_path.glob("*/CollectedData_Anna Hadjitofi.csv"))
    dfs = [load.df_from_manual_labels(path) for path in data_paths]
    # -----
    # Compute features for each labelled bee in each frame (all variables in radians)
    def loop_through_videos():
        for j, df in enumerate(dfs):
            print(j, "/", len(dfs))
            individuals = set(i[0] for i in df.index)
            for ind in individuals:
                if ind == "dancer":
                    continue
                df_ind = df.xs(ind)
                for frame in df_ind:
                    bee_df_for_frame = df_ind[frame].astype(float)
                    dancer_df_for_frame = df.xs("dancer")[frame].astype(float)
                    # print("Bee id", ind, "Frame", frame)
                    angle_to_dancer = features.angle_of_nestmate_rel_to_dancer(
                        dancer_df_for_frame, bee_df_for_frame
                    )  # Angle of nestmate relative to dancer
                    s_s_angle = features.s_s_angle(bee_df_for_frame)  # Between scapes
                    l_angle, r_angle = features.angle_of_antennae_rel_to_body(
                        bee_df_for_frame, antenna_len_str
                    )  # Angle of both antenna relative to body direction (straight ahead)
                    midpoint_angle = scipy.stats.circmean(
                        [l_angle, r_angle], high=np.pi, low=-np.pi
                    )  # Antenna midpoint
                    recruit_gravity_angle = features.angle_of_bee_rel_to_up(
                        bee_df_for_frame
                    )  # Angle of recruit relative to gravity
                    dancer_gravity_angle = features.angle_of_bee_rel_to_up(
                        dancer_df_for_frame
                    )  # Angle of dancer relative to gravity
                    yield ind, frame, angle_to_dancer, s_s_angle, l_angle, r_angle, midpoint_angle, recruit_gravity_angle, dancer_gravity_angle

    (
        bee_ids,
        frames,
        angles_to_dancer,
        s_s_angles,
        l_antenna_angles,
        r_antenna_angles,
        midpoints_angles,
        recruit_gravity_angles,
        dancer_gravity_angles,
    ) = zip(*loop_through_videos())
    angles_to_dancer_deg = np.rad2deg(angles_to_dancer)
    remapped_angle_to_dancer_deg = features.angle_to_360(angles_to_dancer_deg)
    # -----
    # Store appropriate representations of the data
    features_df = pd.DataFrame(
        {
            "bee_id": bee_ids,
            "frame": frames,
            "angle_to_dancer_rad": angles_to_dancer,
            "angle_to_dancer_deg": angles_to_dancer_deg,
            "remapped_angle_to_dancer_rad": np.deg2rad(remapped_angle_to_dancer_deg),
            "remapped_angle_to_dancer_deg": remapped_angle_to_dancer_deg,
            "s_s_rad": s_s_angles,
            "s_s_deg": np.rad2deg(s_s_angles),
            "l_antenna_rad": l_antenna_angles,
            "l_antenna_deg": np.rad2deg(l_antenna_angles),
            "r_antenna_rad": r_antenna_angles,
            "r_antenna_deg": np.rad2deg(r_antenna_angles),
            "antenna_midpoint_rad": midpoints_angles,
            "antenna_midpoint_deg": np.rad2deg(midpoints_angles),
            "recruit_angle_to_gravity_rad": recruit_gravity_angles,
            "recruit_angle_to_gravity_deg": np.rad2deg(recruit_gravity_angles),
            "dancers_angle_to_gravity_rad": dancer_gravity_angles,
            "dancers_angle_to_gravity_deg": np.rad2deg(dancer_gravity_angles),
        }
    ).dropna(
        axis=0
    )  # Drops frames for individuals that dont have labels at that time
    # -----
    # Compute mean orientation of nestmate to their dancer at each time point
    orientations_df = {}
    longest_time_range, window_size = 0, 8
    for name, group in features_df.groupby("bee_id"):
        if group["angle_to_dancer_rad"].shape[0] < 10:
            print(f"Skipping mean orientation for bee {name}. < 10 points")
            continue
        # For each group, compute the appropriate relative timestamps over the time
        # that nestmate was recorded
        time_range = features.get_time_scale(group["angle_to_dancer_rad"].values)
        averaged_time = np.mean(
            np.lib.stride_tricks.sliding_window_view(
                time_range, window_shape=(window_size,)
            ),
            axis=1,
        ).astype(int)
        orientations_df[name] = {
            "time": averaged_time,
            "mean_orientation": features.compute_circmean_over_windows(
                group["angle_to_dancer_rad"].values, window_size
            ),
            "mean_orientation_gravity": features.compute_circmean_over_windows(
                group["dancers_angle_to_gravity_rad"].values, window_size
            ),
        }
        longest_time_range = max(longest_time_range, max(time_range))
    orientations_df = (
        pd.DataFrame(orientations_df)
        .T.explode(["time", "mean_orientation", "mean_orientation_gravity"])
        .reset_index()
        .rename(columns={"index": "bee_id"})
    )
    # -----
    # Compute binned features
    features_df["bin_idx_of_angle_to_dancer_deg"], bins = pd.cut(
        features_df["angle_to_dancer_deg"],
        bins=nbins,
        retbins=True,
        labels=range(nbins),
    )
    bins = np.roll(bins, shift=-1)
    binned_df = (
        features_df.groupby("bin_idx_of_angle_to_dancer_deg")
        .apply(lambda group: features.compute_statistics_for_group(group))
        .reset_index()
    )
    binned_df["bin_angle_to_dancer_deg"] = bins[
        binned_df["bin_idx_of_angle_to_dancer_deg"]
    ]
    binned_df["bin_angle_to_dancer_rad"] = np.deg2rad(
        binned_df["bin_angle_to_dancer_deg"]
    )
    # Remap angle between -180 and 180 to 0 and 360
    binned_df["remapped_bin_angle_to_dancer_deg"] = binned_df[
        "bin_angle_to_dancer_deg"
    ].apply(lambda x: features.angle_to_360(x))
    binned_df["remapped_bin_angle_to_dancer_rad"] = np.deg2rad(
        binned_df["remapped_bin_angle_to_dancer_deg"]
    )
    # -----
    # Compute gravity angles data (mean gravity angle of dancer for each nestmate)
    dancers_gravity_df = {}
    for name, group in features_df.groupby("bee_id"):
        mean_dancer_angle = scipy.stats.circmean(
            group["dancers_angle_to_gravity_rad"].values, high=np.pi, low=-np.pi
        )
        dancers_gravity_df[name] = mean_dancer_angle
    dancers_gravity_df = (
        pd.DataFrame(dancers_gravity_df, index=[0])
        .T.reset_index()
        .rename(columns={"index": "bee_id", 0: "mean_dancer_angle_to_gravity_rad"})
    )
    # -----
    # Save data
    if antenna_len_str == "mid_only":
        file_out = str(file_out) + "-mid_antenna_len"
    elif antenna_len_str == "full_only":
        file_out = str(file_out) + "-full_antenna_len"
    features_df.to_csv(f"{file_out}-features.csv", index=None)
    orientations_df.to_csv(f"{file_out}-orientations.csv", index=None)
    binned_df.to_csv(f"{file_out}-{nbins}-binned.csv", index=None)
    dancers_gravity_df.to_csv(f"{file_out}-{nbins}-gravity.csv", index=None)
    # --------------


def build_error_dataset(data_path: pathlib.Path, file_out: pathlib.Path, nbins: int):
    """Needs to be run after the above dataset has been built and we have simulated
    the CX activity."""
    # --------------
    # Obtain list of np results files generated for each individual bee
    files = list((data_path / "real_bee_dataset").glob("*.npz"))
    (
        errors_df,
        reduced_errors_df,
        binned_error_df,
        consec_waggle_phases,
    ) = load.load_error_results_for_bees(files, nbins)
    # -----
    # Save data
    experiment = data_path.name
    errors_df.to_csv(f"{file_out}-{experiment}-{nbins}bins-errors.csv", index=None)
    reduced_errors_df.to_csv(
        f"{file_out}-{experiment}-{nbins}bins-reduced_errors.csv", index=None
    )
    binned_error_df.to_csv(
        f"{file_out}-{experiment}-{nbins}bins-binned_errors.csv", index=None
    )
    consec_waggle_phases.to_csv(
        f"{file_out}-{experiment}-{nbins}bins-consec_errors.csv", index=None
    )
    # --------------


if __name__ == "__main__":
    # --------------
    # User inputs
    parser = argparse.ArgumentParser(description="Build antennal positioning dataset.")
    parser.add_argument(
        "--build_errors_ds",
        action="store_true",
        help="Build error dataset\
        instead of antennae dataset.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Cropped-Anna Hadjitofi-2022-11-01",
        help="Name of raw data folder. Defaults to 'Cropped-Anna Hadjitofi-2022-11-01'. \
            For building errors dataset, this should be the path to the folder containing \
            the experiments.",
    )
    parser.add_argument(
        "--file_out",
        type=str,
        default=f"datasets/{time.strftime('%Y-%m-%d')}",
        help="Resulting name (and path) to save dataset(s), without extension. \
          Defaults to current date.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=180,
        help="Number of bins to use for binning angles to dancer. Defaults to 180.",
    )
    parser.add_argument(
        "--antenna_len_str",
        default="default",
        type=str,
        choices=["default", "mid_only", "full_only"],
        help="Calculate antenna angle using mid length / bend or full length. The \
            default uses base to tip if available and midpoint as fallback.",
    )
    args = vars(parser.parse_args())
    # -----
    args["file_out"] = pathlib.Path(args["file_out"])
    # Build and save dataset
    build_errors = args.pop("build_errors_ds")
    if build_errors:
        data_path = pathlib.Path(args["data_path"])
        args.pop("antenna_len_str")
        for experiment in data_path.glob("holoFly-*"):
            print(experiment)
            args["data_path"] = experiment
            build_error_dataset(**args)
    else:
        args["data_path"] = pathlib.Path(
            "dlc_antennae_tracking", args["data_path"], "labeled-data"
        )
        build_antenna_dataset(**args)
    # --------------
