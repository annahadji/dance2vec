"""Helper functions to compute features of data."""
import math

import pandas as pd
import numpy as np
import scipy.stats
from typing import Tuple


def angle_to_360(angle: float):
    """Convert angles between 0-180 and -180-0 to 0-360."""
    return (angle + 360) % 360

def opposite_angle_quadrant(theta: float) -> float:
    """Returns the angle in the opposite quadrant."""
    if theta == 0.0:
        return np.radians(180)
    return theta - np.pi * np.sign(theta)

def signed_delta_angle(target: float, source: float) -> float:
    """Return the (smallest) signed delta angle between a target angle and a source,
    i.e. the angle you'd need to rotate from source to reach target."""
    return np.arctan2(np.sin(target - source), np.cos(target - source))

def polar2cartesian(theta: float, r_length: float) -> Tuple[float, float]:
    """Convert polar coordinates (e.g. home vector) to Cartesian coordinates."""
    x_coord = np.round(r_length * np.cos(theta), decimals=3)
    y_coord = np.round(r_length * np.sin(theta), decimals=3)
    return x_coord, y_coord

def cartesian2polar(x_coord: int, y_coord: int):
    """Convert X, Y Cartesian coordinates to polar coordinates (r, angle)."""
    rho = np.sqrt(x_coord ** 2 + y_coord ** 2)
    phi = np.arctan2(y_coord, x_coord)
    return rho, phi

def orientation_vector(bees_df_for_frame: pd.DataFrame) -> np.ndarray:
    """Compute the orientation vector of a bee in a frame. Orientation is regarded as
    the vector connecting the midpoint of the bee's thorax to the midpoint of the bee's
    antennae bases."""
    # Compute angle connecting back and midpoint of antennae bases
    start_x, start_y = bees_df_for_frame.xs(("back", "x")), bees_df_for_frame.xs(
        ("back", "y")
    )  # Coordinate of back of bee
    end_x = np.mean(
        [
            bees_df_for_frame.xs(("antennaebaseL", "x")),
            bees_df_for_frame.xs(("antennaebaseR", "x")),
        ]
    )  # X coord base antennae midpt
    end_y = np.mean(
        [
            bees_df_for_frame.xs(("antennaebaseL", "y")),
            bees_df_for_frame.xs(("antennaebaseR", "y")),
        ]
    )  # Y coord base antennae midpt
    return np.array([end_x - start_x, end_y - start_y])


def signed_angle_rad_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return angle (radians) between two vectors. Angle is returned between
    0 and +- pi."""
    return math.atan2(
        v1[0] * v2[1] - v1[1] * v2[0],
        v1[0] * v2[0] + v1[1] * v2[1],
    )


def angle_of_nestmate_rel_to_dancer(
    dancers_df_for_frame: pd.DataFrame, bees_df_for_frame: pd.DataFrame
) -> float:
    """Get the angle (radians) of a nestmate relative to the dancer in a frame."""
    dancers_vec = orientation_vector(dancers_df_for_frame)
    individuals_vec = orientation_vector(bees_df_for_frame) * -1
    return signed_angle_rad_between_vectors(dancers_vec, individuals_vec)  # Radians


def angle_of_bee_rel_to_up(bees_df_for_frame: pd.DataFrame) -> float:
    """Get the angle (radians) of a bee relative to directly upwards (gravity)."""
    up_vec = [0, 1]
    bees_vec = orientation_vector(bees_df_for_frame)
    return signed_angle_rad_between_vectors(up_vec, bees_vec)  # Radians


def antenna_vector(
    bees_df_for_frame: pd.DataFrame, side: str, antenna_len_str: str = "default"
):
    """Obtain the orientation vector of a bee's antenna (left or right) relative
    to 0 degrees East. If antennae tips are not available, returns nan."""
    start_x, start_y = bees_df_for_frame.xs(
        (f"antennaebase{side}", "x")
    ), bees_df_for_frame.xs((f"antennaebase{side}", "y"))
    # If antenna base is nan, then return None
    if math.isnan(start_x) or math.isnan(start_y):
        return np.nan
    if antenna_len_str == "mid_only":
        end_x, end_y = bees_df_for_frame.xs(
            (f"antennaemid{side}", "x")
        ), bees_df_for_frame.xs((f"antennaemid{side}", "y"))
    else:
        # Full only or default
        end_x, end_y = bees_df_for_frame.xs(
            (f"antennaetip{side}", "x")
        ), bees_df_for_frame.xs((f"antennaetip{side}", "y"))
        # If antenna tip is none, then try get coordinates of midpoint
        if (math.isnan(end_x) or math.isnan(end_y)) & (antenna_len_str == "default"):
            end_x, end_y = bees_df_for_frame.xs(
                (f"antennaemid{side}", "x")
            ), bees_df_for_frame.xs((f"antennaemid{side}", "y"))
    if math.isnan(end_x) or math.isnan(end_y):
        return np.nan
    return [end_x - start_x, end_y - start_y]


def s_s_angle(bee_df_for_frame: pd.DataFrame) -> float:
    """Return angle (radians) between left and right antennae scapes of bee in frame.
    If orientation of antennae cant be determined, returns nan."""
    l_antenna = antenna_vector(bee_df_for_frame, side="L")
    r_antenna = antenna_vector(bee_df_for_frame, side="R")
    if np.isnan(l_antenna).any() or np.isnan(r_antenna).any():
        return np.nan
    signed_s_s = signed_angle_rad_between_vectors(l_antenna, r_antenna)
    return np.abs(signed_s_s)


def angle_of_antennae_rel_to_body(bee_df_for_frame: pd.DataFrame, antenna_len_str: str):
    """Compute the angles (radians) of L and R antenna relative to body orientation."""
    bee_orientation = orientation_vector(bee_df_for_frame)
    l_antenna = antenna_vector(bee_df_for_frame, "L", antenna_len_str)
    r_antenna = antenna_vector(bee_df_for_frame, "R", antenna_len_str)
    if l_antenna is np.nan or r_antenna is np.nan:
        return np.nan, np.nan
    l_angle = signed_angle_rad_between_vectors(bee_orientation, np.array(l_antenna))
    r_angle = signed_angle_rad_between_vectors(bee_orientation, np.array(r_antenna))
    return l_angle, r_angle  # Radians


def get_time_scale(values: np.ndarray) -> np.ndarray:
    """Compute the time range indices where valid values occur in array
    (starts from 0)."""
    time_range = np.arange(values.shape[0])
    nan_mask = np.isnan(values) == False
    time_range = time_range[nan_mask]
    return time_range - time_range[0]


def backfill_nans(xy_df: pd.DataFrame) -> np.ndarray:
    """Backfill pandas datafame with x and y coordinates, replacing nan values with the
    next valid number in the array."""
    xy_backfilled = []
    num_to_backfill = 1
    for _, values in xy_df.iterrows():
        if math.isnan(values["x"]) is False and math.isnan(values["y"]) is False:
            xy_backfilled.append([values.values] * num_to_backfill)
            num_to_backfill = 1
        else:
            num_to_backfill += 1
    return np.concatenate(xy_backfilled).reshape((-1, 2)).astype(float)


def cumulative_euclid_dist(
    bee_df: pd.DataFrame, bodypart: str = "antennaebaseL"
) -> np.ndarray:
    """Compute the accumulative euclidean distance between specified body part across set of frames.
    Unknown positions are backfilled with the next valid number."""
    positions = bee_df.xs(bodypart).T.astype(float)
    # Backfill nan values with the next valid coordinates
    positions = backfill_nans(positions)
    # Compute distance travelled between consecutive frames
    dists = [
        math.dist(positions[i].tolist(), positions[i + 1].tolist())
        for i in range(positions.shape[0] - 1)
    ]
    return np.cumsum(dists)  # Accumulate distances


def cumulative_angular_displacement(angular_data_degrees: pd.Series) -> np.ndarray:
    """Compute the cumulated angular displacement of antennae. Nan values are filled with the
    mean displacement, except for leading or trailing nans."""
    # Returns nan from num->nan and nan->num, i.e. np.diff([1,2,np.nan,4,5]) -> [ 1., nan, nan,  1.]
    abs_displacements = np.abs(np.diff(angular_data_degrees))
    # Fill intermediate nan values with average displacement (beginning nans get filled with 0)
    data_series = pd.Series(angular_data_degrees.values)  # To reset index from 0
    abs_displacements[: data_series.first_valid_index()] = 0.0
    abs_displacements[data_series.last_valid_index() :] = 0.0
    abs_displacements[np.isnan(abs_displacements)] = np.nanmean(abs_displacements)
    return np.cumsum(abs_displacements)  # Accumulate total degrees 'travelled'


def compute_statistics_for_group(group: pd.DataFrame) -> pd.Series:
    """Compute circular statistics for binned group of antennae data (binned by angle
    to dancer)."""
    midpt_mean_rad = scipy.stats.circmean(
        group["antenna_midpoint_rad"], high=np.pi, low=-np.pi
    )
    midpt_std_rad = scipy.stats.circstd(
        group["antenna_midpoint_rad"], high=np.pi, low=-np.pi
    )
    s_s_mean_rad = scipy.stats.circmean(group["s_s_rad"], high=np.pi, low=-np.pi)
    s_s_std_rad = scipy.stats.circstd(group["s_s_rad"], high=np.pi, low=-np.pi)
    left_mean_rad = scipy.stats.circmean(group["l_antenna_rad"], high=np.pi, low=-np.pi)
    left_std_rad = scipy.stats.circstd(group["l_antenna_rad"], high=np.pi, low=-np.pi)
    right_mean_rad = scipy.stats.circmean(
        group["r_antenna_rad"], high=np.pi, low=-np.pi
    )
    right_std_rad = scipy.stats.circstd(group["r_antenna_rad"], high=np.pi, low=-np.pi)
    left_min_rad = group["l_antenna_rad"].min()
    left_max_rad = group["l_antenna_rad"].max()
    right_min_rad = group["r_antenna_rad"].max()  # Because right are -ve angles
    right_max_rad = group["r_antenna_rad"].min()  # Because right are -ve angles
    right_median_rad = group["r_antenna_rad"].median()
    left_median_rad = group["l_antenna_rad"].median()
    return pd.Series(
        {
            "midpt_mean_rad": midpt_mean_rad,
            "midpt_std_rad": midpt_std_rad,
            "midpt_mean_deg": np.rad2deg(midpt_mean_rad),
            "midpt_std_deg": np.rad2deg(midpt_std_rad),
            "s_s_mean_rad": s_s_mean_rad,
            "s_s_std_rad": s_s_std_rad,
            "s_s_mean_deg": np.rad2deg(s_s_mean_rad),
            "s_s_std_deg": np.rad2deg(s_s_std_rad),
            "l_antenna_mean_rad": left_mean_rad,
            "l_antenna_std_rad": left_std_rad,
            "l_antenna_mean_deg": np.rad2deg(left_mean_rad),
            "l_antenna_std_deg": np.rad2deg(left_std_rad),
            "r_antenna_mean_rad": right_mean_rad,
            "r_antenna_std_rad": right_std_rad,
            "r_antenna_mean_deg": np.rad2deg(right_mean_rad),
            "r_antenna_std_deg": np.rad2deg(right_std_rad),
            "l_antenna_min_rad": left_min_rad,
            "l_antenna_max_rad": left_max_rad,
            "l_antenna_min_deg": np.rad2deg(left_min_rad),
            "l_antenna_max_deg": np.rad2deg(left_max_rad),
            "r_antenna_min_rad": right_min_rad,
            "r_antenna_max_rad": right_max_rad,
            "r_antenna_min_deg": np.rad2deg(right_min_rad),
            "r_antenna_max_deg": np.rad2deg(right_max_rad),
            "l_antenna_median_rad": left_median_rad,
            "r_antenna_median_rad": right_median_rad,
            "l_antenna_median_deg": np.rad2deg(left_median_rad),
            "r_antenna_median_deg": np.rad2deg(right_median_rad),
        }
    )


def add_time_following(
    errors_df: pd.DataFrame, reduced_df: pd.DataFrame
) -> pd.DataFrame:
    """Add column to reduced dataframe containing the number of frames the nestmate was
    recorded following the dancer per waggle phase."""
    num_frames = errors_df.groupby("bee_id").apply(lambda x: x["frame"].max())
    num_frames.name = "num_frames_following"
    reduced_df = reduced_df.join(num_frames, on="bee_id")
    return reduced_df


def reduce_statistics_for_errors(errors_df: pd.DataFrame, nbins: int) -> pd.DataFrame:
    """Compute reduced statistics for the vector error for each bee."""

    def reduce(bee_df: pd.DataFrame):
        """Reduce data: mean and std of vector error, mean angle to dancer."""
        mean_vector_error_rad = scipy.stats.circmean(
            bee_df["vector_error_rad"], high=np.pi, low=-np.pi
        )
        mean_vector_error_std = scipy.stats.circstd(
            bee_df["vector_error_rad"], high=np.pi, low=-np.pi
        )
        mean_angle_to_dancer = scipy.stats.circmean(
            bee_df["angle_to_dancer_rad"], high=np.pi, low=-np.pi
        )
        final_vector_error_rad = bee_df["vector_error_rad"].iloc[-1]
        return (
            mean_vector_error_rad,
            mean_vector_error_std,
            mean_angle_to_dancer,
            final_vector_error_rad,
        )

    # Compute mean for each bee over all time points
    reduced_across_all_time_df = pd.DataFrame(
        errors_df.groupby("bee_id").apply(lambda x: reduce(x))
    )
    reduced_across_all_time_df = (
        pd.DataFrame(
            reduced_across_all_time_df[0].tolist(),
            index=reduced_across_all_time_df.index,
        )
        .reset_index()
        .rename(
            columns={
                0: "mean_rad",
                1: "std_rad",
                2: "angle_to_dancer_rad",
                3: "final_vector_rad",
            }
        )
    )
    # Add time spent following waggle phase
    reduced_across_all_time_df = add_time_following(
        errors_df, reduced_across_all_time_df
    )
    # Compute mean error for each bee over bins
    errors_df["bin_idx_of_angle_to_dancer_deg"], bins = pd.cut(
        errors_df["angle_to_dancer_deg"],
        bins=nbins,
        retbins=True,
        labels=range(nbins),
    )
    binned_dancer_angle_df = (
        errors_df.groupby("bin_idx_of_angle_to_dancer_deg")
        .apply(lambda group: reduce(group))
        .reset_index()
    )
    binned_dancer_angle_df = (
        pd.DataFrame(
            binned_dancer_angle_df[0].tolist(),
            index=binned_dancer_angle_df.index,
        )
        .reset_index()
        .drop(columns=3)
        .rename(
            columns={
                0: "mean_rad",
                1: "std_rad",
                2: "angle_to_dancer_rad",
            }
        )
    ).drop(columns=["index"])
    return reduced_across_all_time_df, binned_dancer_angle_df  # (num_bees, 3)


def calculate_angle_consensus(
    all_angles: np.ndarray, inlier_cutoff: float = np.pi / 4.0
):
    """Perform RANSAC to determine consensus angle (returned in radians, mapped
    between -pi and pi)."""
    assert all_angles.shape[0] >= 2, "Need more angles to perform RANSAC."
    regular_mean_rad = scipy.stats.circmean(all_angles, high=np.pi, low=-np.pi)
    all_angles = (all_angles + 2.0 * np.pi) % (2.0 * np.pi)  # Normalise angles [0, 2pi]
    sample_indices = np.arange(all_angles.shape[0])
    n_samples = all_angles.shape[0] * 4
    max_inliers, max_inlier_consensus_angle, inlier_indices = 0, None, None
    for _ in range(n_samples):
        samples = np.random.choice(sample_indices, size=2)
        consensus_angle = scipy.stats.circmean(all_angles[samples])
        differences0 = np.abs(all_angles - consensus_angle)
        differences1 = (2.0 * np.pi) - differences0
        inliers = (differences0 < inlier_cutoff) | (differences1 < inlier_cutoff)
        n_inliers = np.sum(inliers)
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            max_inlier_consensus_angle = consensus_angle
            inlier_indices = inliers

    consensus_angle = scipy.stats.circmean(all_angles[inlier_indices])
    consensus_angle = (consensus_angle + np.pi) % (2.0 * np.pi) - np.pi
    # print(
    #     "Reg mean:",
    #     np.round(np.degrees(regular_mean_rad), 2),
    #     "Consensus:",
    #     np.round(np.degrees(consensus_angle), 2),
    # )
    assert max_inliers != 0, "Could not find consensus angle."
    return consensus_angle, max_inliers


def compute_mean_across_waggle_runs(errors_df: pd.DataFrame, use_ransac: bool = False):
    """Compute mean error vector across consecutive waggle phases using
    the dataframe containing each time step. i.e. the update of the mean
    from the first and second, and second and third waggle runs."""
    # Create waggle run index column for each bee
    def get_bee_id_parent(bee_id):
        name_components = bee_id.split("_")  # i.e. "n9", "n9_1"
        parent_name = name_components[0]
        if len(name_components) == 1:
            return parent_name, 0
        return parent_name, int(name_components[1])

    errors_df["bee_id_parent"], errors_df["waggle_run_idx"] = zip(
        *errors_df["bee_id"].apply(get_bee_id_parent)
    )
    errors_df = errors_df.sort_values(by=["bee_id_parent", "waggle_run_idx"])
    # Perform RANSAC or regular mean to determine consensus angle successively
    # across waggle runs
    parent_to_consensus_angles = {}
    for name, group in errors_df.groupby("bee_id_parent"):
        combinations = group["waggle_run_idx"].unique()
        combinations = [combinations[: i + 1] for i in range(len(combinations))]
        parent_to_consensus_angles[name] = {}
        for comb in combinations:
            # print("Combination for", name, comb)
            angles_so_far = group[group["waggle_run_idx"].isin(comb)][
                "vector_error_rad"
            ].values
            if use_ransac:
                mean_angle, _ = calculate_angle_consensus(angles_so_far)
            else:
                mean_angle = scipy.stats.circmean(angles_so_far, high=np.pi, low=-np.pi)
            parent_to_consensus_angles[name][comb.max()] = mean_angle
    consec_waggle_runs_df = (
        pd.DataFrame(parent_to_consensus_angles)
        .T.reset_index()
        .rename(columns={"index": "bee_id"})
    )
    return consec_waggle_runs_df


def compute_circmean_over_windows(angles, window_size) -> np.ndarray:
    """Compute circular mean (between pi and -pi) over windows of certain size."""
    windows = np.lib.stride_tricks.sliding_window_view(
        angles, window_shape=(window_size,)
    )
    circ_mean_for_each_window = scipy.stats.circmean(
        windows, high=np.pi, low=-np.pi, axis=1
    )
    return circ_mean_for_each_window


def compute_predictability(counts_dict: dict) -> dict:
    """Compute the predictability index for all entries in given dict."""
    dur_to_predictability = {}
    for dur in counts_dict:
        if (
            counts_dict[dur]["RR"] + counts_dict[dur]["RL"] == 0
            or counts_dict[dur]["LL"] + counts_dict[dur]["LR"] == 0
        ):
            dur_to_predictability[dur] = -1
            continue
        tp_rr = counts_dict[dur]["RR"] / (
            counts_dict[dur]["RR"] + counts_dict[dur]["RL"]
        )
        tp_rl = counts_dict[dur]["RL"] / (
            counts_dict[dur]["RR"] + counts_dict[dur]["RL"]
        )
        tp_ll = counts_dict[dur]["LL"] / (
            counts_dict[dur]["LL"] + counts_dict[dur]["LR"]
        )
        tp_lr = counts_dict[dur]["LR"] / (
            counts_dict[dur]["LL"] + counts_dict[dur]["LR"]
        )
        predictability_index = 1 - (tp_rr * tp_rl + tp_ll * tp_lr)
        dur_to_predictability[dur] = predictability_index
    return dur_to_predictability
