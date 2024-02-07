"""Helper functions organising data to visuailse."""
import copy
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


def get_bee_to_colour_dict(bee_ids: list) -> dict:
    clrs = sns.color_palette("husl", n_colors=len(bee_ids))
    bee_to_colour = {}
    for i, col in enumerate(clrs):
        bee_to_colour[bee_ids[i]] = col
    return bee_to_colour


def obtain_consective_chunks(indices, df_to_split):
    """Return consecutive chunks of a dataframe according to consecutive sections
    of given indices."""
    split_at = np.where(pd.DataFrame(indices).diff(1) > 1.0)[0]
    return np.split(df_to_split, split_at)


def circ_scatter(angles, radial_base=1, radial_interval=0.1, bin_size=5):
    """Given a series of angles, return a list of associated radii, such
    that each duplicated angle gets an increased radius. Specifically
    for stupid polar scatter plots.

    Credit to Robert Mitchell.

    Args:
        angles (np.ndarray): The anglar data (degrees).
        bins: Existing bins used to bin data.
        radial_bases: The existing radial distances accumulated so th
            previously binned angular data.
        radial base: The base radius of angles not yet found in binned data.
        radial_interval: The radial distance added with each duplicate.
    """
    bins = np.arange(-181, 181, bin_size)
    inds = np.digitize(angles, bins, right=True)  # Index of each angle into bins
    binned_angles = bins[inds]
    unique, counts = np.unique(binned_angles, return_counts=True)
    radii = []
    angle_out = []
    for idx in range(len(unique)):
        radius = radial_base
        for c in range(counts[idx]):
            # For each instance of this unique value, increment radius
            # and add binned angle back to the output list.
            radii.append(radius)
            angle_out.append(unique[idx])
            radius += radial_interval
    return radii, angle_out, bins


def add_to_circ_scatter(
    angles, bins, prev_angle_out, radial_bases, radial_base, radial_interval=0.1
):
    """Given another series angles and existing bins, return the list of associated
    radii and binned angles such that radii also reflect duplicates from previous
    circular scatter data.

    Args:
        angles (np.ndarray): The anglar data (degrees).
        bins: Existing bins used to bin data.
        prev_angle_out (np.ndarray): Previous binned angular data (degrees).
        radial_bases: The existing radial distances accumulated so th
            previously binned angular data.
        radial base: The base radius of angles not yet found in binned data.
        radial_interval: The radial distance added with each duplicate.
    """
    inds = np.digitize(angles, bins, right=True)
    binned_angles = bins[inds]
    unique, counts = np.unique(binned_angles, return_counts=True)
    radii, angle_out = [], []
    for idx in range(len(unique)):
        # Get the radial base for this bin that might exist already
        angle_idx = np.where(prev_angle_out == unique[idx])[0]
        radius = radial_base
        if len(angle_idx) != 0:
            radius = np.max(np.array(radial_bases)[angle_idx]) + radial_interval
        for c in range(counts[idx]):
            # For each instance of this unique value, increment radius
            radii.append(radius)
            angle_out.append(unique[idx])
            radius += radial_interval

    # Combine these angles with previous angles (to be fed back if necessary)
    angles_combined, radii_combined = copy.deepcopy(angle_out), copy.deepcopy(radii)
    for i, ang in enumerate(prev_angle_out):
        if ang not in angles_combined:
            angles_combined.append(ang)
            radii_combined.append(radial_bases[i])

    return radii, angle_out, radii_combined, angles_combined


def circ_scatter_reduce(
    angles_around_dancer,
    angles_feature,
    bin_size=5,
    lower_bin_range=-181,
    upper_bin_range=181,
    circmean_high_low=(np.pi, -np.pi),
):
    """Computes circular mean of a feature of antennal position
    of nestmates in certain orientations relative to dancer (returns degrees).

    Args:
        angles_around_dancer (np.ndarray): Angle of nestmates relative to dancer.
        angles_feature (np.ndarray): Array of feature of angles (in degrees).
        bin_size (int, optional): Bin size (degrees) to define groups in certain
            orientations relative to dancer. Defaults to 5 degrees.
        lower_bin_range (int, optional): Lower bin boundary for binning orientations.
        upper_bin_range (int, optional): Upper bin boundary for binning orientations.
        circmean_high_low ((float, float)): High and low boundaries for mean calc.
    """
    bins = np.arange(
        lower_bin_range, upper_bin_range, bin_size
    )  # (180 + 180 / bin_size,)
    bin_idxs = np.digitize(
        angles_around_dancer, bins, right=True
    )  # Index of each angle into a bin
    unique_idxs = np.unique(bin_idxs)
    mean_radii, std_radii, binned_angle_around_dancer = [], [], []
    for bin_idx in unique_idxs:
        # Gather scapes in this group and compute mean and std
        indices_in_bin = np.where(bin_idxs == bin_idx)[0]
        angles = angles_feature.values[indices_in_bin]
        angles_rad = np.deg2rad(angles)
        mean_radius = scipy.stats.circmean(
            angles_rad, high=circmean_high_low[0], low=circmean_high_low[1]
        )
        mean_radius = np.rad2deg(mean_radius)
        std_radius = scipy.stats.circstd(
            angles_rad, high=circmean_high_low[0], low=circmean_high_low[1]
        )
        std_radius = np.rad2deg(std_radius)
        mean_radii.append(mean_radius)
        std_radii.append(std_radius)
        binned_angle_around_dancer.append(bins[bin_idx])
    return np.array(mean_radii), np.array(std_radii), binned_angle_around_dancer
