"""Util plotting functions."""
import datetime
import os
import pathlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections
import matplotlib.colors

from . import features


def rgb_img_array_from_recruits_antenna_data(  # pylint: disable=too-many-arguments,too-many-statements
    i: int,
    decoded_pos: np.ndarray,
    working_memories: np.ndarray,
    angles_to_dancer: np.ndarray,
    headings: np.ndarray,
    decoded_vec_error: np.ndarray,
    mean_waggle_run_angle: float,
    left_antenna_angles: np.ndarray,
    right_antenna_angles: np.ndarray,
    antenna_velocity_vectors: np.ndarray,
    save_path: str = None,
    hdelta: np.ndarray = None,
):
    """Return rgb np image array for period of recruit following a dancer."""
    # colors = np.linspace(0, 1, working_memories.shape[0])
    # cmap = sns.color_palette("spring", as_cmap=True)
    plt.figure(dpi=200, figsize=(12, 4))
    ax1 = plt.subplot(141)
    ax2 = plt.subplot(142)
    ax3 = plt.subplot(143, projection="polar")
    ax4 = plt.subplot(144, projection="polar")
    # Antenna angles - these are the deflection from the recruit's heading
    # Default would plot w.r.t. East, so we must add on an extra 90 degrees to plot
    # w.r.t. North (gravity)
    left_antenna_angle = (
        left_antenna_angles[i] + headings[i] + np.radians(90)
    ) % np.radians(360)
    right_antenna_angle = (
        right_antenna_angles[i] + headings[i] + np.radians(90)
    ) % np.radians(360)
    ax1.quiver(
        [0],
        [0],
        np.cos(left_antenna_angle),
        np.sin(left_antenna_angle),
        pivot="tail",
        width=0.008,
        scale=4.0,
        headwidth=5,
        color="tab:pink",
        label="L antenna",
    )
    ax1.quiver(
        [0],
        [0],
        np.cos(right_antenna_angle),
        np.sin(right_antenna_angle),
        pivot="tail",
        width=0.008,
        scale=4.0,
        headwidth=5,
        color="tab:green",
        label="R antenna",
    )
    offset_heading = (headings[i] + np.radians(90)) % np.radians(360)
    ax1.quiver(
        [0],
        [0],
        np.cos(offset_heading),
        np.sin(offset_heading),
        pivot="tail",
        width=0.008,
        scale=4.0,
        headwidth=5,
        color="tab:blue",
        label="Heading (gravity)",
    )
    # If using single antenna, we have a left and right velocity vector
    if isinstance(antenna_velocity_vectors[i][0], tuple):
        for side_x, side_y in antenna_velocity_vectors[i]:
            flow_vel_angle = (
                features.cartesian2polar(side_x, side_y)[1] + np.radians(90)
            ) % np.radians(360)
            flow_vec = features.polar2cartesian(flow_vel_angle, 1)
            ax1.quiver(
                [0],
                [0],
                flow_vec[0],
                flow_vec[1],
                pivot="tail",
                width=0.008,
                scale=4.0,
                headwidth=5,
                alpha=0.8,
                color="tab:red",
                label="Flow vec",
            )
    else:
        # If using midpoint, we only have one velocity vector
        flow_vel_angle = (
            features.cartesian2polar(
                antenna_velocity_vectors[i][0], antenna_velocity_vectors[i][1]
            )[1]
            + np.radians(90)
        ) % np.radians(360)
        flow_vec = features.polar2cartesian(flow_vel_angle, 1)
        ax1.quiver(
            [0],
            [0],
            flow_vec[0],
            flow_vec[1],
            pivot="tail",
            width=0.008,
            scale=4.0,
            headwidth=5,
            alpha=0.8,
            color="tab:red",
            label="Flow vec",
        )
    run_angle = (mean_waggle_run_angle + np.radians(90)) % np.radians(360)
    ax1.quiver(
        [0],
        [0],
        np.cos(run_angle),
        np.sin(run_angle),
        pivot="tail",
        width=0.008,
        scale=4.0,
        headwidth=5,
        color="darkorange",
        alpha=0.8,
        label="Waggle angle",
    )
    ax1.grid(False)
    ax1.legend(loc="upper center", fontsize=9)
    ax1.set_title("Nestmate relative to gravity", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ####################
    # Working memory
    ax2.plot(working_memories[i].T, color="tab:blue", label="memory")
    # ax2.hlines(0.5, xmin=0, xmax=16, color="grey")
    ax2.set_ylim([0, 1.5])
    ax2.set_xlabel("Memory neuron index")
    ax2.set_ylabel("Accumulated activity")
    if hdelta is not None:
        ax2.plot(hdelta[i], color="lightblue", label="hdelta")
        ax2.legend()
    ####################
    # Accumulating decoded positions (difference from expected)
    ax3.scatter(decoded_vec_error[:i], range(decoded_vec_error[:i].shape[0]), s=1)
    ax3.set_ylim([0, working_memories.shape[0]])
    ax3.set_yticklabels([])
    ax3.set_theta_zero_location("N")
    ax3.set_thetamin(-180)
    ax3.set_thetamax(180)
    ax3.set_thetagrids([0, 90, 180, -90])
    ax3.spines["polar"].set_visible(False)
    error_degrees = np.round(np.degrees(decoded_vec_error[i]))
    title = (
        f"Deviation from mean waggle angle: {error_degrees}$^\circ$\n"
        + f"Intended: {int(np.degrees(mean_waggle_run_angle))}$^\circ$"
        + f"     Decoded: {int(np.degrees(decoded_pos[:, 0][i]))}$^\circ$"
    )
    ax3.set_title(title, fontsize=10)
    ####################
    # Angle of nestmate to dancer
    ax4.scatter(
        angles_to_dancer[:i],
        range(decoded_pos[:, 0][:i].shape[0]),
        s=2,
    )
    ax4.set_ylim([0, working_memories.shape[0]])
    ax4.set_yticklabels([])
    ax4.set_theta_zero_location("N")
    ax4.set_thetamin(-180)
    ax4.set_thetamax(180)
    ax4.set_thetagrids([0, 90, 180, -90])
    ax4.spines["polar"].set_visible(False)
    ax4.set_title("Angle to dancer\n", fontsize=10)
    ####################
    # Save figure
    plt.tight_layout()
    img_path = f"{save_path}/img{str(i).zfill(4)}.jpg"
    plt.savefig(img_path)
    plt.close()


def save_image_arrays_as_video(
    data_dir: pathlib.Path, save_dir: pathlib.Path, frame_rate: int = 30
) -> None:
    """Save a list of RGB images in np array format to a video output.
    Recording wont save if mp4 file with same timestamp exists in dir already."""
    # Save output
    pathlib.Path.mkdir(save_dir, parents=True, exist_ok=True)
    time_now = datetime.datetime.now()
    timestamp = f"{time_now.hour}{time_now.minute}{time_now.second}"
    os.system(
        f"ffmpeg -r {frame_rate} -i {data_dir}/img%04d.jpg -vcodec libx264 -crf 25  \
            -pix_fmt yuv420p {save_dir}/recording{timestamp}.mp4  >/dev/null 2>&1"
    )
