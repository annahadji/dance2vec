"""CX experiments for simulations recruit assimilating information."""
import datetime
import pathlib
import logging
import tempfile
import time
from tqdm import tqdm
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

import agents.recruit
import models.cx_recruit
import utils.rng
import utils.plotting
import utils.features

from configlib import config as C
import configlib

logger = logging.getLogger("SimulateAntennae")
logging.basicConfig(level=logging.INFO)

add_argument = configlib.add_group("Simulation options", prefix="")
add_argument(
    "experiment_name",
    default=datetime.datetime.now().strftime("%Y%m%d-%H%M"),
    type=str,
    help="Experiment name, default current datetime.",
)
add_argument(
    "--cx_model",
    default="CXRecruitHoloFly",
    type=str,
    choices=["CXRecruitHoloBee", "CXRecruitHoloFly"],
    help="Type of CX model to use in experiments.",
)
add_argument(
    "recruit_angle_to_gravity",
    default=None,
    type=int,
    help="Recruit's angle to gravity (degrees), going ccw from 0 N to 90 W, +- 180 S, -90 E.",
)
add_argument(
    "dancer_angle_to_gravity",
    default=None,
    type=int,
    help="Dancer's angle to gravity (degrees), going ccw from 0 N to 90 W, +- 180 S, -90 E.",
)
add_argument(
    "--max_antenna_pos",
    type=float,
    default=None,
    help="Max +ve valid angle of antenna relative to bees midline (degrees). Used if running \
        prefect simulation.",
)
add_argument(
    "--add_noise",
    action="store_true",
    help="Add noise to any simulated antennae positions.",
)
add_argument(
    "--simulation_time",
    default=3,
    type=float,
    help="Maximum simulation time (in seconds) per trial.",
)
add_argument(
    "--dt",
    type=float,
    default=0.01,
    help="Time resolution to sample. Default records at base time (samples once per second)",
)
add_argument(
    "--antennal_positioning_dataset",
    type=str,
    default=None,
    help="Path to real antennal positioning dataset. Defaults to None (i.e. no use). \
        If provided, will feed this data to the model instead of simulating positions.",
)
add_argument(
    "--bee_id",
    type=str,
    default=None,
    help="Index of bee to monitor if antennal positioning dataset is given. \
        If not specified, a random bee id will be visualised.",
)
add_argument(
    "--remove_antennae_input",
    action="store_true",
    help="Don't use antenna info. Only available when using real antennal positions.",
)
add_argument("--seed", default=1, type=int, help="Seed to set.")


NAME_TO_MODEL = {
    "CXRecruitHoloBee": (
        agents.recruit.RecruitHoloBee,
        models.cx_recruit.CXRecruitHoloBee,
    ),
    "CXRecruitHoloFly": (
        agents.recruit.RecruitHoloFly,
        models.cx_recruit.CXRecruitHoloFly,
    ),
}


def get_model_and_agent_subclass(model: str) -> Any:
    """Return uninstantiated agent and CX model subtype (e.g. "CXVectMem", "CXGPSteer")."""
    if model in NAME_TO_MODEL:
        return NAME_TO_MODEL[model]
    raise ValueError(f"Unknown CX model string passed {model}.")


# Valid angular positions that can be occupied around dancer
stepsize = np.radians(1)
pos_half = np.arange(0.0, np.pi + stepsize, stepsize)
neg_half = np.arange(-np.pi, 0.0, stepsize)
VALID_ANGLES_TO_DANCER = np.concatenate((pos_half, neg_half))
# Correspondings headings with respect to dancer for the above
# angles to the dancer (use only when direction of dancer == direction to gravity)
pos_half = np.arange(-np.pi, 0.0, stepsize)
neg_half = np.arange(0, np.pi + stepsize, stepsize)
HEADINGS_FOR_ANGLES_TO_DANCER = np.concatenate((pos_half, neg_half))


def generate_antennae_from_target(angles_to_dancer: np.ndarray, headings: np.ndarray):
    """Generate the left and right antenna positions for requested
    angles relative to dancer. Also returns the appropriate headings
    that the bee would be in to be in these angles relative to the dancer."""
    assert C["max_antenna_pos"] is not None, "Max antenna pos needs to be specified."
    # Define vaild antennae positions
    max_antenna_pos = np.radians(C["max_antenna_pos"])
    step = max_antenna_pos / VALID_ANGLES_TO_DANCER.shape[0]
    right_antennae = np.flip(np.arange(-max_antenna_pos, 0, step))
    left_antennae = np.flip(np.arange(0, max_antenna_pos, step))
    # Predictors - left and right antennae positions
    # Also return appropriate headings for bee at those angles
    left, right = [], []
    for angle in angles_to_dancer:
        closest_index = (np.abs(VALID_ANGLES_TO_DANCER - angle)).argmin()
        left.append(left_antennae[closest_index])
        right.append(right_antennae[closest_index])
    left, right = np.stack(left), np.stack(right)
    if C["add_noise"]:
        logger.info("Adding noise to R and L antenna position....")
        left += utils.rng.get_rng("antenna").vonmises(0.0, 10.0, left.shape[0])
        right += utils.rng.get_rng("antenna").vonmises(0.0, 10.0, right.shape[0])
        # headings += utils.rng.get_rng("antenna").vonmises(0.0, 10.0, headings.shape[0])
    return left, right, headings


def get_angles_to_dancer(
    recruits_headings: np.ndarray, dancers_headings: np.ndarray
) -> np.ndarray:
    """Compute the signed angle (radians) of the nestmate relative to the dancer
    from their headings."""
    target, source = dancers_headings.copy(), recruits_headings.copy()
    source = [utils.features.opposite_angle_quadrant(i) for i in source]
    angles = utils.features.signed_delta_angle(target, source)
    return -angles


def get_dataset_from_args():
    """Helper function to load desired real antenna data from bees,
    or synthesise a dataset from a valid angle index given."""
    assert (
        (C["dancer_angle_to_gravity"] is not None)
        and (C["recruit_angle_to_gravity"] is not None)
    ) or (C["antennal_positioning_dataset"] is not None)
    if C["antennal_positioning_dataset"] is None:
        # Generate antennae positions associated with angles we want to test
        logger.info(
            "Running simulation for %.1f deg dancer and %.1f deg recruit rel to gravity.",
            C["dancer_angle_to_gravity"],
            C["recruit_angle_to_gravity"],
        )
        num_iters = int(C["simulation_time"] * (1 / C["dt"]))
        dancers_headings = np.stack(
            [np.radians(C["dancer_angle_to_gravity"])] * num_iters
        )
        recruits_headings = np.stack(
            [np.radians(C["recruit_angle_to_gravity"])] * num_iters
        )
        angles_to_dancer = get_angles_to_dancer(recruits_headings, dancers_headings)
        (
            left_antennae_angles,
            right_antennae_angles,
            recruits_headings,  # Only passed to add noise
        ) = generate_antennae_from_target(angles_to_dancer, recruits_headings)
        #######################################################
        # np.savez(
        #     "model_sim_data_extra.npz",
        #     dancers_headings=dancers_headings,
        #     recruits_headings=recruits_headings,
        #     left=left_antennae_angles,
        #     right=right_antennae_angles,
        #     angles_to_dancer=angles_to_dancer,
        # )
        #######################################################
    else:
        logger.info(
            "Loading antennae data from %s",
            str(C["antennal_positioning_dataset"]),
        )
        antenna_df = pd.read_csv(C["antennal_positioning_dataset"])
        bee_id = C["bee_id"]
        if bee_id is None:
            bee_id = utils.rng.get_rng("bees").choice(antenna_df["bee_id"].unique())
        logger.info("Simulating input for bee %s...", str(bee_id))
        bee_df = antenna_df[antenna_df["bee_id"] == bee_id]
        dancers_headings = bee_df[
            "dancers_angle_to_gravity_rad"
        ].values  # 180 to 0 to -180
        angles_to_dancer = bee_df["angle_to_dancer_rad"].values  # 180 to 0 to -180
        left_antennae_angles = bee_df["l_antenna_rad"].values  # 180 to 0 to -180
        right_antennae_angles = bee_df["r_antenna_rad"].values  # 180 to 0 to -180
        recruits_headings = bee_df[
            "recruit_angle_to_gravity_rad"
        ].values  # 180 to 0 to -180

    if C["remove_antennae_input"]:
        logger.info("Removing antennae input (i.e. not using antennae info)...")
        left_antennae_angles[:] = np.radians(45)
        right_antennae_angles[:] = np.radians(-45)
    logger.info(
        "Mean dancers heading: %0.0f degrees",
        np.degrees(scipy.stats.circmean(dancers_headings, low=-np.pi, high=np.pi)),
    )
    logger.info(
        "Mean recuits heading: %0.0f degrees",
        np.degrees(scipy.stats.circmean(recruits_headings, low=-np.pi, high=np.pi)),
    )
    logger.info(
        "Mean angle to dancer: %0.0f degrees",
        np.degrees(scipy.stats.circmean(angles_to_dancer, low=-np.pi, high=np.pi)),
    )
    # # For these angles to dancer, get the assumed left and right antennae angles...
    # (
    #     left_antennae_angles,
    #     right_antennae_angles,
    #     recruits_headings,
    # ) = generate_antennae_from_target(angles_to_dancer, recruits_headings)
    return (
        dancers_headings,
        angles_to_dancer,
        left_antennae_angles,
        right_antennae_angles,
        recruits_headings,
    )


def compute_foodward_vector_error(
    dancers_headings: np.ndarray,
    decoded_pos: np.ndarray,
    angles_to_dancer: np.ndarray,
    save_path: pathlib.Path,
    bee_id: str = None,
) -> np.ndarray:
    """Compute the error (degrees) in the offset of the decoded vector
    to the intended signalled vector. Saves errors to given save path.

    If recruit is directed leftward of the food, this error is -ve.
    If recruit is directed rightward of the food, this error is +ve."""
    # Compute error with respect to mean waggle run angle
    mean_waggle_run_angle = scipy.stats.circmean(
        dancers_headings, low=-np.pi, high=np.pi
    )
    errors = np.stack(
        [
            utils.features.angle_between_angles(i, mean_waggle_run_angle)
            for i in decoded_pos[:, 0]
        ]
    )  # Radians

    logger.info("Mean run angle: %.2f", np.round(np.degrees(mean_waggle_run_angle)))
    logger.info(
        "Mean decoded angle: %.2f",
        np.round(
            np.degrees(scipy.stats.circmean(decoded_pos[:, 0], low=-np.pi, high=np.pi))
        ),
    )
    logger.info(
        "Mean error: %.2f",
        np.round(np.degrees(scipy.stats.circmean(errors, low=-np.pi, high=np.pi))),
    )
    if bee_id is not None:
        sub_save_path = save_path.parents[0] / "real_bee_dataset"
        sub_save_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            sub_save_path / f"errors_{bee_id}.npz",
            errors=errors,
            angle_to_dancer=angles_to_dancer,
        )
        logger.info("Errors saved to: %s", str(sub_save_path / f"errors_{bee_id}.npz"))
    else:
        np.savez(
            save_path / "errors.npz", errors=errors, angle_to_dancer=angles_to_dancer
        )
        logger.info("Errors saved to: %s", save_path / "errors.npz")
    return errors, mean_waggle_run_angle


def replay_activity_trial(config_hash: str) -> None:
    """Loop running simulation experiments of agent in environment.

    Args:
        config_hash (str): Generated hash of configuration file.
    """
    utils.rng.setup_rng(["agent", "cx", "bees"], [C["seed"], C["seed"], C["seed"]])

    # Save experiment configuration
    if C["bee_id"] is not None:
        config_hash += f"-{C['bee_id']}"
    run_path = pathlib.Path("logs", "recruits", C["experiment_name"], config_hash)
    run_path.mkdir(parents=True, exist_ok=True)
    configlib.save_config(str(run_path / "config.json"))
    start_time = time.time()

    # Initialise agent with CX model
    agent_cls, cx = get_model_and_agent_subclass(C["cx_model"])
    bee = agent_cls(cx, experiment_dt=C["dt"])

    # Generate antennae positions associated with angles we want to test
    (
        dancers_headings,
        angles_to_dancer,  # Relax using this when perfect simulation
        left_antennae_angles,
        right_antennae_angles,
        headings,
    ) = get_dataset_from_args()

    # Run simulation
    max_antenna_pos = np.radians(C["max_antenna_pos"]) if C["max_antenna_pos"] else None
    working_memories, decoded_pos = bee.replay_input(
        left_antennae_angles,
        right_antennae_angles,
        headings,
        max_antenna_pos=max_antenna_pos,
    )
    bee.save_logs(run_path)
    # Create and save error in foodward vector
    errors, waggle_run_angle = compute_foodward_vector_error(
        dancers_headings, decoded_pos, angles_to_dancer, run_path, C["bee_id"]
    )
    # Save images
    # logger.info("Generating images...")
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     for i in tqdm(range(working_memories.shape[0])):
    #         utils.plotting.rgb_img_array_from_recruits_antenna_data(
    #             i,
    #             decoded_pos,  # (angle, dist) where angle 0 - (-)180 - 0
    #             working_memories,
    #             angles_to_dancer,
    #             headings,
    #             errors,
    #             waggle_run_angle,
    #             left_antennae_angles,
    #             right_antennae_angles,
    #             bee.cx_logs["antenna_velocity_vector"],
    #             tmp_dir,
    #             hdelta=bee.cx_logs.get("hdelta"),
    #         )
    #     utils.plotting.save_image_arrays_as_video(tmp_dir, run_path, frame_rate=5)

    logger.info(
        "Simulation finished after %.3f s to %s",
        time.time() - start_time,
        str(run_path),
    )


if __name__ == "__main__":

    # Run trials
    CONFIG_HASH = configlib.parse()
    replay_activity_trial(CONFIG_HASH)
