"""Classes for agents with CX models that can comply with use antennal activity to
charge a vector."""
import abc
import logging
import pathlib
import numpy as np

import models.cx_recruit
import models.weight_matrices as W

from configlib import config as C
import configlib

logger = logging.getLogger("Recruit")
logging.basicConfig(level=logging.INFO)

add_argument = configlib.add_group("Recruit options", prefix="recruit")
add_argument(
    "--antenna_flow_method",
    default="midpoint",
    type=str,
    choices=["midpoint", "left_only", "right_only", "single_antennae"],
    help="Type of CX model to use in experiments.",
)
add_argument(
    "--dont_clip_antenna_angles",
    action="store_true",
    help="Whether or not to clip \
        antennae angles if they exceed the maximum antennae position specified.",
)

MIDPOINT = "midpoint"
LEFT_ONLY = "left_only"
RIGHT_ONLY = "right_only"
SINGLE_ANTENNAE = "single_antennae"


class RecruitBase:
    """Recruit abstract base class."""

    def __init__(
        self, cx_model: models.cx_recruit.CXRecruit, experiment_dt: float = 1.0
    ):
        self.cx = cx_model()  # pylint: disable=invalid-name
        self.dt = experiment_dt  # pylint: disable=invalid-name
        self.cx_logs = {}
        self.antenna_flow_method = C["recruit_antenna_flow_method"]
        self.clip_antenna_angles = not C["recruit_dont_clip_antenna_angles"]

        self.antenna_flow = None
        self.antenna_velocity_vector = None
        self.working_memory = None
        self.decoded_pos = None  # Decoded as (angle, dist)
        self.log_keys = None

    @abc.abstractmethod
    def update_cells(
        self,
        heading: float,
        l_antenna_angle: float,
        r_antenna_angle: float,
        max_antenna_pos: float,
    ) -> float:
        """Generate activity for all cells, based on previous activity and current motion.
        Returns memory directed turn from CX, according to state of agent.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def compute_and_set_antenna_flow(
        self,
        heading: float,
        l_antenna_angle: float,
        r_antenna_angle: float,
        max_antenna_pos: float,
    ) -> None:
        """Compute and set antenna flow.

        Args:
            heading (float): Current heading (polar coordinates).
            l_antenna_angle (float): Current angle of left antenna relative to midline (rad).
            r_antenna_angle (float): Current angle of right antenna relative to midline (rad).
            max_antenna_pos (float): Max position of antenna beyond which clipping.
        """
        if self.clip_antenna_angles:
            r_antenna_angle = np.clip(r_antenna_angle, -max_antenna_pos, 0)
            l_antenna_angle = np.clip(l_antenna_angle, 0, max_antenna_pos)
        mapping = 2 * np.pi / max_antenna_pos

        if self.antenna_flow_method == SINGLE_ANTENNAE:
            # Using input integrating using both positions of antennae
            self.antenna_velocity_vector = (
                self.cx.single_antenna_vel_vec(
                    heading, l_antenna_angle, -max_antenna_pos, mapping
                ),
                self.cx.single_antenna_vel_vec(
                    heading, r_antenna_angle, max_antenna_pos, mapping
                ),
            )
            self.antenna_flow = self.cx.get_flow_using_each_antenna(
                heading,
                self.antenna_velocity_vector[0],
                self.antenna_velocity_vector[1],
            )
        else:
            if self.antenna_flow_method == MIDPOINT:
                self.antenna_velocity_vector = self.cx.midpt_antenna_vel_vec(
                    heading, l_antenna_angle, r_antenna_angle, mapping=mapping
                )
            elif self.antenna_flow_method == LEFT_ONLY:
                self.antenna_velocity_vector = self.cx.single_antenna_vel_vec(
                    heading, l_antenna_angle, -max_antenna_pos, mapping
                )
            elif self.antenna_flow_method == RIGHT_ONLY:
                self.antenna_velocity_vector = self.cx.single_antenna_vel_vec(
                    heading, r_antenna_angle, max_antenna_pos, mapping
                )
            self.antenna_flow = self.cx.get_flow(heading, self.antenna_velocity_vector)

    def replay_input(
        self,
        l_antenna_angles: np.ndarray,
        r_antenna_angles: np.ndarray,
        headings: np.ndarray,
        max_antenna_pos: np.ndarray,
    ):
        """Replay antennal input to CX and accumulate the working memory and decoded position.

        Args:
            heading (float): Current heading (polar coordinates).
            l_antenna_angle (float): Current angle of left antenna relative to midline (rad).
            r_antenna_angle (float): Current angle of right antenna relative to midline (rad).
            max_antenna_pos (float): Max position of antenna beyond which clipping.
        """
        num_iters = len(l_antenna_angles)
        for i in range(num_iters):
            self.update_cells(
                headings[i], l_antenna_angles[i], r_antenna_angles[i], max_antenna_pos
            )
        working_memories = np.stack(self.cx_logs.get("working_memory"))
        decoded_pos = np.stack(self.cx_logs.get("decoded_pos"))
        return working_memories, decoded_pos

    def log_cx(self) -> None:
        """Log current states of CX cells. Note that the motor cells are logged before
        the turn sharpness is applied."""
        for key in self.log_keys:
            self.cx_logs.setdefault(key, []).append(getattr(self, key))

    def save_logs(self, experiment_path: pathlib.Path) -> None:
        """Save logs of agent to given directory path."""
        pathlib.Path.mkdir(experiment_path, parents=True, exist_ok=True)
        if self.cx_logs is not None:
            np.savez(experiment_path / "cx_events", **self.cx_logs)


class RecruitHoloBee(RecruitBase):
    """Class representative of bee agent, with CXRecruit model."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Agent keeps the state of CX cells
        # Compass cells
        self.tl2, self.cl1 = None, None
        self.tb1 = np.zeros(W.N_TB1)
        # Antenna cells
        self.tn1 = None
        # Working memory (non activated homevector) and vector memory initialised with 0.5
        self.working_memory = np.ones(W.N_CPU4) * 0.5
        self.vector_memory = np.ones(W.N_VMEM) * 0.5
        # CPU4 cells decoded into (angle, distance)
        self.cpu4 = None
        self.motor, self.cpu1 = None, None

        # CX cells to log
        self.log_keys = [
            "tl2",
            "cl1",
            "tb1",
            "tn1",
            "antenna_flow",
            "antenna_velocity_vector",
            "working_memory",
            "cpu4",
            "vector_memory",
            "decoded_pos",
            "motor",
        ]

    def update_cells(
        self,
        heading: float,
        l_antenna_angle: float,
        r_antenna_angle: float,
        max_antenna_pos: float,
    ) -> float:
        """Generate activity for all cells, based on previous activity and current motion.

        Args:
            heading (float): Current heading (polar coordinates).
            l_antenna_angle (float): Current angle of left antenna relative to midline (rad).
            r_antenna_angle (float): Current angle of right antenna relative to midline (rad).
            max_antenna_pos (float): Max position of antenna beyond which clipping.
        """
        self.compute_and_set_antenna_flow(
            heading, l_antenna_angle, r_antenna_angle, max_antenna_pos
        )
        self.tn1 = self.cx.tn1_output(self.antenna_flow)

        self.tl2 = self.cx.tl2_output(heading)
        self.cl1 = self.cx.cl1_output(self.tl2)
        self.tb1 = self.cx.tb1_output(self.cl1, self.tb1)

        self.working_memory = self.cx.cpu4_update(
            self.working_memory, self.tb1, self.tn1
        )

        self.cpu4 = self.cx.cpu4_output(self.working_memory)
        self.decoded_pos = self.cx.decode_cpu4(self.cpu4)

        self.cpu1 = self.cx.cpu1_output(self.tb1, self.cpu4)
        self.motor = self.cx.motor_output(self.cpu1)
        self.log_cx()


class RecruitHoloFly(RecruitBase):
    """Class representative of agent with latest holonomic circuit in the fly, with
    CXRecruitHoloFly model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Agent keeps the state of CX cells
        # Compass cells
        self.epg = np.zeros(W.N_EPG)
        # Antenna cells
        self.lno1, self.lno2 = None, None
        # Working memory (non activated homevector)
        self.pfnd = np.ones(W.N_PFNd) * 0.5
        self.pfnv = np.ones(W.N_PFNv) * 0.5
        self.hdelta = None
        self.working_memory = np.ones(W.N_HDELTA) * 0.0

        # CX cells to log
        self.log_keys = [
            "epg",
            "lno1",
            "lno2",
            "antenna_flow",
            "antenna_velocity_vector",
            "working_memory",
            "pfnd",
            "pfnv",
            "hdelta",
            "decoded_pos",
        ]

    def update_cells(
        self,
        heading: float,
        l_antenna_angle: float,
        r_antenna_angle: float,
        max_antenna_pos: float,
    ):
        """Generate activity for all cells, based on previous activity and current motion.

        Args:
            heading (float): Current heading (polar coordinates).
            l_antenna_angle (float): Current angle of left antenna relative to midline (rad).
            r_antenna_angle (float): Current angle of right antenna relative to midline (rad).
            max_antenna_pos (float): Max position of antenna beyond which clipping.
        """
        self.compute_and_set_antenna_flow(
            heading, l_antenna_angle, r_antenna_angle, max_antenna_pos
        )
        self.lno1, self.lno2 = self.cx.lno_output(self.antenna_flow)
        self.epg = self.cx.epg_output(heading)
        self.pfnd, self.pfnv = self.cx.pfn_output(self.epg, self.lno1, self.lno2)
        self.hdelta = self.cx.hdelta_output(self.pfnd, self.pfnv)
        self.working_memory = self.cx.hdelta_memory_update(
            self.working_memory, self.hdelta
        )
        self.decoded_pos = self.cx.decode_hdelta(self.working_memory)
        self.log_cx()
