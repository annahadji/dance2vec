"""Base central complex model class, Stone et al. (2017)."""
from typing import Tuple
import numpy as np
import scipy.special

import utils.rng

from configlib import config as C
import configlib
from . import weight_matrices as W

add_argument = configlib.add_group("CX options", prefix="cx")
add_argument(
    "--noise",
    default=0.0,
    type=float,
    help="Level of noise to add to layer outputs of CX.",
)
add_argument(
    "--cpu4_memory_gain", default=0.005, type=float, help="Rate of memory accumulation."
)


# Tuned parameters for properties of cells sigmoid function
TL2_SLOPE = 6.8
TL2_BIAS = 3.0
CL1_SLOPE = 3.0
CL1_BIAS = -0.5
TB1_SLOPE = 5.0
TB1_BIAS = 0.0
CPU4_SLOPE = 5.0
CPU4_BIAS = 2.5
CPU1_SLOPE = 5.0
CPU1_BIAS = 2.5
MOTOR_SLOPE = 1.0
MOTOR_BIAS = 3.0

# Proportion of input from CL1 versus TB1
PROP_CL1 = 0.667
PROP_TB1 = 1 - PROP_CL1


def noisy_sigmoid(
    input_v: np.ndarray, slope: float, bias: float, noise: float
) -> np.ndarray:
    """Pass cell activity vector through sigmoid function with given parameters.

    Args:
        input_v (np.ndarray): Input vector of cell activity.
        slope (float): Slope for sigmoid function.
        bias (float): Bias for sigmoid function.
        noise (float): Level of noise to add to sigmoid.

    Returns:
        np.ndarray: Output rate of firing, clipped between 0 and 1.
    """
    sigmoid_output = scipy.special.expit(input_v * slope - bias)
    if noise > 0:
        sigmoid_output += utils.rng.get_rng("cx").normal(scale=noise, size=len(input_v))
    return np.clip(sigmoid_output, 0, 1)


def decode_position(cpu4_reshaped: np.ndarray) -> Tuple[float, float]:
    """Decode position from sinusoid to polar coordinates.
    Amplitude is distance and the angle is angle from nest outwards pointing towards
    the bee. Without offset angle gives the home vector (between -np.pi and np.pi)."""
    signal = np.sum(cpu4_reshaped, axis=0)  # (8,)
    fund_freq = np.fft.fft(signal)[1]
    fund_freq_conj = np.conjugate(fund_freq)
    angle = -np.angle(fund_freq_conj)
    # angle = np.arctan2(fund_freq_conj.imag, fund_freq_conj.real)  # Angle of complex num
    distance = np.absolute(fund_freq)  # Magnitude of complex num
    return angle, distance


class CX:
    """Base central complex class."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self) -> None:
        """Central complex model from Stone et al (2017). In the base class, the homevector
        is centered around 0.5 and carries no representation of the outbound journey."""
        # Initialise chosen params of model
        self.cpu4_memory_gain = C["cx_cpu4_memory_gain"] * C["dt"]
        self.noise = C["cx_noise"]

        # TN and TL prefs (NB: TL2 hasnt been shifted to match cell index with their preference)
        self.tn_prefs = W.TN_PREFS  # Offset 45 degrees
        self.tl2_prefs = W.TL2_PREFS  # Sampled between 0 and 2pi, twice over

        # Initialise connection weights according to anatomy
        self.w_cl1_tb1 = W.W_CL1_TB1  # (16 CL1, 8 TB1)
        self.w_tb1_tb1 = W.generate_tb_tb_weights()  # (8 TB1, 8 TB1)
        self.w_tb1_cpu4 = W.W_TB1_CPU4  # (8 TB1, 16 CPU4)
        self.w_tn_cpu4 = W.W_TN_CPU4  # (2 TN2, 16 CPU4)
        self.w_tb1_cpu1a = W.W_TB1_CPU1A  # (8 TB1, 14 CPU1A
        self.w_tb1_cpu1b = W.W_TB1_CPU1B  # (8 TB1, 2 CPU1B)
        self.w_cpu4_cpu1a = W.W_CPU4_CPU1A  # (16 CPU4, 14 CPU1A)
        self.w_cpu4_cpu1b = W.W_CPU4_CPU1B  # (16 CPU4, 2 CPU1B)
        self.w_cpu1a_motor = W.W_CPU1A_MOTOR  # (14 CPU1A, 2 MOTOR)
        self.w_cpu1b_motor = W.W_CPU1B_MOTOR  # (2 CPU1B, 2 MOTOR)

    def get_flow(self, heading: float, velocity: np.ndarray) -> np.ndarray:
        """Calculate optic flow depending on preference angles of TN cells. [L, R].

        Args:
            heading (float): Current heading (polar coordinates, radians).
            velocity (np.ndarray): Velocity [X, Y] (cartesian coordinates), (2,).

        Returns:
            np.ndarray: Optic flow activity in L and R hemispheres, [L, R], (2,).
        """
        # Align the tn preference angles to the current heading, to get both
        # offset [X,Y] 'heading' unit vectors (+45 and -45 degrees).
        # Original Stone et al. code has + and then -
        # It depends how you interpret Figure 5G and Figure 3I
        # I've implemented it as below because of Figure 3I
        heading_xy_lr = np.array(
            [
                [
                    np.cos(heading - self.tn_prefs),
                    np.sin(heading - self.tn_prefs),
                ],  # Activity in L hemisphere (optic flow on R of insect)
                [
                    np.cos(heading + self.tn_prefs),
                    np.sin(heading + self.tn_prefs),
                ],  # Activity in R hemisphere (optic flow on L of insect)
            ]
        )

        # How much velocity components project onto preference directions
        flow = np.dot(heading_xy_lr, velocity)
        return flow  # (2,)

    def tn1_output(self, flow: np.ndarray) -> np.ndarray:
        """Return output of speed sensing TN1 cells. Output of 0.5 means stationary for TN1.

        Args:
            flow (np.ndarray): Optic flow for [L, R] (hemispheres), (2,).

        Returns:
            np.ndarray: TN cells output (2,) for [L, R].
        """
        # Linear transformation of the flow, TN1 cells are excited by backwards and forwards flow
        output = (1.0 - flow) / 2.0  # (2,)
        if self.noise > 0.0:
            output += utils.rng.get_rng("cx").normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)  # (2,)

    def tn2_output(self, flow: np.ndarray) -> np.ndarray:
        """Return output of speed sensing TN2 cells. Linearly sensitive to forwards motion only.

        Args:
            flow (np.ndarray): Optic flow for [L, R] hemispheres, (2,).

        Returns:
            np.ndarray: TN2 cells output (2,) for [L, R].
        """
        # get_flow() models TN2's preference angles already
        if self.noise > 0.0:
            flow += utils.rng.get_rng("cx").normal(scale=self.noise, size=flow.shape)
        return np.clip(flow, 0.0, 1.0)  # (2,)

    def tl2_output(self, heading: float) -> np.ndarray:
        """Return output of TL2 cells (16 inhibitory neurons encoding heading). Heading layer 1.

        Args:
            heading (float): Current heading (polar coords, radians, 0 to pi and 0 to -pi).

        Returns:
            np.ndarray: TL2 cells output (16,).
        """
        # Similarity of current heading to each TL2 preference heading
        tl2_input = np.cos(heading - self.tl2_prefs)  # cos(0) == 1.0
        return noisy_sigmoid(tl2_input, TL2_SLOPE, TL2_BIAS, self.noise)  # (16,)

    def cl1_output(self, tl2: np.ndarray) -> np.ndarray:
        """Return output of CL1 cells. Heading layer 2. Handles batched input.

        Args:
            tl2 (np.ndarray): TL2 cells output (heading layer 1), (16,).

        Returns:
            np.ndarray: CL1 cells output (16,).
        """
        # CL1 are inhibited by TL
        return noisy_sigmoid(-tl2, CL1_SLOPE, CL1_BIAS, self.noise)  # (16,)

    def tb1_output(self, cl1: np.ndarray, tb1: np.ndarray) -> np.ndarray:
        """Ring attractor state on the protocerebral bridge. Heading layer 3.
        This output will show most activity in the direction agent is not travelling,
        thus, we use 1 - tb1_output when it arborises with CPU4 and CPU1 cells.
        Handles batched inputs.

        Args:
            cl1 (np.ndarray): CL1 cells output (heading layer 2), (16,).
            tb1 (np.ndarray): TB1 cells output (mutually inhibitory TB1 connections), (8,).

        Returns:
            np.ndarray: TB1 cells output (8,).
        """
        # TB1 connections are inhibitory, constants scale for relative effect of CL1 vs TB1
        output = (PROP_CL1 * cl1 @ self.w_cl1_tb1) - (
            PROP_TB1 * tb1 @ W.W_TB1_TB1
        )  # (8,)
        return noisy_sigmoid(output, TB1_SLOPE, TB1_BIAS, self.noise)  # (8,)

    def cpu4_update(
        self, cpu4_mem: np.ndarray, tb1: np.ndarray, tn1: np.ndarray, tn2: np.ndarray
    ) -> np.ndarray:
        """CPU4 memory cells are clipped between 0 and 1 as they accumulate.
        CPU4[0-7] stores activity on the L hemisphere, which peaks when moving along the
            45 degrees R TN preference vector (see TN2 to CPU4 weights).
        CPU4[8-15] stores activity on the R hemisphere, which peaks when moving along the
            45 degrees L TN preference vector (see TN2 to CPU4 weights).

        Args:
            cpu4_mem (np.ndarray): CPU4 memory, (16,).
            tb1 (np.ndarray): TB1 cells output (heading layer 3), (8,).
            tn1 (np.ndarray): TN1 cells output (speed layer), (2,).
            tn2 (np.ndarray): TN2 cells output (speed layer), (2,).

        Returns:
            np.ndarray: CPU4 cells output clipped between 0 and 1, (16,).
        """
        # Negate TB1 sinusoid for memorising backwards motion, TB1 sinusoid is ihibitory
        mem_update = np.clip((0.5 - tn1) @ self.w_tn_cpu4, 0, 1) * (
            (1.0 - tb1) @ self.w_tb1_cpu4
        )  # (16,)
        mem_update -= 0.25 * tn2 @ self.w_tn_cpu4  # (16,)
        ## Above line is equivalent to:
        ## mem_update -= 0.5 * np.clip((0.5 - tn1) @ self.w_tn_cpu4, 0, 1)  # (16,)

        cpu4_mem += self.cpu4_memory_gain * mem_update
        return np.clip(cpu4_mem, 0.0, 1.0)  # (16,)

    def cpu4_output(self, cpu4_mem: np.ndarray) -> np.ndarray:
        """Population encoding of home vector. Handles batched input.

        Args:
            cpu4_mem (np.ndarray): CPU4 memory, (16,)

        Returns:
            np.ndarray: CPU4 cells output (16,).
        """
        return noisy_sigmoid(cpu4_mem, CPU4_SLOPE, CPU4_BIAS, self.noise)  # (16,)

    def cpu1a_output(self, tb1: np.ndarray, cpu4: np.ndarray) -> np.ndarray:
        """Return output of CPU1A cells (steering layer). Handles batched input.

        Args:
            tb1 (np.ndarray): TB1 cells output (heading layer 3), (8,).
            cpu4 (np.ndarray): CPU4 cells output (encoding of home vector), (16,).

        Returns:
            np.ndarray: CPU1A cells output (14,).
        """
        inputs = cpu4 @ self.w_cpu4_cpu1a * ((1.0 - tb1) @ self.w_tb1_cpu1a)  # (14,)
        return noisy_sigmoid(inputs, CPU1_SLOPE, CPU1_BIAS, self.noise)  # (14,)

    def cpu1b_output(self, tb1: np.ndarray, cpu4: np.ndarray) -> np.ndarray:
        """Return output of CPU1B cells (steering layer). Handles batched input.

        Args:
            tb1 (np.ndarray): TB1 cells output (heading layer 3).
            cpu4 (np.ndarray): CPU4 cells output (encoding of home vector).

        Returns:
            np.ndarray: CPU1B cells output (2,).
        """
        inputs = cpu4 @ self.w_cpu4_cpu1b * ((1.0 - tb1) @ self.w_tb1_cpu1b)  # (2,)
        return noisy_sigmoid(inputs, CPU1_SLOPE, CPU1_BIAS, self.noise)  # (2,)

    def cpu1_output(self, tb1: np.ndarray, cpu4: np.ndarray) -> np.ndarray:
        """Return output of CPU1 steering layer. Difference between the integrated path
        (excitatory CPU4 response) and current heading direction (inhibitory TB1 response).

        Args:
            tb1 (np.ndarray): TB1 cells output (heading layer 3), (8,).
            cpu4 (np.ndarray): CPU4 cells output (encoding of home vector), (16,).

        Returns:
            np.ndarray: CPU1 cells (a and b) activity, (16,).
        """
        cpu1a = self.cpu1a_output(tb1, cpu4)  # (14,)
        cpu1b = self.cpu1b_output(tb1, cpu4)  # (2,)
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])  # Reassemble cell indices, (16,)

    def motor_output(self, cpu1: np.ndarray) -> float:
        """Return whether agent should turn left or right, and by how much.

        Args:
            cpu1 (np.ndarray): CPU1 (a and b) cells output.

        Returns:
            float: Output rotation where sign indicates right (-) or left (+) turn,
                and ranges between -1 and 1.
        """
        cpu1a = cpu1[1:-1]  # CPU1a are indices 2-15, (14,)
        cpu1b = np.array([cpu1[-1], cpu1[0]])  # (2,)
        motor = cpu1a @ self.w_cpu1a_motor  # (2,)
        motor += cpu1b @ self.w_cpu1b_motor  # (2,)
        # Original models output*0.25 here to kill noise
        # This implementation uses a turn sharpness parameter in the agent
        return motor[0] - motor[1]

    def decode_cpu4(  # pylint: disable=no-self-use
        self, cpu4: np.ndarray
    ) -> Tuple[float, float]:
        """Decode the CPU4 sinusoids to get the estimated direction and distance from home.

        Args:
            cpu4 (np.ndarray): CPU4 cells output (encoding of home vector), (16,).

        Returns:
            Tuple[float, float]: Theta (direction) and r (amplitude of home vector).
        """
        cpu4_reshaped = cpu4.reshape(2, -1)  # (2, 8)
        # Shift CPU4 by +1 and -1 columns to realign the two CPU4 sinusoids, to cancel
        # 45 degree flow preference. When summed, a single sinusoid should point home
        cpu4_shifted = np.vstack(
            [np.roll(cpu4_reshaped[0], 1), np.roll(cpu4_reshaped[1], -1)]
        )
        return decode_position(cpu4_shifted)
