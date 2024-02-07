"""Central complex model class for recruit."""
import numpy as np
import scipy.stats

import utils.rng
import utils.features

from . import cx_base
from . import weight_matrices as W


class CXRecruit(cx_base.CX):
    """Central complex abstract base class for interpreting dance."""

    def get_flow_using_each_antenna(
        self,
        heading: float,
        left_velocity: np.ndarray,
        right_velocity: np.ndarray,
    ) -> np.ndarray:
        """Calculate optic flow depending on preference angles of TN cells. [L, R].

        Args:
            heading (float): Current heading (polar coordinates, radians).
            left_velocity (np.ndarray): Left antenna velocity [X, Y], (2,).
            right_velocity (np.ndarray): Right antenna velocity [X, Y], (2,).

        Returns:
            np.ndarray: Optic flow activity in L and R hemispheres, [L, R], (2,).
        """
        heading_xy_l = np.array(
            [np.cos(heading - self.tn_prefs), np.sin(heading - self.tn_prefs)]
        )
        heading_xy_r = np.array(
            [np.cos(heading + self.tn_prefs), np.sin(heading + self.tn_prefs)]
        )
        # How much velocity components project onto preference directions
        l_flow = np.dot(heading_xy_l, left_velocity)
        r_flow = np.dot(heading_xy_r, right_velocity)
        return np.array([l_flow, r_flow])  # (2,)

    def midpt_antenna_vel_vec(  # pylint: disable=no-self-use
        self,
        heading: float,
        l_antenna_angle: float,
        r_antenna_angle: float,
        mapping: int,
    ):
        """Compute 'velocity' vector for antennal positioning as the midpoint of
        the two antennae mapped to some number of degrees around the nestmate."""
        # The antennae angles effectively map a range of 2*pi possible relative
        # positions to the dancer to pi/2 possible positions of the antennae
        # (theta). If the antennae are assumed to be pi/2 apart, it doesn't really
        # matter that there are two and it might be simplest to take their midpoint
        # as the 'antennae position', i.e., ranging from -pi/4 to +pi/4 with 0 when
        # the follower is behind the dancer.
        midpoint_angle = scipy.stats.circmean(
            [l_antenna_angle, r_antenna_angle], high=np.pi, low=-np.pi
        )
        # We want to invert that mapping - i.e. use 4*theta - and treat this (in the
        # standard PI model) as the direction component of the 'velocity' relative
        # to the heading of the follower. If 45 degrees, 4*theta; if 30 degrees, 6*theta
        midpoint_angle *= mapping
        midpoint_angle = (midpoint_angle + heading) % (2 * np.pi)  # Add heading
        # print("Midpt", np.degrees(midpoint_angle))
        return utils.features.polar2cartesian(midpoint_angle, 1)  # Return 'vel' vector

    def single_antenna_vel_vec(
        self, heading: float, antenna_angle: float, max_antenna_pos: float, mapping: int
    ):
        """Compute 'velocity' vector for antennal positioning as the single antennae
        positions (L and R) mapped to 360 degrees."""
        # e.g. centre antenna position between 0 and (-)90 to be between 45 and -45
        # To map between 180 and -180 (instead of 0 to 360)
        antenna_angle += max_antenna_pos / 2
        antenna_angle *= mapping
        antenna_angle = (antenna_angle + heading) % (2 * np.pi)  # Add heading
        return utils.features.polar2cartesian(antenna_angle, 1)


class CXRecruitHoloBee(CXRecruit):
    """Central complex class for interpreting dance."""

    # pylint: disable=too-many-instance-attributes

    def tl2_output(self, heading: np.ndarray) -> np.ndarray:
        """Dummy function."""
        return heading

    def cl1_output(self, tl2: np.ndarray) -> np.ndarray:
        """Dummy function."""
        return tl2

    def tb1_output(self, cl1: np.ndarray, tb1: np.ndarray = None) -> np.ndarray:
        """Simplified sinusoidal response to solar compass."""
        # In the simplified case, CL1 activity is the heading
        cardinal_dirs = np.linspace(0, 2 * np.pi, W.N_TB1, endpoint=False)
        return (1.0 + np.cos(np.pi + cardinal_dirs + cl1)) / 2.0

    def cpu4_update(
        self, cpu4_mem: np.ndarray, tb1: np.ndarray, tn1: np.ndarray
    ) -> np.ndarray:
        """Use TN1 activity only to update the CPU4."""
        mem_update = (0.5 - tn1 @ self.w_tn_cpu4) * (1.0 - tb1 @ self.w_tb1_cpu4)
        mem_update -= 0.5 * (0.5 - tn1 @ self.w_tn_cpu4)
        cpu4_mem += self.cpu4_memory_gain * mem_update
        return np.clip(cpu4_mem, 0.0, 1.0)  # (16,)


class CXRecruitHoloFly(CXRecruit):
    """Central complex class for interpreting dance using latest holonomic circuit"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lno1_prefs = W.LNO1_PREFS
        self.lno2_prefs = W.LNO2_PREFS
        self.w_lno_pfn = W.W_LNO_PFN
        self.w_epg_pfn = W.W_EPG_PFN
        self.w_pfnd_hdelta = W.W_PFNd_HDELTA
        self.w_pfnv_hdelta = W.W_PFNv_HDELTA
        self.hdetla_memory_gain = 0.005

    def get_flow(self, heading: float, velocity: np.ndarray) -> np.ndarray:
        """Calculate optic flow depending on preference angles of LNO cells. [L, R].

        Args:
            heading (float): Current heading (polar coordinates, radians).
            velocity (np.ndarray): Velocity [X, Y] (cartesian coordinates), (2,).

        Returns:
            np.ndarray: Optic flow activity in L and R hemispheres, [L, R], (2,).
        """
        lno1_heading_xy_lr = np.array(
            [
                [
                    np.cos(heading - self.lno1_prefs),
                    np.sin(heading - self.lno1_prefs),
                ],  # Activity in L hemisphere (optic flow on R of insect)
                [
                    np.cos(heading + self.lno1_prefs),
                    np.sin(heading + self.lno1_prefs),
                ],  # Activity in R hemisphere (optic flow on L of insect)
            ]
        )
        lno2_heading_xy_lr = np.array(
            [
                [
                    np.cos(heading - self.lno2_prefs),
                    np.sin(heading - self.lno2_prefs),
                ],
                [
                    np.cos(heading + self.lno2_prefs),
                    np.sin(heading + self.lno2_prefs),
                ],
            ]
        )

        # How much velocity components project onto preference directions
        lno1_flow = np.clip(np.dot(lno1_heading_xy_lr, velocity), 0, 1)
        lno2_flow = np.clip(np.dot(lno2_heading_xy_lr, velocity), 0, 1)
        return lno1_flow, lno2_flow  # (2,), (2,)

    def get_flow_using_each_antenna(
        self,
        heading: float,
        left_velocity: np.ndarray,
        right_velocity: np.ndarray,
    ) -> np.ndarray:
        """Calculate optic flow depending on preference angles of TN cells. [L, R].

        Args:
            heading (float): Current heading (polar coordinates, radians).
            left_velocity (np.ndarray): Left antenna velocity [X, Y], (2,).
            right_velocity (np.ndarray): Right antenna velocity [X, Y], (2,).

        Returns:
            np.ndarray: Optic flow activity in L and R hemispheres, [L, R], (2,).
        """
        lno1_heading_xy_l = np.array(
            [np.cos(heading - self.lno1_prefs), np.sin(heading - self.lno1_prefs)]
        )
        lno1_heading_xy_r = np.array(
            [np.cos(heading + self.lno1_prefs), np.sin(heading + self.lno1_prefs)]
        )
        lno2_heading_xy_l = np.array(
            [np.cos(heading - self.lno2_prefs), np.sin(heading - self.lno2_prefs)]
        )
        lno2_heading_xy_r = np.array(
            [np.cos(heading + self.lno2_prefs), np.sin(heading + self.lno2_prefs)]
        )
        # How much velocity components project onto preference directions
        lno1_l_flow = np.dot(lno1_heading_xy_l, left_velocity)
        lno1_r_flow = np.dot(lno1_heading_xy_r, right_velocity)
        lno1_flow = np.array([lno1_l_flow, lno1_r_flow])
        lno2_l_flow = np.dot(lno2_heading_xy_l, left_velocity)
        lno2_r_flow = np.dot(lno2_heading_xy_r, right_velocity)
        lno2_flow = np.array([lno2_l_flow, lno2_r_flow])
        return lno1_flow, lno2_flow  # (2,), (2,)

    def epg_output(self, heading: np.ndarray) -> np.ndarray:
        """Simplified sinusoidal response to solar compass."""
        # Single EPG neurons have a mixed, input–output, ‘dendritic’ terminal
        # in one wedge of the EB and an ‘axonal’ terminal in one glomerulus of the PB
        cardinal_dirs = np.linspace(0, 2 * np.pi, W.N_EPG, endpoint=False)
        return (1.0 + np.cos(cardinal_dirs + heading)) / 2.0

    def lno_output(self, flow: np.ndarray):
        """Return output of speed sensing LNO cells. Redundant function, returns
        LNO1 output, LNO2 output accordingly from given flow.

        LNO1 respond to egocentric forward movement, LNO2 respond to backward."""
        return flow[0], flow[1]  # (2,), (2,)

    def pfn_output(self, epg: np.ndarray, lno1: np.ndarray, lno2: np.ndarray):
        """Return output of PFN cells (d and v subtypes) in the PB. PFN activity
        represents a duplicated EPG bump but with the amplitude scaled by LNO.
        PFNd respond to forward motion, PFNv respond to backward."""
        # PFNd receive input from forward responding LNO1 activity
        pfnd = (lno1 @ self.w_lno_pfn) * (epg @ self.w_epg_pfn)
        # PFNv receive input from backward responding LNO2 activity
        pfnv = (lno2 @ self.w_lno_pfn) * (epg @ self.w_epg_pfn)  # (L, R)
        return pfnd, pfnv  # (16,), (16,)

    def hdelta_output(self, pfnd: np.ndarray, pfnv: np.ndarray) -> np.ndarray:
        """Return output of h delta b cells, which is the accurate travelling
        direction vector. Returns"""
        hdelta = pfnd @ self.w_pfnd_hdelta + pfnv @ self.w_pfnv_hdelta
        return hdelta  # (8,)

    def hdelta_memory_update(
        self, hdelta_mem: np.ndarray, hdelta: np.array
    ) -> np.ndarray:
        """Accumulate the h delta b cell activity using some memory gain."""
        return hdelta_mem + self.hdetla_memory_gain * hdelta  # (8,)

    def decode_hdelta(self, hdelta: np.ndarray):
        """Return travel direction angle from hdelta activity."""
        fund_freq = np.fft.fft(hdelta)[1]
        fund_freq_conj = np.conjugate(fund_freq)
        angle = -np.angle(fund_freq_conj)
        distance = np.absolute(fund_freq)
        return (angle, distance)  # Angle and distance
