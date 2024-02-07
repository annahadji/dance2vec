"""Script defining all weight matrices used in models."""
import numpy as np

# Number of different cell types
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN1 = 2
N_TN2 = 2
N_CPU4 = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B
N_VMEM = 16
N_EPG = 8
N_PFNd = 16
N_PFNv = 16
N_HDELTA = 8


def generate_tb_tb_weights() -> np.ndarray:
    """Return weights mapping mutually inhibitory interconnections between TB1 neurons.

    Returns:
        np.ndarray: TB to TB weight matrix (8, 8).
    """
    weight_matrix = np.zeros([N_TB1, N_TB1])  # (8, 8)
    sinusoid = -(np.cos(np.linspace(0, 2 * np.pi, N_TB1, endpoint=False)) - 1) / 2
    for i in range(N_TB1):
        values = np.roll(sinusoid, i)
        weight_matrix[i, :] = values
    return weight_matrix


################ Connections in all models ################
# TN cell preferences, 45 degrees offset from front / back
# Aids with the model with holonomic motion
TN_PREFS = np.pi / 4
# TL2 preferences
TL2_PREFS = np.tile(np.linspace(0, 2 * np.pi, N_TB1, endpoint=False), 2)  # (16,)
# CL1 -> TB1
W_CL1_TB1 = np.tile(np.eye(N_TB1), reps=2).T  # (16, 8)
# Self-inhibition TB1 -> TB1
W_TB1_TB1 = generate_tb_tb_weights()  # (8, 8)
# TB1 -> CPU4, current head direction input to CPU4 accumulator
W_TB1_CPU4 = np.tile(np.eye(N_TB1), reps=(1, 2))  # (8, 16)
# TN -> CPU4, each L and R TN2 neuron excites one half of each CPU4 accumulator
W_TN_CPU4 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)  # (2, 16)
# TB1 -> CPU1A
W_TB1_CPU1A = np.tile(np.eye(N_TB1), reps=(2, 1))[1 : N_CPU1A + 1, :].T  # (8, 14)
# TB1 -> CPU1A
W_TB1_CPU1B = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0]]  # Cell indices 8, 9
).T  # (8, 2)
# CPU4 -> CPU1A
W_CPU4_CPU1A = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
).T  # (16, 14)
# CPU4 -> CPU1B
W_CPU4_CPU1B = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cell index 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # cell index 9
    ]
).T  # (16, 2)
# CPU1A -> MOTOR
W_CPU1A_MOTOR = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    ]
).T  # (14 CPU1A, 2 MOTOR)
# CPU1A -> MOTOR
W_CPU1B_MOTOR = np.array([[0, 1], [1, 0]]).T  # (2 CPU1B, 2 MOTOR)


################ Connections in variations ################
# CPU4 -> PONTINE
W_CPU4_PONTINE = np.eye(N_CPU4)  # (16, 16)
# PONTINE -> CPU1A
W_PONTINE_CPU1A = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
    ]
).T  # (16, 14)
# PONTINE -> CPU1B
W_PONTINE_CPU1B = np.array(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 9
    ]
).T  # (16, 2)
# VECTOR MEMORY -> PONTINE, but possibly could be implemented in two ways, TODO
W_VECTORMEM_PONTINE = np.eye(N_VMEM)  # (16, 16)
# VECTOR MEMORY -> CPU1A
W_VECTMEM_CPU1A = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
    ]
).T  # (16, 14)
# VECTOR MEMORY -> CPU1B
W_VECTMEM_CPU1B = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
    ]
).T  # (16, 2)

# EPG -> PFN
W_EPG_PFN = np.tile(np.eye(N_EPG), reps=(1, 2))  # (8, 16)
# LNO prefs
LNO1_PREFS = np.pi / 4
LNO2_PREFS = 3 * np.pi / 4
# LNO -> PFN, each L and R LNO1 neuron excites one half of each PFN subtype
W_LNO_PFN = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)  # (2, 16)
# PFN -> HDELTA
W_PFNdl_HDELTA = np.roll(np.eye(8), 1, axis=1)  # (8, 8)
W_PFNdr_HDELTA = np.roll(np.eye(8), -1, axis=1)  # (8, 8)
W_PFNd_HDELTA = np.vstack((W_PFNdl_HDELTA, W_PFNdr_HDELTA))  # (16, 8)
W_PFNvl_HDELTA = np.roll(np.eye(8), 3, axis=1)  # (8, 8)
W_PFNvr_HDELTA = np.roll(np.eye(8), -3, axis=1)  # (8, 8)
W_PFNv_HDELTA = np.vstack((W_PFNvl_HDELTA, W_PFNvr_HDELTA))  # (16, 8)
