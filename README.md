# Simulating and analysing the movement of antennae of nestmates during the honeybee waggle dance :honeybee:

## Running the assimilation model :brain:

```
[17:56:23] 🚀 dance2vec $ python simulate_antenna.py --help
usage: simulate_antenna.py [-h] [--cx_noise CX_NOISE] [--cx_cpu4_memory_gain CX_CPU4_MEMORY_GAIN] [--recruit_antenna_flow_method {midpoint,left_only,right_only,single_antennae}] [--recruit_dont_clip_antenna_angles]
                           [--experiment_name EXPERIMENT_NAME] [--cx_model {CXRecruitHoloBee,CXRecruitHoloFly}] [--recruit_angle_to_gravity RECRUIT_ANGLE_TO_GRAVITY] [--dancer_angle_to_gravity DANCER_ANGLE_TO_GRAVITY]
                           [--max_antenna_pos MAX_ANTENNA_POS] [--add_noise] [--simulation_time SIMULATION_TIME] [--dt DT] [--antennal_positioning_dataset ANTENNAL_POSITIONING_DATASET] [--bee_id BEE_ID]
                           [--remove_antennae_input] [--seed SEED]

Configuration for experiments. Taken from https://github.com/nuric/pix2rule

optional arguments:
  -h, --help            show this help message and exit

CX options:

  --cx_noise CX_NOISE   Level of noise to add to layer outputs of CX. (default: 0.0)
  --cx_cpu4_memory_gain CX_CPU4_MEMORY_GAIN
                        Rate of memory accumulation. (default: 0.005)

Recruit options:

  --recruit_antenna_flow_method {midpoint,left_only,right_only,single_antennae}
                        Type of CX model to use in experiments. (default: midpoint)
  --recruit_dont_clip_antenna_angles
                        Whether or not to clip antennae angles if they exceed the maximum antennae position specified. (default: False)

Simulation options:

  --experiment_name EXPERIMENT_NAME
                        Experiment name, default current datetime. (default: 20240207-1756)
  --cx_model {CXRecruitHoloBee,CXRecruitHoloFly}
                        Type of CX model to use in experiments. (default: CXRecruitHoloFly)
  --recruit_angle_to_gravity RECRUIT_ANGLE_TO_GRAVITY
                        Recruit's angle to gravity (degrees), going ccw from 0 N to 90 W, +- 180 S, -90 E. (default: None)
  --dancer_angle_to_gravity DANCER_ANGLE_TO_GRAVITY
                        Dancer's angle to gravity (degrees), going ccw from 0 N to 90 W, +- 180 S, -90 E. (default: None)
  --max_antenna_pos MAX_ANTENNA_POS
                        Max +ve valid angle of antenna relative to bees midline (degrees). Used if running prefect simulation. (default: None)
  --add_noise           Add noise to any simulated antennae positions. (default: False)
  --simulation_time SIMULATION_TIME
                        Maximum simulation time (in seconds) per trial. (default: 3)
  --dt DT               Time resolution to sample. Default records at base time (samples once per second) (default: 0.01)
  --antennal_positioning_dataset ANTENNAL_POSITIONING_DATASET
                        Path to real antennal positioning dataset. Defaults to None (i.e. no use). If provided, will feed this data to the model instead of simulating positions. (default: None)
  --bee_id BEE_ID       Index of bee to monitor if antennal positioning dataset is given. If not specified, a random bee id will be visualised. (default: None)
  --remove_antennae_input
                        Don't use antenna info. Only available when using real antennal positions. (default: False)
  --seed SEED           Seed to set. (default: 1)
```

## Analysing the data :chart_with_upwards_trend:

```
(.venv) [17:31:50] 🚀 roslin-2022-analysis $ python build_dataset.py --help
usage: build_dataset.py [-h] [--build_errors_ds] [--data_path DATA_PATH] [--file_out FILE_OUT] [--nbins NBINS] [--antenna_len_str {default,mid_only,full_only}]

Build antennal positioning dataset.

optional arguments:
  -h, --help            show this help message and exit
  --build_errors_ds     Build error dataset instead of antennae dataset.
  --data_path DATA_PATH
                        Name of raw data folder. Defaults to 'Cropped-Anna Hadjitofi-2022-11-01'. For building errors dataset, this should be the path to the folder containing the experiments.
  --file_out FILE_OUT   Resulting name (and path) to save dataset(s), without extension. Defaults to current date.
  --nbins NBINS         Number of bins to use for binning angles to dancer. Defaults to 180.
  --antenna_len_str {default,mid_only,full_only}
                        Calculate antenna angle using mid length / bend or full length. The default uses base to tip if available and midpoint as fallback.
```

## Built With :hammer:

- [Matplotlib](https://matplotlib.org/stable/) - main plotting library
- [seaborn](https://seaborn.pydata.org/) - helper plotting library for some charts
- [NumPy](https://numpy.org/) - main numerical library for data vectorisation
- [Pandas](https://pandas.pydata.org/) - helper data manipulation library
- [OpenCV](https://pypi.org/project/opencv-python/) - video loading and extraction library
