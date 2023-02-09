import numpy as np

# TODO add yaml file reading, so that you could save conf and then ask to load them with CLI.
# TODO you can use ruamel.yaml https://yaml.readthedocs.io/en/latest/install.html


class MotionImitationConfig:
    # TO SET.
    # TODO add wandb
    POLICY_DIR = None
    POLICY_ITERATION = None
    INPUT_DIMS = None
    OUTPUT_DIMS = None
    LOGDIR = "/tmp/logs"  # where the data of the tests will be saved
    OBS_NORMALIZATION = True
    INI_JOINT_CONFIG = np.array([0.0, 0.5, -1.0]*4)
    LEG_JOINT_BOUND = [0.15, 0.4, 0.4]
    HIP_INDEX = range(0, 12, 3)
    THIGH_INDEX = range(1, 12, 3)
    CALF_INDEX = range(2, 12, 3)
    FILTER_WINDOW_LENGTH = 3
    LEG_BOUNDS = {"hip": [-0.5, 0.5], "thigh": [-0.1, 1.5], "calf": [-2.1, -0.5]}
    LEG_BOUNDS_RESIDUAL_POLICY = {"hip": [-0.4, 0.4], "thigh": [-0.6, 0.6], "calf": [-0.6, 0.6]}
    OBS_WINDOW = [-1000,-500,-200,-20,20,200,500,1000]
    OBS_WINDOW_RESIDUAL_POLICY = [-1000,-500,-200,-20,0,20,200,500,1000]
