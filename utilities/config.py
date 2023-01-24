import numpy as np

# TODO add yaml file reading, so that you could save conf and then ask to load them with CLI.
# TODO you can use ruamel.yaml https://yaml.readthedocs.io/en/latest/install.html

class Config:
    # TO SET.
    # TODO add wandb
    POLICY_DIR = "./weights/16rueoxi/raw/"
    POLICY_ITERATION = 8700
    INPUT_DIMS = 58
    OUTPUT_DIMS = 24
    LOGDIR = "/tmp/logs"  # where the data of the tests will be saved
    OBS_NORMALIZATION = True
    INI_JOINT_CONFIG = np.array([0.0, 0.5, -1.0]*4)
    LEG_JOINT_BOUND = [0.15, 0.4, 0.4]
    HIP_INDEX = range(0, 12, 3)
    THIGH_INDEX = range(1, 12, 3)
    CALF_INDEX = range(2, 12, 3)
    FILTER_WINDOW_LENGTH = 2
    # COMPUTED.
    WEIGHT_PATH = "./weights/16rueoxi/raw/"+"full_" + str(POLICY_ITERATION) + ".pt"
