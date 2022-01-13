class Config:
    # TO SET.
    POLICY_DIR = "./weights/16rueoxi/raw/"
    POLICY_ITERATION = 8700
    INPUT_DIMS = 58
    OUTPUT_DIMS = 24
    LOGDIR = "/tmp/logs"
    OBS_NORMALIZATION = True
    INI_JOINT_CONFIG = [0.0, 0.5, -1.0]*4
    # COMPUTED.
    WEIGHT_PATH = "./weights/16rueoxi/raw/"+"full_" + str(POLICY_ITERATION) + ".pt"
