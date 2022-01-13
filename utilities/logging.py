from utilities.config import Config
import numpy as np
import os


class Logger:

    def __init__(self, obs_ref=None, obsn_ref=None, action_policy_ref=None,
                 action_ref=None):
        """
        :param obs_ref: ref to the obs buffer.
        Any mutable object (list, dict, arrays) that contains data you want to store.
        """
        self.data_to_log = {}
        if obs_ref is not None:
            self.data_to_log["observations"] = obs_ref
        if obsn_ref is not None:
            self.data_to_log["normalized_observations"] = obsn_ref
        if action_policy_ref is not None:
            self.data_to_log["policy_action"] = action_policy_ref
        if action_ref is not None:
            self.data_to_log["robot_action"] = action_ref

    def log(self):
        if not os.path.exists(Config.LOGDIR):
            os.makedirs(Config.LOGDIR)
        for data_name in self.data_to_log:
            np.savetxt(f"{Config.LOGDIR}/{data_name}.csv", self.data_to_log[data_name], fmt="%1.5f")

