from utilities.config import Config
import numpy as np
import os
import datetime


class Logger:

    def __init__(self, obs_ref=None, obsn_ref=None, action_policy_ref=None,
                 action_ref=None, policy_dt_ref=None,
                 last_action_time_ref=None, last_state_time_ref=None):
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
        if policy_dt_ref is not None:
            self.data_to_log["control_times"] = policy_dt_ref
        if last_state_time_ref is not None:
            self.data_to_log["last_state_times"] = last_state_time_ref
        if last_action_time_ref is not None:
            self.data_to_log["last_action_times"] = last_action_time_ref    
    def log(self):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder = Config.LOGDIR + "/" + date
        if not os.path.exists(folder):
            os.makedirs(folder)
        for data_name in self.data_to_log:
            np.savetxt(f"{folder}/{data_name}.csv", self.data_to_log[data_name], fmt="%1.5f")

