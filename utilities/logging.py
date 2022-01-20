from utilities.config import Config
import numpy as np
import os
import datetime
# import pandas as pd


class Logger:

    def __init__(self, args=None, obs_ref=None, obsn_ref=None, action_policy_ref=None,
                 action_ref=None, policy_dt_ref=None,
                 last_action_time_ref=None, last_state_time_ref=None):
        """
        :param obs_ref: ref to the obs buffer.
        Any mutable object (list, dict, arrays) that contains data you want to store.
        """
        self.data_to_log = {}
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder = Config.LOGDIR + "/" + date
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.args = args
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
        for data_name in self.data_to_log:
            # TODO add labels to csv file, you can use pandas to do it
            # TODO or just use the header arg of savetxt as in https://stackoverflow.com/questions/36210977/python-numpy-savetxt-header-has-extra-character
            # Check if the first list of the buffer contains labels for the columns, if so removes them and use
            # it after for labelling.
            np.savetxt(f"{self.folder}/{data_name}.csv", self.data_to_log[data_name], fmt="%1.5f")
            # df = pd.read_csv(f"{folder}/{data_name}.csv", header=None)
            # df.to_csv(f"{folder}/{data_name}_labels.csv", header=["Letter", "Number", "Symbol"], index=False)

    def log_now(self, data_name, data, fmt="%1.5f", extension=".csv"):
        np.savetxt(f"{self.folder}/{data_name}.{extension}", data, fmt=fmt)

    def save_args(self):
        if self.args is not None and self.folder is not None:
            arg_file = open(f"{self.folder}/args.txt", "w")
            for argname in self.args.__dict__.keys():
                arg_file.write(f"{argname}: {self.args.__dict__[argname]}\n")
            arg_file.close()
