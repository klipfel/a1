# TODO write an exception in case the robot server crashes, i.e. save data
# saved as greeting-client.py
import Pyro5.api
import config
import numpy as np
import time
import argparse
import os
# Pybullet.
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
from utilities.control_util import HdwMotionImitationObservationParser, MotionImitationActionBridge,\
    ImitationPolicy, save_single_data_to_csv
from motion_clips.motionClip import MotionClipParser
import datetime
from utilities.motion_imitation_config import MotionImitationConfig
# TODO not complete, I actually don't need this code.


def create_tmp_data_folder():
    # Folder where the test data will be saved
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_data_folder = "data" + "/" + date
    if not os.path.exists(test_data_folder):
        os.makedirs(test_data_folder)
    return test_data_folder


class LaptopPolicy:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-u", "--uri", help="URI of the proxy of the Policy object", type=str, default=None)
        parser.add_argument('--motion_clip_folder', help='path of the motion clip folder', type=str, default='')
        parser.add_argument('--motion_clip_name', help='Name of the motion clip interpolation file.', type=str, default='')
        parser.add_argument("--wandb", help='If present as an arg the model will be downloaded from wandb directly.', action='store_true')
        parser.add_argument('--run_path', help='wandb run path entity/project/run_id.', type=str, default='')
        parser.add_argument('--update', help='update number of the model to test', type=int, default=None)
        parser.add_argument("-obsn", "--obs_normalization", help="Normalize or not observations based on the data accumulated in Raisim.", action='store_true')
        args = parser.parse_args()
        self.args = args
        self.robot = self.get_robot()  # HERE IT IS THE URI OF THE OBJECT SO REMOTE
        self.test_data_folder = create_tmp_data_folder()
        self.motion_clip_parser = None
        self.motion_clip_folder = args.motion_clip_folder
        self.motion_clip_name = args.motion_clip_name
        self.policy = None
        self.obs_parser = None
        self.motion_clip_parser = None
        self.action_bridge = None
        self.leg_bounds = MotionImitationConfig.LEG_BOUNDS
        self.uri = None
        self.data = {"obs": [],
                     "action_np": [],
                     "action_robot": [],
                     "control_times": []
                     }
        self.motion_clip_frame_rate = 0.001 # in seconds
        self.control_time = 0.02 # in seconds

    def get_robot(self):
        '''
        gets the remote robot object class.
        :return:
        '''
        # robot = Pyro5.api.Proxy(self.args.uri)     # get a Pyro proxy to the greeting object
        self.uri = input("ENTER THE ROBOT URI AND PRESS ENTER .... "
                         "for example PYRO:obj_fb3e9bf61d3248e0ae5f9c421ae7c5f2@127.0.0.1:2020.\n")
        robot = Pyro5.api.Proxy(self.uri)
        robot._pyroBind()
        robot._pyroSerializer = "marshal"  # faster communication.
        robot._pyroTimeout = 1.5    # 1.5 seconds
        return robot

    def get_motion_clip_parser(self):
        self.motion_clip_parser = MotionClipParser(data_folder=self.test_data_folder)
        # TODO implement a child class that calls the observation from the robot
        # self.obs_parser = MotionImitationObservationParser(self.robot, self.args, self.policy,
        #                                                    motion_clip_parser=self.motion_clip_parser,
        #                                                    data_folder=self.test_data_folder)
        # self.leg_bounds = {"hip": [-0.5, 0.5], "thigh": [-0.1, 1.5], "calf": [-2.1, -0.5]}
        # self.action_bridge = MotionImitationActionBridge(self.robot, leg_bounds=self.leg_bounds)
        # self.ini_conf = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][-12:]
        # self.ini_base_state = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][:7]
        # self.ini_com = self.ini_base_state[:3]
        # self.ini_orn = self.ini_base_state[3:]

    def get_policy(self):
        self.policy = ImitationPolicy(self.args, folder=self.test_data_folder)

    def get_obs_parser(self):
        self.obs_parser = HdwMotionImitationObservationParser(self.robot, self.args, self.policy,
                                                              motion_clip_parser=self.motion_clip_parser,
                                                              data_folder=self.test_data_folder)

    def get_action_bridge(self):
        self.action_bridge = MotionImitationActionBridge(self.robot, leg_bounds=self.leg_bounds)

    def test_inference_loop(self):
        self.get_motion_clip_parser()
        # TODO get the policy
        self.get_policy()
        self.get_obs_parser()
        self.get_action_bridge()
        while True:
            # TODO implement motion tracking, give the target frames
            # Inference loop.
            t0 = time.time()
            obs_np = self.obs_parser.observe()  # REMOTE
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            action_robot = self.action_bridge.adapt(action_np)
            self.robot.apply_action(action_robot.tolist())  # REMOTE
            delta = time.time() - t0
            print(f"Time of inference: {delta}")

    def save_data(self, obs, action_np, action_robot, control_time):
        self.data["obs"].append(list(obs.flatten()))
        self.data["action_np"].append(list(action_np.flatten()))
        self.data["action_robot"].append(list(action_robot.flatten()))
        self.data["control_times"].append(control_time)

    def write_data_to_csv(self):
        folder = f"{self.test_data_folder}/imitation_controller-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(folder)
        for data_name in self.data.keys():
            save_single_data_to_csv(np.array(self.data[data_name]), data_name, folder)

    def inference_loop(self):
        self.get_motion_clip_parser()
        # TODO get the policy
        self.get_policy()
        self.get_obs_parser()
        self.get_action_bridge()
        self.test_loop()
        self.motion_clip_tracking()
        self.write_data_to_csv()

    def test_loop(self):
        '''
        Test loop for the hardware where the robot is controlled to an initial joint configuration.
        Initial state matching is not possible on hdw.
        :return:
        '''
        input("PROCEED TO INITIAL TEST ON HDW?")
        for _ in range(100):
            # Inference loop.
            t0 = time.time()
            obs_np = self.obs_parser.observe(target_frame=0)  # REMOTE
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            action_robot = self.action_bridge.adapt(action_np)
            self.robot.get_action(action_robot.tolist())  # REMOTE
            delta = time.time() - t0
            print(f"Time of inference: {delta}")
            self.save_data(obs=obs_np,
                           action_np=action_np,
                           action_robot=action_robot,
                           control_time=delta)

    def motion_clip_tracking(self):
        '''
        Function that implements the motion clip tracking.
        :return:
        '''
        input("PROCEED TO MOTION CLIP TRACKING ON HDW?")
        frame = 0
        while frame < self.motion_clip_parser.motion_clip_sim_frames:
            # Inference loop.
            t0 = time.time()
            obs_np = self.obs_parser.observe(target_frame=frame)  # REMOTE
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            action_robot = self.action_bridge.adapt(action_np)
            self.robot.get_action(action_robot.tolist())  # REMOTE
            delta = time.time() - t0
            print(f"Time of inference: {delta}")
            # Frame update
            if delta < self.control_time:
                # in this case you can make the robot sleep a bit
                time.sleep(self.control_time-delta)
            dframe = int(delta/self.motion_clip_frame_rate) + 1
            frame += dframe
            # save data
            self.save_data(obs=obs_np,
                           action_np=action_np,
                           action_robot=action_robot,
                           control_time=delta)


# Client loop.
if __name__ == "__main__":
    input("PRESS ENTER IF YOU WANT TO START THE LAPTOP SERVER ....")
    policy = LaptopPolicy()
    while True:
        policy.inference_loop()
