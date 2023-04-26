# TODO write an exception in case the robot server crashes, i.e. save data
# saved as greeting-client.py
import copy

import Pyro5.api
import config
import numpy as np
import time
import argparse
import os, sys
import pandas as pd

import self as self
import tqdm
# Pybullet.
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
from utilities.control_util import HdwMotionImitationObservationParser, MotionImitationActionBridge,\
    ImitationPolicy, save_single_data_to_csv, MotionImitationResidualPolicyActionBridge
from motion_clips.motionClip import MotionClipParser
import datetime
from utilities.motion_imitation_config import MotionImitationConfig
# TODO not complete, I actually don't need this code.

LINE = "-"*100
CONTROL_SIM_RATE = 0.001 # 0.0001 # sim 0.001
REF_FRAME_RATE = 0.001


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
        parser.add_argument("-mu", "--mocap_uri", help="URI of the proxy of the mocap system object", type=str, default=None)
        parser.add_argument("-a", "--activation_fn", help="Activation function for the hidden layers [Tanh,ReLU, LeakyReLU, ELU].", type=str, default="LeakyReLU")
        parser.add_argument('--motion_clip_folder', help='path of the motion clip folder', type=str, default='')
        parser.add_argument('--motion_clip_name', help='Name of the motion clip interpolation file.', type=str, default='')
        parser.add_argument("--wandb", help='If present as an arg the model will be downloaded from wandb directly.', action='store_true')
        parser.add_argument("--obs_filtering", help='If present as an arg observations will be filtered.', action='store_true')
        parser.add_argument("--leg_control", help='Control mode to control only one leg.', action='store_true')
        parser.add_argument('--run_path', help='wandb run path entity/project/run_id.', type=str, default='')
        parser.add_argument('--update', help='update number of the model to test', type=int, default=None)
        parser.add_argument("-obsn", "--obs_normalization", help="Normalize or not observations based on the data accumulated in Raisim.", action='store_true')
        parser.add_argument("--policy_type", help="Either residual or non-residual policy.", type=str, default='')
        parser.add_argument("--ob_dim", help="Input dimension of the policy.", type=int, default='')
        parser.add_argument("--rel_info", help="Adds relInfo to the start of observations.", action='store_true')
        parser.add_argument("--use_mocap", help="Flag to use the mocap lab in the policy observations.", action='store_true')
        parser.add_argument("--base_state_matching", help="Matches initial base state with the reference, only in sim.",
                            action='store_true')
        parser.add_argument("--obs_action_hist", help="Add actions history in the observations.", type=int, default=0)
        parser.add_argument("--hdw_com_issue", help="Adds a fixed Com observation [0.012731, 0.002186, 1.000515]"
                                                    "like what is cimputed on the hdw to simulate ir.", action='store_true')
        parser.add_argument("--remove_robot_com_in_obs", help="Flag to remove robot CoM in observations",
                            action='store_true')
        parser.add_argument('--filter_window', help='Window of action filter.', type=int, default=3)
        args = parser.parse_args()
        self.args = args
        self.robot = self.get_robot()  # HERE IT IS THE URI OF THE OBJECT SO REMOTE
        if args.use_mocap:
            self.mocap_system = self.get_mocap_system()
        self.test_data_folder = create_tmp_data_folder()
        self.motion_clip_parser = None
        self.motion_clip_folder = args.motion_clip_folder
        self.motion_clip_name = args.motion_clip_name
        self.policy = None
        self.obs_parser = None
        self.motion_clip_parser = None
        self.action_bridge = None
        self.leg_bounds = None
        self.uri = None
        self.mocap_system_uri = None
        self.data = {"obs": [],
                     "action_np": [],
                     "action_robot": [],
                     "control_times": [],
                     "mocap_sys_data": []
                     }
        self.motion_clip_frame_rate = CONTROL_SIM_RATE # in seconds
        self.control_time = 0.02 # in seconds
        self.ini_conf = None
        self.most_recent_robot_sensor_data = None
        # Filter window setting
        MotionImitationConfig.FILTER_WINDOW_LENGTH = args.filter_window

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


    def get_mocap_system(self):
        '''
        gets the remote robot object class.
        :return:
        '''
        # robot = Pyro5.api.Proxy(self.args.uri)     # get a Pyro proxy to the greeting object
        self.mocap_system_uri = input("ENTER THE MOCAP SYSTEM URI AND PRESS ENTER .... "
                         "for example PYRO:obj_fb3e9bf61d3248e0ae5f9c421ae7c5f2@127.0.0.1:2020.\n")
        mocap_system = Pyro5.api.Proxy(self.mocap_system_uri)
        mocap_system._pyroBind()
        mocap_system._pyroSerializer = "marshal"  # faster communication.
        mocap_system._pyroTimeout = 1.5    # 1.5 seconds
        return mocap_system

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
        ob_dim = self.args.ob_dim             
        self.policy = ImitationPolicy(self.args, folder=self.test_data_folder, ob_dim=ob_dim,
                                      activation_fn_name=self.args.activation_fn)

    def get_obs_parser(self):
        self.obs_parser = HdwMotionImitationObservationParser(self.robot, self.args, self.policy,
                                                              motion_clip_parser=self.motion_clip_parser,
                                                              data_folder=self.test_data_folder)
        # REl info in obs
        if self.args.rel_info:
            self.obs_parser.set_rel_info_flag(True)
        else:
            self.obs_parser.set_rel_info_flag(False)
        # Filter window length
        if self.args.policy_type == "no-res":
            self.obs_parser.set_obs_window(MotionImitationConfig.OBS_WINDOW)
        elif self.args.policy_type == "res":
            self.obs_parser.set_obs_window(MotionImitationConfig.OBS_WINDOW_RESIDUAL_POLICY)
        else:
            print("Unknow policy type. Please choose between res or no-res.")
            sys.exit(1)

    def get_action_bridge(self):
        if self.args.policy_type == "no-res":
            self.leg_bounds = MotionImitationConfig.LEG_BOUNDS
            self.action_bridge = MotionImitationActionBridge(self.robot, leg_bounds=self.leg_bounds)
        elif self.args.policy_type == "res":
            self.leg_bounds = MotionImitationConfig.LEG_BOUNDS_RESIDUAL_POLICY
            self.action_bridge = MotionImitationResidualPolicyActionBridge(self.robot, leg_bounds=self.leg_bounds)
        else:
            print("Unknow policy type. Please choose between res or no-res.")
            sys.exit(1)

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
            self.action_bridge.set_mean(new_mean=list(self.obs_parser.motion_clip["Interp_Motion_Data"][0][-12:]))
            action_robot = self.action_bridge.adapt(action_np)
            self.robot.apply_action(action_robot.tolist())  # REMOTE
            delta = time.time() - t0
            print(f"Time of inference: {delta}")

    def save_data(self, obs, action_np, action_robot, control_time, mocap_sys_data):
        self.data["obs"].append(list(obs.flatten()))
        self.data["action_np"].append(list(action_np.flatten()))
        self.data["action_robot"].append(list(action_robot.flatten()))
        self.data["control_times"].append(control_time)
        self.data["mocap_sys_data"].append(list(mocap_sys_data.flatten()))

    def write_data_to_csv(self):
        # TODO add the non normalized obs to the data buffer and save as csv obs_buffer
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
        self.initial_joint_matching()
        if self.args.base_state_matching:
            self.base_state_matching()
        # self.keep_initial_frame()
        # self.motion_clip_tracking()
        self.smooth_motion_clip_tracking()

    def base_state_matching(self):
        print("/!\ WARNING: base state matching activated, only possible in simulation.")
        ini_position = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][:3]
        ini_orn = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][3:3+4]
        self.robot.get_and_set_initial_reference_base_state(ini_position=ini_position.tolist(),
                                                    ini_orn=ini_orn.tolist())


    def get_sensor_data(self):
        self.most_recent_robot_sensor_data = self.robot.get_sensor_data()
        print(f"Sensor data/COM: {self.most_recent_robot_sensor_data[:3]}")
        print(f"Sensor data/Rotation matrix: {self.most_recent_robot_sensor_data[3+12:3+12+9]}")

    def get_robot_joint_postions(self):
        '''
        :return: list of joint positions
        '''
        return self.most_recent_robot_sensor_data[3:3+12]

    def send_action_to_robot(self, action):
        self.robot.get_action(action.tolist())

    def go_to_fixed_configuration(self, target_jp, alpha=0.8, nsteps=2000, dt=0.005):
        """
        Sets the robot in an initial configuration. Preferably close to the ones the robot was trained on at the start
        of the training episodes. Prepares the robot for policy.
        :param alpha: during 0.8*steps the robot will gradually be guided to the desired_motor_angle, and during 0.2*n_steps
        it will be asked to go there directly. First step: transition and then once the joint configuration is not too
        far the robot is controlled to it.
        """
        print(LINE)
        print("WARNING, THE ROBOT IS GOING TO BE CONTROLLED TO A FIXED CONFIGURATION...")
        print(f"Setting joint positions to: {target_jp}")
        self.get_sensor_data()
        current_motor_angle = np.array(self.get_robot_joint_postions())
        print("Current joint positions:", current_motor_angle)
        for t in tqdm.tqdm(range(nsteps)):
            blend_ratio = np.minimum(t / (nsteps*alpha), 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * target_jp
            self.send_action_to_robot(action)
            self.get_sensor_data()
            time.sleep(dt)  # the example used 0.005.
        print(LINE)

    def smooth_control(self, starting_motor_angles, target_jp, nsteps=2000, dt=0.005):
        """
        Sets the robot in an initial configuration. Preferably close to the ones the robot was trained on at the start
        of the training episodes. Prepares the robot for policy.
        :param alpha: during 0.8*steps the robot will gradually be guided to the desired_motor_angle, and during 0.2*n_steps
        it will be asked to go there directly. First step: transition and then once the joint configuration is not too
        far the robot is controlled to it.
        """
        for t in range(nsteps):
            blend_ratio = np.minimum(t / (nsteps), 1)
            action = (1 - blend_ratio) * starting_motor_angles + blend_ratio * target_jp
            if self.args.leg_control:
                action[3:] = 0
            self.send_action_to_robot(action)
            time.sleep(dt)  # the example used 0.005.

    def initial_joint_matching(self):
        self.ini_joint_positions = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][-12:]
        # self.ini_joint_positions = np.array([-0.5, 0.9, -1.8,
        #                                       0.5, 0.9, -1.8,
        #                                       0, 0.9, -1.8,
        #                                       0, 0.9, -1.8])
        input(f"PRESS ENTER TO START THE INITIAL JOINT MATCHING......\n")
        self.go_to_fixed_configuration(self.ini_joint_positions,
                                       alpha=0.8,
                                       nsteps=4000,
                                       dt=0.001)
        print(LINE)

    def keep_initial_frame(self):
        '''
        Test loop for the hardware where the robot is controlled to an initial joint configuration.
        Initial state matching is not possible on hdw.
        :return:
        '''
        n_control_step = 2000
        #input(f"PROCEED TO INITIAL TEST ON HDW? THE POLICY IS GOING TO BE ASKED TO KEEP THE INITIAL STATE"
        #      f" FOR {n_control_step} control steps.")
        for _ in range(n_control_step):
            # Inference loop.
            t0 = time.time()
            obs_np = self.obs_parser.observe(target_frame=0)  # REMOTE
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            self.action_bridge.set_mean(new_mean=list(self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][-12:]))
            action_robot = self.action_bridge.adapt(action_np)
            self.send_action_to_robot(action_robot)  # REMOTE
            delta = time.time() - t0
            print(f"Time of inference: {delta}")
            # Frame update
            # TODO maybe dilute the control a bit, best would to smaple from policy every 0.02 sec amnd in the
            # TODO meantime, on the robot, you can just set some intermediary goals
            if delta < self.control_time:
                # in this case you can make the robot sleep a bit
                time.sleep(self.control_time-delta)
            # TODO add a time.sleep in any case
            self.save_data(obs=obs_np,
                           action_np=action_np,
                           action_robot=action_robot,
                           control_time=delta)

    def motion_clip_tracking(self):
        '''
        Function that implements the motion clip tracking.
        :return:
        '''
        #input("PROCEED TO MOTION CLIP TRACKING ON HDW?")
        frame = 0
        while frame < self.motion_clip_parser.motion_clip_sim_frames:
            # Inference loop.
            t0 = time.time()
            # Action history computed from the last actions
            action_hist = self.compute_action_history()
            obs_np = self.obs_parser.observe(target_frame=frame,
                                             action_history=action_hist) 
            # REMOTE
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            self.action_bridge.set_mean(new_mean=list(self.motion_clip_parser.motion_clip["Interp_Motion_Data"][frame][-12:]))
            action_robot = self.action_bridge.adapt(action_np)
            self.send_action_to_robot(action_robot)  # REMOTE
            delta = time.time() - t0
            print(f"Time of inference: {delta}")
            # Frame update
            if delta < self.control_time:
                # in this case you can make the robot sleep a bit
                time.sleep(self.control_time-delta)
            dframe = int(delta/REF_FRAME_RATE) + 1
            frame += dframe
            # save data
            self.save_data(obs=obs_np,
                           action_np=action_np,
                           action_robot=action_robot,
                           control_time=delta)

    def smooth_motion_clip_tracking(self):
        '''
        Function that implements the motion clip tracking.
        :return:
        '''
        #input("PROCEED TO MOTION CLIP TRACKING ON HDW?")
        frame = 0
        #for _ in range(100):
        while frame < self.motion_clip_parser.motion_clip_sim_frames:
            # Inference loop.
            t0 = time.time()
            # Action history computed from the last actions
            action_hist = self.compute_action_history()
            obs_np = self.obs_parser.observe(target_frame=frame,
                                             action_history=action_hist) 
            # REMOTE
            # Mocap lab
            if self.args.use_mocap:
                # Gets the data from the mocap system
                mocap_sys_data = self.mocap_system.get_data()
                mocap_sys_data = np.array(mocap_sys_data[0]).reshape((1,-1))# removes first dimension
                print(f"Mocap system data: {mocap_sys_data}")
                obs_np = self.update_obs_with_mocap_data(obs_np, mocap_sys_data)
            else:
                mocap_sys_data = np.array([0])
            # filtering observation
            if self.args.obs_filtering:
                obs_np = self.filter_obs(obs_np)
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            self.action_bridge.set_mean(new_mean=list(self.motion_clip_parser.motion_clip["Interp_Motion_Data"][frame][-12:]))
            action_robot = self.action_bridge.adapt(action_np)
            # TODO do the control dilution on the robotServer to save communication times
            self.smooth_control(starting_motor_angles=self.obs_parser.robot_data[3:3+12],
                                target_jp=action_robot,
                                nsteps=20,
                                dt=0.003)
            delta = time.time() - t0
            print(f"Control time: {delta}")
            # dframe = int(delta/REF_FRAME_RATE) + 1
            dframe = 20
            # dframe = 10 # try to make it like in sim
            frame += dframe
            # save data
            self.save_data(obs=obs_np,
                           action_np=action_np,
                           action_robot=action_robot,
                           control_time=delta,
                           mocap_sys_data=mocap_sys_data)

    def update_obs_with_mocap_data(self, obs_np, mocap_sys_data):
        # Adds the mocap information to the policy observations
        obs = copy.deepcopy(obs_np)
        # CoM
        obs[:, :3] = mocap_sys_data[:, :3]
        # Rotation Matrix of the robot, 9 coefficients afyer CoM, and jp.
        obs[:, 3+12:3+12+9] = mocap_sys_data[:, 3:3+9]
        # TODO I am not using the occlusion flag at the end of the mocap stream for now
        return obs

    def filter_obs(self, obs):
        filter_obs = copy.deepcopy(obs)
        if len(self.data["obs"]) > 0:
            filter_obs = (obs + np.array(self.data["obs"][-1]))/2.0
        return filter_obs
    
    def compute_action_history(self):
        """
         Adds the action history to the action_robot
        """
        action_hist = None
        if self.args.obs_action_hist > 0: # if there is action history
            action_hist_size = 12*self.args.obs_action_hist
            n =12
            action_hist = np.zeros((1, action_hist_size))
            k = -1
            n_action = len(self.data["action_robot"])
            while n_action + k >= 0 and self.args.obs_action_hist + k >= 0:
                # Better to use positive indexes when doing slicing
                end = action_hist_size+(k+1)*n-1
                start = end - (n-1)
                # you cannot access the last element of the array with -1 when doing slicing as the last is excluded
                action_hist[:, start:end+1] = np.array(self.data["action_robot"][n_action+k])
                k -= 1
        return action_hist


# Client loop.
if __name__ == "__main__":
    input("PRESS ENTER IF YOU WANT TO START THE LAPTOP SERVER ....")
    policy = LaptopPolicy()
    try:
        policy.inference_loop()
    except Exception as e:
        print(f"Exception : {e}")
        policy.write_data_to_csv()
