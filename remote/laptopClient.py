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
# TODO quantify the mocap marker offset.
MOCAP_MARKER_Z_OFFSET_POSITIVE = 0.419668594644447 - 0.346179394121176 # Marker is on top of the robot, and not exactly at the CoM.


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
        parser.add_argument("-aop", "--actions_open_loop_file", help="Actions open loop to read and apply to the DC motors. Csv file.", type=str, default=None)
        parser.add_argument("-ml", "--mocap_length", help="Percentage of mocap to play.", type=float, default=1.0)
        parser.add_argument("--sleep_dt", help="Sleep between control sent to actuation.", type=float, default=0.005)
        parser.add_argument("-ni", "--n_interpolation", help="Number of interpolated actions.", type=int, default=20)
        parser.add_argument("-nra", "--n_repeat_action", help="Number of interpolated actions.", type=int, default=4)
        parser.add_argument("--start_frame", help="Starting mocap frame.", type=int, default=0)
        parser.add_argument("-mu", "--mocap_uri", help="URI of the proxy of the mocap system object", type=str, default=None)
        parser.add_argument("-a", "--activation_fn", help="Activation function for the hidden layers [Tanh,ReLU, LeakyReLU, ELU].", type=str, default="LeakyReLU")
        parser.add_argument('--motion_clip_folder', help='path of the motion clip folder', type=str, default='')
        parser.add_argument('--motion_clip_name', help='Name of the motion clip interpolation file.', type=str, default='')
        parser.add_argument("--wandb", help='If present as an arg the model will be downloaded from wandb directly.', action='store_true')
        parser.add_argument("--obs_filtering", help='If present as an arg observations will be filtered.', action='store_true')
        parser.add_argument("--leg_control", help='Control mode to control only one leg.', action='store_true')
        parser.add_argument("--slower", help='Control mode to slow mocap 2x.', action='store_true')
        parser.add_argument('--run_path', help='wandb run path entity/project/run_id.', type=str, default='')
        parser.add_argument('--update', help='update number of the model to test', type=int, default=None)
        parser.add_argument("-obsn", "--obs_normalization", help="Normalize or not observations based on the data accumulated in Raisim.", action='store_true')
        parser.add_argument("--policy_type", help="Either residual or non-residual policy.", type=str, default='')
        parser.add_argument("--ob_dim", help="Input dimension of the policy.", type=int, default='')
        parser.add_argument("--rel_info", help="Adds relInfo to the start of observations.", action='store_true')
        parser.add_argument("--use_mocap", help="Flag to use the mocap lab in the policy observations.", action='store_true')
        parser.add_argument("--base_state_matching", help="Matches initial base state with the reference, only in sim.",
                            action='store_true')
        parser.add_argument("--base_position", help="To set the initial base position only in sim.",
                            type=float, nargs='+', default=None)
        parser.add_argument("--base_orn", help="To set the initial base orientation only in sim.",
                            type=float, nargs='+', default=None)
        parser.add_argument("--policy_nn", help="Architecture of the policy.",
                            type=int, nargs='+', default=[256, 256])
        parser.add_argument("--obs_action_hist", help="Add actions history in the observations.", type=int, default=0)
        parser.add_argument("--hdw_com_issue", help="Adds a fixed Com observation [0.012731, 0.002186, 1.000515]"
                                                    "like what is cimputed on the hdw to simulate ir.", action='store_true')
        parser.add_argument("--remove_robot_com_in_obs", help="Flag to remove robot CoM in observations",
                            action='store_true')
        parser.add_argument('--filter_window', help='Window of action filter.', type=int, default=3)
        parser.add_argument("--repeat", help='Repeats actions instead of interpolation.', action='store_true')
        parser.add_argument("--add_mocap_marker_offset", help='Adds the mocap marker offset to the CoM observations.', action='store_true')
        parser.add_argument("--dframe", help="Number of frames in a control step.", type=int, default=20)
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
        self.mocap_local = MocapSystem()
        # TODO add filtered observations
        self.data = {"obs": [],
                     "obs_not_normalized": [],
                     "action_np": [],
                     "action_robot": [],
                     "control_times": [],
                     "mocap_sys_data": []
                     }
        self.motion_clip_frame_rate = CONTROL_SIM_RATE # in seconds
        self.control_time = 0.02 # in seconds
        self.ini_conf = None
        self.most_recent_robot_sensor_data = None
        self.static_mocap_readings = None
        # Filter window setting
        MotionImitationConfig.FILTER_WINDOW_LENGTH = args.filter_window
        # Getting open loop control actions from a csv file using pandas.
        self.actions_open_loop = None
        if args.actions_open_loop_file is not None:
            print(f"Getting open loop actions from csv file: {args.actions_open_loop_file}")
            self.actions_open_loop = pd.read_csv(args.actions_open_loop_file, header=None, lineterminator=';').to_numpy()
            print(f"!!!!!!!!!!! Open loop actioons provided with shape {self.actions_open_loop.shape}.")


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
                                      activation_fn_name=self.args.activation_fn,
                                      architecture=self.args.policy_nn)

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
            obs = self.obs_parser.do_normalize()
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
        # Save args to a txt file
        with open(f"{folder}/args.txt", "w") as f:
            f.write(str(self.args))
            # Save the data
        f.close()
        # adds other data in the buffer
        self.data["obs_not_normalized"] = self.obs_parser.obs_buffer # non normalized obs
        # saves the data generated during the inference loop
        for data_name in self.data.keys():
            save_single_data_to_csv(np.array(self.data[data_name]), data_name, folder)

    def inference_loop(self):
        self.get_motion_clip_parser()
        # TODO get the policy
        self.get_policy()
        self.get_obs_parser()
        self.get_action_bridge()
        self.initial_joint_matching()
        if self.args.use_mocap:
            self.mocap_local.calibrate()
        if self.args.base_state_matching:
            self.base_state_matching()
        self.set_base_state()
        # self.keep_initial_frame()
        # self.motion_clip_tracking()
        if self.actions_open_loop is not None:
            self.open_loop_control()
        else:
            if self.args.slower:
                self.smooth_motion_clip_tracking_slow_down()
            else:
                self.smooth_motion_clip_tracking()

    def set_base_state(self):
        if self.args.base_position is not None and self.args.base_orn is not None:
            ini_position = self.args.base_position
            ini_orn = pybullet.getQuaternionFromEuler(self.args.base_orn)
            self.robot.get_and_set_initial_reference_base_state(ini_position=ini_position,
                                                                ini_orn=ini_orn)

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
            self.get_sensor_data() #  TODO why did I do this?
            time.sleep(dt)  # the example used 0.005.
            # Mocap system calibration. gathers data to determine initial position and then remove it in the data.
            # TODO is that the initial frame in which the robot data is computed? What is the robot data frame?
            if self.args.use_mocap:
                mocap_sys_data = self.mocap_system.get_data()
                mocap_sys_data = np.array(mocap_sys_data).reshape((1,-1))
                self.mocap_local.store(mocap_sys_data)
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
            self.get_sensor_data()
            blend_ratio = np.minimum(t / (nsteps), 1)
            action = (1 - blend_ratio) * starting_motor_angles + blend_ratio * target_jp
            #if self.args.leg_control:
            #    action[3:] = 0
            self.send_action_to_robot(action)
            time.sleep(dt)  # the example used 0.005.

    def initial_joint_matching(self):
        self.ini_joint_positions = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][-12:]
        # In case the user wants to open loop control the robot to a specific joint configuration.
        if self.actions_open_loop is not None:
            self.ini_joint_positions = self.actions_open_loop[0][1:]
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
            obs = self.obs_parser.do_normalize()
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
            obs = self.obs_parser.do_normalize()
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
            
    def open_loop_control(self):
        """
        open loop control of the robot. Read stored robot action as csv and send them to the robot.
        The control loop is the same as the closed-loop to take into accoutn sensor delay and actuation delay
        """
        #input("PROCEED TO MOTION CLIP TRACKING ON HDW?")
        for frame in range(1,self.actions_open_loop.shape[0]):
            # Inference loop.
            t0 = time.time()
            # Action history computed from the last actions
            action_hist = self.compute_action_history()
            obs = self.obs_parser.observe(target_frame=frame,
                                             action_history=action_hist) 
            # REMOTE
            # Mocap lab
            if self.args.use_mocap:
                # Gets the data from the mocap system
                mocap_sys_data = self.mocap_system.get_data()
                mocap_sys_data = np.array(mocap_sys_data).reshape((1,-1))# removes first dimension
                print(f"Received mocap system data: {mocap_sys_data}")
                obs = self.update_obs_with_mocap_data(obs, mocap_sys_data)
            else:
                mocap_sys_data = np.array([0])
            # filtering observation
            if self.args.obs_filtering:
                obs = self.filter_obs(obs)
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            # Normalizes observation
            obs_np = self.obs_parser.do_normalize(obs)
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            self.action_bridge.set_mean(new_mean=list(self.motion_clip_parser.motion_clip["Interp_Motion_Data"][frame][-12:]))
            action_robot = self.action_bridge.adapt(action_np)
            # Open loop control
            action_robot_t_1 = self.actions_open_loop[frame-1][1:]
            action_robot_t = self.actions_open_loop[frame][1:] # removes the first column which is the time
            print(f"Sending action to robot: {action_robot}")
            # TODO do the control dilution on the robotServer to save communication times
            self.smooth_control(starting_motor_angles=action_robot_t_1,
                                target_jp=action_robot_t,
                                nsteps=self.args.n_interpolation,
                                dt=self.args.sleep_dt)
            #self.repeat_action(action=action_robot,
            #                   n=self.args.n_repeat_action,
            #                   dt=self.args.sleep_dt)
            delta = time.time() - t0
            print(f"Control time: {delta}")
            # save data
            self.save_data(obs=obs_np,
                           action_np=action_np,
                           action_robot=action_robot,
                           control_time=delta,
                           mocap_sys_data=mocap_sys_data)
            
    def repeat_action(self, action, nsteps, dt):
        print("repeat actions!!")
        for t in range(nsteps):
            self.get_sensor_data() # put it in the interpolation loop
            self.send_action_to_robot(action)
            time.sleep(dt)  # the example used 0.005.
        
    def smooth_motion_clip_tracking(self):
        '''
        Function that implements the motion clip tracking.
        :return:
        '''
        #input("PROCEED TO MOTION CLIP TRACKING ON HDW?")
        frame = 0
        action_robot_t_1 = self.ini_joint_positions  # initial joint positions
        while frame < self.args.mocap_length*self.motion_clip_parser.motion_clip_sim_frames:
            # Inference loop.
            t0 = time.time()
            ####### OBSERVATIONS ################
            # Action history computed from the last actions
            action_hist = self.compute_action_history()
            obs = self.obs_parser.observe(target_frame=frame,
                                          action_history=action_hist) 
            # REMOTE
            # Mocap lab
            if self.args.use_mocap:
                # Gets the data from the mocap system
                mocap_sys_data = self.mocap_system.get_data()
                mocap_sys_data = np.array(mocap_sys_data).reshape((1,-1))# removes first dimension
                print(f"Received mocap system data: {mocap_sys_data}")
                obs = self.update_obs_with_mocap_data(obs, mocap_sys_data)
            else:
                mocap_sys_data = np.array([0])
            # filtering observation
            if self.args.obs_filtering:
                obs = self.filter_obs(obs)
            # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
            # Normalizes observation
            obs_np = self.obs_parser.do_normalize(obs)
            # Disturbs the observation with the mocap system data
            #com = obs[:,:3]
            #obs_np[:,:3] = com
            ########## INFERENCE ####################
            action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
            self.action_bridge.set_mean(new_mean=list(self.motion_clip_parser.motion_clip["Interp_Motion_Data"][frame][-12:]))
            action_robot = self.action_bridge.adapt(action_np)
            # TODO do the control dilution on the robotServer to save communication times
            #self.smooth_control(starting_motor_angles=self.obs_parser.robot_data[3:3+12],
            #                    target_jp=action_robot,
            #                    nsteps=self.args.n_interpolation,
            #                    dt=self.args.sleep_dt)
            # TODO you can give the previous readings as the previous action but make them of the same reference
            ############## ACRUATOR COMMAND ##################
            if self.args.repeat:
                self.repeat_action(action=action_robot,
                                   nsteps=self.args.n_repeat_action,
                                   dt=self.args.sleep_dt)
            else:
                self.smooth_control(starting_motor_angles=action_robot_t_1,
                                    target_jp=action_robot,
                                    nsteps=self.args.n_interpolation,
                                    dt=self.args.sleep_dt)
            delta = time.time() - t0
            print(f"Control time: {delta}")
            # dframe = int(delta/REF_FRAME_RATE) + 1
            dframe = self.args.dframe
            # dframe = 10 # try to make it like in sim
            frame += dframe
            action_robot_t_1 = action_robot
            # save data
            self.save_data(obs=obs_np,
                           action_np=action_np,
                           action_robot=action_robot,
                           control_time=delta,
                           mocap_sys_data=mocap_sys_data)
    
    def smooth_motion_clip_tracking_slow_down(self):
        '''
        Function that implements the motion clip tracking.
        :return:
        '''
        #input("PROCEED TO MOTION CLIP TRACKING ON HDW?")
        frame = self.args.start_frame
        action_robot_t_1 = self.ini_joint_positions  # initial joint positions
        while frame < self.args.mocap_length*self.motion_clip_parser.motion_clip_sim_frames:
            for k in range(2):
                # Inference loop.
                t0 = time.time()
                ####### OBSERVATIONS ################
                # Action history computed from the last actions
                action_hist = self.compute_action_history()
                obs = self.obs_parser.observe(target_frame=frame,
                                              action_history=action_hist)
                # REMOTE
                # Mocap lab
                if self.args.use_mocap:
                    # Gets the data from the mocap system
                    mocap_sys_data = self.mocap_system.get_data()
                    mocap_sys_data = np.array(mocap_sys_data).reshape((1,-1))# removes first dimension
                    print(f"Received mocap system data: {mocap_sys_data}")
                    obs = self.update_obs_with_mocap_data(obs, mocap_sys_data)
                else:
                    mocap_sys_data = np.array([0])
                # filtering observation
                if self.args.obs_filtering:
                    obs = self.filter_obs(obs)
                # print(f"SENSOR DATA at time = {t0}:{self.obs_parser.robot_data}")
                # Normalizes observation
                obs_np = self.obs_parser.do_normalize(obs)
                # Disturbs the observation with the mocap system data
                #com = obs[:,:3]
                #obs_np[:,:3] = com
                ########## INFERENCE ####################
                action_np = self.policy.inference(obs_np,  std=[0.1,0.3,0.3]*4)
                self.action_bridge.set_mean(new_mean=list(self.motion_clip_parser.motion_clip["Interp_Motion_Data"][frame][-12:]))
                action_robot = self.action_bridge.adapt(action_np)
                # TODO do the control dilution on the robotServer to save communication times
                #self.smooth_control(starting_motor_angles=self.obs_parser.robot_data[3:3+12],
                #                    target_jp=action_robot,
                #                    nsteps=self.args.n_interpolation,
                #                    dt=self.args.sleep_dt)
                # TODO you can give the previous readings as the previous action but make them of the same reference
                ############## ACRUATOR COMMAND ##################
                if self.args.repeat:
                    self.repeat_action(action=action_robot,
                                       nsteps=self.args.n_repeat_action,
                                       dt=self.args.sleep_dt)
                else:
                    self.smooth_control(starting_motor_angles=action_robot_t_1,
                                        target_jp=action_robot,
                                        nsteps=self.args.n_interpolation,
                                        dt=self.args.sleep_dt)
                delta = time.time() - t0
                print(f"Control time: {delta}")
                action_robot_t_1 = action_robot
                # save data
                self.save_data(obs=obs_np,
                               action_np=action_np,
                               action_robot=action_robot,
                               control_time=delta,
                               mocap_sys_data=mocap_sys_data)
            # New frame.
            # dframe = int(delta/REF_FRAME_RATE) + 1
            dframe = self.args.dframe
            # dframe = 10 # try to make it like in sim
            frame += dframe


    def update_obs_with_mocap_data(self, obs_np, real_mocap_sys_data):
        # Adds the mocap information to the policy observations
        obs = copy.deepcopy(obs_np)
        mocap_sys_data = copy.deepcopy(real_mocap_sys_data)
        # Removes mocap offset for the height of the CoM
        self.ini_pos = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][:3]
        mocap_sys_data[:,2] -= MOCAP_MARKER_Z_OFFSET_POSITIVE
        mocap_sys_data[:,0] = mocap_sys_data[:,0] - self.mocap_local.initial_offset_com_xy[0] + self.ini_pos[0]
        mocap_sys_data[:,1] = mocap_sys_data[:,1] - self.mocap_local.initial_offset_com_xy[1] + self.ini_pos[1]  
        # Removes the initial difference in xy from the reference so the agent matches the mocap
        # CoM
        obs[:, :3] = mocap_sys_data[:, :3]
        # Rotation Matrix of the robot, 9 coefficients afyer CoM, and jp.
        obs[:, 3+12:3+12+9] = mocap_sys_data[:, 3:3+9]
        # Linear vel
        obs[:,3+12+9:3+12+9+3] = mocap_sys_data[:, 3+9+3+4+1+1:3+9+3+4+1+1+3]
        # Angular vel
        obs[:,3+12+9+3:3+12+9+3+3] = mocap_sys_data[:, 3+9+3+4+1+1+3:3+9+3+4+1+1+3+3]
        # TODO I am not using the occlusion flag at the end of the mocap stream for now
        return obs
    
    #def mocap_calibrate(self):
    # TODO remove initial position from the mocap system


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


class Filter:

    def __init__(self):
        self.obs_list = []

    def __call__(self, obs):
        self.obs_list.append(obs)
        if len(self.obs_list) > 1:
            obs = (obs + self.obs_list[-2])/2.0
        return obs

class MocapSystem:

    def __init__(self):
        self.positions = []
        self.initial_offset_com_xy = None

    def store(self, data):
        self.positions.append(data[:, :3].flatten())

    def calibrate(self):
        print("Calibrating mocap system ...")
        positions_np = np.array(self.positions)
        x_offset = np.mean(positions_np[:,0])
        y_offset = np.mean(positions_np[:,1])
        print(f"Calibration offsets: {x_offset}, {y_offset}")
        self.initial_offset_com_xy = [x_offset, y_offset]

# Client loop.
if __name__ == "__main__":
    input("PRESS ENTER IF YOU WANT TO START THE LAPTOP SERVER ....")
    policy = LaptopPolicy()
    try:
        policy.inference_loop()
    except Exception as e:
        print(f"Exception : {e}")
        policy.write_data_to_csv()
