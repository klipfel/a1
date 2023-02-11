import numpy as np
import pandas as pd
import copy
from absl import logging
import os
import sys
import subprocess
import inspect
import argparse

from tqdm import tqdm
import time
import datetime
# Pybullet.
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
import wandb

subprocess = subprocess.Popen("echo $HOME", shell=True, stdout=subprocess.PIPE)
HOME = subprocess.stdout.read()
HOME = str(HOME)
# Motion imitation wrapper
if HOME.find('unitree')!=-1:  # adds the path to the local motion_imitation wrapper installation.
    os.sys.path.append("/home/unitree/arnaud/motion_imitation")
from motion_imitation.robots import robot_config, a1

from utilities.config import Config
from utilities.motion_imitation_config import MotionImitationConfig
from utilities.logging import Logger

from motion_clips.motionClip import MotionClipParser, raisim_rotmat_to_quat

import torch
from torch.distributions import Normal
import torch.nn as nn


# Remote client
import Pyro5.api
from remote.util import recover_data, adapt_data_for_comm

LINE = "-"*100


def error(x, y):
    np.linalg.norm(np.array(x-y))


def save_single_data_to_csv( data, name, folder):
    df = pd.DataFrame(data)
    df.to_csv(f"{folder}/{name}.csv")


def raisim_quat_to_pybullet(quat):
    assert type(quat) is list
    # [w,x,y,z] -> [x,y,z,w]
    w = quat.pop(0)
    return quat + [w]

class ControlFramework:

    def __init__(self,):
        # TODO add argcomplete for autocompletion in terminal.
        # TODO Fix other boolean parameters have to use store_true or store_false.
        # TODO use kwargs to automate a bit or make it more general
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--visualize", action='store_true', help='Activates the rendering in sim mode when present.')
        # TODO rack in hardware mode.
        parser.add_argument("-r", "--rack", help='rack boolean. If true the robot is considered to be on a rack. For now only in simulation', action='store_true')
        parser.add_argument("-t", "--test_type", help='Type of the test: static.', type=str, default="static")
        parser.add_argument("-m", "--mode", help='sim or hdw', type=str, default="sim")
        parser.add_argument("--kp", help='Proportional for thigh and calf.', type=float, default=50.0)
        parser.add_argument("--kp_policy", help='Proportional for thigh and calf.', type=float, default=50.0)
        parser.add_argument("--kp_policy_list", help='Proportional for all joints.', nargs='+', type=float, default=None)
        parser.add_argument("--kpa", help='Proportional for hip.', type=float, default=50.0)
        parser.add_argument("--kd_policy", help='Derivative for thigh and calf.', type=float, default=2.0)
        parser.add_argument("--kd_policy_list", help='Derivative for all joints.', nargs='+', type=float, default=None)
        parser.add_argument("--kd", help='Derivative for thigh and calf.', type=float, default=2.0)
        parser.add_argument("--kda", help='Derivative for hip.', type=float, default=2.0)
        parser.add_argument("--dt", help="Control time step.", type=float, default=0.02)
        parser.add_argument("--dt_policy", help="Control time step for the policy test.", type=float, default=0.001)
        parser.add_argument("--time_step", help="Control time step between two repeated commands when calling the Step function.", type=float, default=0.001)
        parser.add_argument("--max_policy_dt", help="Sets the maximum time in seconds between two sampling from the policy. Used in the adaptive controller.", type=float, default=0.020)
        parser.add_argument("--nsteps", help="Total control steps to reach joint position.", type=int, default=200)
        parser.add_argument("--nrepeat", help="Number of steps to control to intermediary targets between policy commands.", type=int, default=5)
        parser.add_argument("--sp", help="Smoothing percentage.", type=float, default=2/3)
        parser.add_argument("--sjt", nargs="+", help="Single joint target specification for one leg.", type=float, default=None)
        parser.add_argument("-w", "--weight", help="pre-trained weight path", type=str, default=Config.WEIGHT_PATH)
        # parser.add_argument("-obsn", "--obs_normalization", help="Normalize or not observations based on the data accumulated in Raisim.", type=bool, default=Config.OBS_NORMALIZATION)
        parser.add_argument("-obsn", "--obs_normalization", help="Normalize or not observations based on the data accumulated in Raisim.", action='store_true')
        parser.add_argument("-rh", "--run_hdw", action='store_true', help="Apply actions on hardware.")
        parser.add_argument("-ps", "--policy_synch_sleep", action='store_true', help="Synchronization of the policy control time step with sleep calls.")
        parser.add_argument("-ac", "--adaptive_controller", action='store_true', help="If present the flag enables to select the AdaptiveController class.")
        parser.add_argument("-fic", "--fixed_interpolation_controller", action='store_true', help="If present the flag enables to select the FixedInterpolationController class.")
        parser.add_argument("--fic_policy_dt", help="Target control time step for policy sampling.", type=float, default=0.025)
        parser.add_argument("--fic_ll_dt", help="Target control time step between two repeated commands when calling the Step function.", type=float, default=0.003)
        parser.add_argument("-arp", "--action_repeat", help="Repeats the action applied on hardware.", type=int, default=1)
        parser.add_argument("-u", "--uri", help="URI of the proxy of the Policy object", type=str, default=None)
        parser.add_argument("--wandb", help='If present as an arg the model will be downloaded from wandb directly.', action='store_true')
        parser.add_argument('--run_path', help='wandb run path entity/project/run_id.', type=str, default='')
        parser.add_argument('--update', help='update number of the model to test', type=int, default=None)
        parser.add_argument('--motion_clip_folder', help='path of the motion clip folder', type=str, default='')
        parser.add_argument('--motion_clip_name', help='Name of the motion clip interpolation file.', type=str, default='')
        parser.add_argument('--imitation_policy', help='Activates the imitation policy.', action='store_true')
        parser.add_argument('--obs_mode', help='update number of the model to test', type=int, default=1)
        parser.add_argument("--no_control", help='If flag is present the actions of the policy are inferred but the robot is not controlled, '
                                                 'the actions are not applied..', action='store_true')

        # Folder where the test data will be saved
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.test_data_folder = Config.LOGDIR + "/" + date
        if not os.path.exists(self.test_data_folder):
            os.makedirs(self.test_data_folder)

        args = parser.parse_args()
        logging.set_verbosity(logging.INFO)
        logging.info("WARNING: this code executes low-level controller on the robot.")
        logging.info("Make sure the robot is hang on rack before proceeding.")
        input("Press enter to continue...")
        # Construct sim env and real robot
        is_sim_env = args.mode == "simEnv"
        is_sim_gui = args.mode == "simGui"
        is_hdw = args.mode == "hdw"
        nsteps = args.nsteps
        # Gains.
        KP = args.kp
        KD = args.kd
        KPA = args.kpa
        KDA = args.kda
        # Policy setup.
        if args.uri is not None:
            self.policy = RemotePolicyAdapter(args.uri)
        else:
            if args.imitation_policy:
                self.policy = ImitationPolicy(args, folder=self.test_data_folder)
            else:
                self.policy = Policy(args, folder=self.test_data_folder)

        # Creates a simulation using a gym environment.
        if is_sim_env:
            from motion_imitation.robots import a1
            from motion_imitation.envs import env_builder # moved it here since it also imports tensforflow. Don't need this on
            # Motion imitation wrapper.
            # the hardware.
            # Create an environment for simulation.
            # TODO why is the environment like if the robot was always on a rack? I used another approach in sim2. It does
            # TODO not use an environment.
            env = env_builder.build_regular_env(
                robot_class=a1.A1,  # robot class for simulation
                motor_control_mode=robot_config.MotorControlMode.POSITION,
                on_rack=args.rack,
                enable_rendering=args.visualize,
                wrap_trajectory_generator=False)
            robot = env.robot
            self.env = env
        # HDW.
        elif is_hdw:
            # Motion imitation wrapper.
            os.sys.path.append("/home/unitree/arnaud/motion_imitation")
            from motion_imitation.robots import a1_robot  # imports the robot interface in the case where the code is
            # run on hardware.
            # No environment is needed for hardware tests.
            p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # Hardware class for the robot. (wrapper)
            robot = a1_robot.A1Robot(pybullet_client=p,
                                     action_repeat=args.action_repeat,
                                     time_step=args.time_step,
                                     control_latency=0.002)  # TODO wtf?
            robot.motor_kps = np.array([KPA,KP,KP] * 4)
            robot.motor_kds = np.array([KDA,KD,KD] * 4)
            print("Robot Kps: ", robot.motor_kps)
            print("Robot Kds: ", robot.motor_kds)
        # simulation using the pybullet GUI, no gym environment. Does not use tf, or any learning.
        elif args.mode == "simGui":
            from motion_imitation.robots import a1
            if args.visualize:
                p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
            else:
                p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            # p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            num_bullet_solver_iterations = 30
            p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setPhysicsEngineParameter(numSolverIterations=30)
            simulation_time_step = args.time_step  # TODO understand how that is translated on the hdw. Makes the robot fail
            # TODO as on the real hdw when set to > 0.003 s
            # TODO how does pybullet deal with real time and time steps?
            p.setTimeStep(simulation_time_step)
            p.setGravity(0, 0, -9.8)
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
            """
            action_repeat: The number of simulation steps that the same action is
            repeated.
            TODO what is the equivalent on hardware?
            """
            robot = a1.A1(pybullet_client=p,
                          on_rack=args.rack,
                          action_repeat=args.action_repeat,
                          time_step=simulation_time_step,  # time step of the simulation
                          control_latency=0.0,
                          enable_action_interpolation=True,
                          enable_action_filter=False)
            motor_kps = np.array([KPA,KP,KP] * 4)
            motor_kds = np.array([KDA,KD,KD] * 4)
            robot.SetMotorGains(motor_kps, motor_kds)
            gains = robot.GetMotorGains()
            print("Robot Kps:", gains[0])
            print("Robot Kds:", gains[1])
            if args.visualize:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        else:
            logging.error("ERROR: unsupported mode. Either sim or hdw.")
        robot.ReceiveObservation()
        # Class variables.
        self.robot = robot
        self.args = args
        self.is_sim_env = is_sim_env
        self.is_sim_gui = is_sim_gui
        self.is_hdw = is_hdw
        if args.imitation_policy:
            self.motion_clip_parser = MotionClipParser(data_folder=self.test_data_folder)
            self.obs_parser = MotionImitationObservationParser(self.robot, self.args, self.policy,
                                                               motion_clip_parser=self.motion_clip_parser,
                                                               data_folder=self.test_data_folder)
            self.leg_bounds = {"hip": [-0.5, 0.5], "thigh": [-0.1, 1.5], "calf": [-2.1, -0.5]}
            self.action_bridge = MotionImitationActionBridge(self.robot, leg_bounds=self.leg_bounds)
            self.ini_conf = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][-12:]
            self.ini_base_state = self.motion_clip_parser.motion_clip["Interp_Motion_Data"][0][:7]
            self.ini_com = self.ini_base_state[:3]
            self.ini_orn = self.ini_base_state[3:]
            # Sets the initial state of the robot to be the initial motion clip state.
            # TODO on hdw you will need to either randomize the init state more or just change obs?
            if not is_hdw:
                a1.INIT_POSITION = self.ini_base_state[:3]
                a1.INIT_ORIENTATION = self.ini_base_state[3:]
                a1.INIT_MOTOR_ANGLES = self.ini_conf
                print(f"init position : {a1.INIT_POSITION}, init orientation : {a1.INIT_ORIENTATION}")
                # self.robot.Reset(reload_urdf=False)
                pybullet.resetBasePositionAndOrientation(self.robot.quadruped, self.ini_com, self.ini_orn)
                # TODO implement a proper initialization
                # pybullet.resetJointStatesMultiDof(self.robot.quadruped, self.ini_conf)
                print(f"Initial state to track in the reference: {self.ini_conf}......")
                # Builds a  reference model
                self.robot_ref_model = self.build_ref_model(pybullet_client=p,
                                     urdf_file=a1.URDF_FILENAME)
        else:
            self.policy_dir, self.policy_it = self.policy_info_from_dir_path()
            self.obs_parser = ObservationParser(self.robot, self.args,
                                                policy_dir=self.policy_dir,
                                                policy_iteration=self.policy_it)
            self.action_bridge = ActionBridge(self.robot)
            self.ini_conf = Config.INI_JOINT_CONFIG
        # Buffers.
        self.policy_dt_buffer = []
        self.last_action_time_buffer = []
        self.last_state_time_buffer = []
        # Logger.
        if args.obs_normalization:
            self.logger = Logger(args=self.args,
                                 folder=self.test_data_folder,
                                 obs_ref=self.obs_parser.obs_buffer,
                                 obsn_ref=self.obs_parser.obsn_buffer,
                                 action_policy_ref=self.action_bridge.action_policy_buffer,
                                 action_ref=self.action_bridge.action_buffer,
                                 policy_dt_ref=self.policy_dt_buffer,
                                 last_action_time_ref=self.last_action_time_buffer,
                                 last_state_time_ref=self.last_state_time_buffer
                                 )
        else:
            self.logger = Logger(args=self.args,
                                 folder=self.test_data_folder,
                                 obs_ref=self.obs_parser.obs_buffer,
                                 action_policy_ref=self.action_bridge.action_policy_buffer,
                                 action_ref=self.action_bridge.action_buffer,
                                 policy_dt_ref=self.policy_dt_buffer,
                                 last_action_time_ref=self.last_action_time_buffer,
                                 last_state_time_ref=self.last_state_time_buffer
                                 )
        # p.resetBasePositionAndOrientation(self.robot.ObjectId(), self.ini_base_state[:3], self.ini_base_state[3:])
        self.pybullet_client = p

    def reset(self):
        '''
        Resets the robot to its initial state.
        Be carefulto not use it on hdw
        :return:
        '''
        if not self.is_hdw:
            self.robot.Reset(reload_urdf=False)
            # self.robot.quadruped

    def policy_info_from_dir_path(self):
        components = self.args.weight.split('/')
        policy_dir = self.args.weight[:-len(components[-1])]
        policy_it = ''
        for e in components[-1]:
            if e.isdigit():
                policy_it += e
        policy_it = int(policy_it)
        return policy_dir, policy_it

    def process_single_joint_target(self):
        """Process the single joint target specification."""
        sjt = None
        if self.args.sjt is None:
            sjt = np.array([0., 1.0, -1.8] * 4)
        else:
            assert(len(self.args.sjt) == 3)
            sjt = np.array(self.args.sjt*4)
        print("Single joint target set to: ", sjt)
        return sjt

    def set_pd_gains(self, motor_kps=np.array([100.0] * 12), motor_kds=np.array([2.0] * 12)):
        # TODO check if it work on hdw.
        self.robot.SetMotorGains(motor_kps, motor_kds)
        gains = self.robot.GetMotorGains()
        print("Setting motor gains....")
        print("Robot Kps:", gains[0])
        print("Robot Kds:", gains[1])

    def go_to_initial_configuration(self, alpha=0.8, nsteps=2000, dt=0.005):
        """
        Sets the robot in an initial configuration. Preferably close to the ones the robot was trained on at the start
        of the training episodes. Prepares the robot for policy.
        :param alpha: during 0.8*steps the robot will gradually be guided to the desired_motor_angle, and during 0.2*n_steps
        it will be asked to go there directly. First step: transition and then once the joint configuration is not too
        far the robot is controlled to it.
        """
        print(LINE)
        print("PREPARES ROBOT FOR POLICY...")
        print(f"Setting joint positions to: {self.ini_conf}")
        current_motor_angle = np.array(self.robot.GetTrueMotorAngles())
        print("Current joint positions:", current_motor_angle)
        for t in tqdm(range(nsteps)):
            blend_ratio = np.minimum(t / (nsteps*alpha), 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * self.ini_conf
            if self.is_sim_env:
                self.env.step(action)
            elif self.is_hdw or self.is_sim_gui:
                self.robot.Step(action, robot_config.MotorControlMode.POSITION)
            else:
                logging.error("ERROR: unsupported mode. Either sim or hdw.")
            time.sleep(dt)  # the example used 0.005.
        print(LINE)

    def run(self):
        print(LINE)
        print("Running the policy....")
        if self.args.kp_policy_list is not None and self.args.kd_policy_list is not None:
            self.set_pd_gains(motor_kps=np.array([self.args.kp_policy_list]),
                              motor_kds=np.array([self.args.kd_policy_list]))
        else:
            self.set_pd_gains(motor_kps=np.array([self.args.kp_policy] * 12),
                              motor_kds=np.array([self.args.kd_policy] * 12))
        if self.args.adaptive_controller:
            self.controller = AdaptiveController(self)
            self.controller.control()
        elif self.args.fixed_interpolation_controller:
            self.controller = FixedInterpolationController(self)
            print(self.controller)
            self.controller.control()
        elif self.args.imitation_policy:
            print("Motion imitation policy control mode. The policy will start tracking the motion .... please press enter..."
                  "to proceed")
            input()
            self.controller = MotionImitationSimpleController(self)
            self.controller.control()
        else:
            for _ in tqdm(range(self.args.nsteps)):
                # Time measurements
                times = []
                last_action_times = []
                last_state_times = []
                action_dt = []  # time between action application
                state_dt = []  # time between obs readings.
                t0 = time.time()
                obs = self.observe()
                tinf0 = time.time()  # measures inference times.
                action_np = self.policy.inference(obs)
                tinf1 = time.time()
                times.append(tinf0-t0)  # sensor reading time.
                times.append(tinf1-tinf0)  # policy inference time.
                action_robot = self.action_bridge.adapt(action_np)
                # Adds residual to nomimal configuration.
                joint_target = action_robot.flatten() + self.ini_conf
                current_motor_angle = np.array(self.robot.GetTrueMotorAngles())
                for k in range(self.args.nrepeat):
                    t10 = time.time()

                    blend_ratio = np.minimum(k / (self.args.nrepeat-1), 1)
                    if self.args.run_hdw:
                        intermediary_joint_target = (1 - blend_ratio) * current_motor_angle + blend_ratio * joint_target
                    else:
                        intermediary_joint_target = Config.INI_JOINT_CONFIG
                    # Store time data on obsrevations and actions
                    # last_action_times.append(self.robot.last_action_time)
                    # last_state_times.append(self.robot.last_state_time)
                    # if len(last_action_times) > 1:
                    #     action_dt.append(last_action_times[-1]-last_action_times[-2])
                    #     state_dt.append(last_state_times[-1]-last_state_times[-2])
                    # else:
                    #     action_dt.append(last_action_times[-1])
                    #     state_dt.append(last_state_times[-1])
                    # Applies commands.
                    if self.is_sim_env:
                        self.env.step(intermediary_joint_target)
                    elif self.is_hdw or self.is_sim_gui:
                        self.robot.Step(intermediary_joint_target, robot_config.MotorControlMode.POSITION)
                    else:
                        logging.error("ERROR: unsupported mode. Either sim or hdw.")

                    time.sleep(self.args.dt_policy)
                    t11 = time.time()
                    measured_repeat_dt = t11-t10
                    times.append(measured_repeat_dt)

                t1 = time.time()
                measured_policy_dt = t1-t0
                # Waits the necessary time to match the policy control frequency in case the sampling was too fast.
                # In case the sampling is too slow nothing can be done. You should set other parameters in order to
                # Have a faster control that you desire and then use some wait to synchronize.
                # TODO add the delay in the repeat control loops.
                # TODO Rather than just waiting you could also add other intermedary control until a min time is reached
                policy_time_to_wait = 0.020-measured_policy_dt
                if policy_time_to_wait > 0 and self.args.policy_synch_sleep:
                    time.sleep(policy_time_to_wait)
                t2 = time.time()
                # Stores policy control time data.
                times.append(measured_policy_dt)
                times.append(policy_time_to_wait)
                times.append(t2-t0)  # stores the policy control time + wait time.
                # Adds to buffer.
                self.policy_dt_buffer.append(np.array(times))
                # self.last_action_time_buffer.append(np.array(action_dt))
                # self.last_state_time_buffer.append(np.array(state_dt))
        print(LINE)

    def apply_action(self, action):
        # Applies commands.
        if self.args.no_control:
            print("Control is not activated.")
        else:
            if self.is_sim_env:
                self.env.step(action)
            elif self.is_hdw or self.is_sim_gui:
                self.robot.Step(action, robot_config.MotorControlMode.POSITION)
            else:
                logging.error("ERROR: unsupported mode. Either sim or hdw.")

    def observe(self):
        """Returns the agent observations."""
        return self.obs_parser.observe()

    def build_ref_model(self, pybullet_client, urdf_file):
        """Constructs simulated model for playing back the reference motion.
        Returns:
          Handle to the simulated model for the reference motion.
        Source :
        https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/motion_imitation/envs/env_wrappers/imitation_task.py#L553
        """
        ref_col = [1, 1, 1, 0.5]
        ref_model = pybullet_client.loadURDF(urdf_file, useFixedBase=True)
        pybullet_client.changeDynamics(ref_model, -1, linearDamping=0, angularDamping=0)

        pybullet_client.setCollisionFilterGroupMask(
            ref_model, -1, collisionFilterGroup=0, collisionFilterMask=0)

        pybullet_client.changeDynamics(
            ref_model,
            -1,
            activationState=pybullet_client.ACTIVATION_STATE_SLEEP +
            pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
            pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)

        pybullet_client.changeVisualShape(ref_model, -1, rgbaColor=ref_col)

        num_joints = pybullet_client.getNumJoints(ref_model)
        num_joints_sim = pybullet_client.getNumJoints(self.robot.quadruped)
        assert (
            num_joints == num_joints_sim
        ), "ref model must have the same number of joints as the simulated model."

        for j in range(num_joints):
            pybullet_client.setCollisionFilterGroupMask(
              ref_model, j, collisionFilterGroup=0, collisionFilterMask=0)

            pybullet_client.changeDynamics(
              ref_model,
              j,
              activationState=pybullet_client.ACTIVATION_STATE_SLEEP +
              pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
              pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)

            pybullet_client.changeVisualShape(ref_model, j, rgbaColor=ref_col)

        return ref_model

    def set_state(self, pybullet_client, phys_model, position, orn, joint_positions):
        """Set the state of a character to the given pose and velocity. here the joints are considered to have
        only 1DOF.
        Args:
          phys_model: handle of the character
          pose: pose to be applied to the character
          vel: velocity to be applied to the character
          Source:
          https://github.com/erwincoumans/motion_imitation/blob/d0e7b963c5a301984352d25a3ee0820266fa4218/motion_imitation/envs/env_wrappers/imitation_task.py#L747
        """
        pybullet_client.resetBasePositionAndOrientation(phys_model, position, orn)
        # pybullet_client.resetBaseVelocity(phys_model, root_vel, root_ang_vel)
        joint_name_to_id = self.BuildJointNameToIdDict(phys_model, pybullet_client)
        num_joints = len(a1.MOTOR_NAMES)
        # assert (num_joints == len(joint_positions)), f"{num_joints} =/= {len(joint_positions)}"
        for name, i in zip(a1.MOTOR_NAMES, range(len(a1.MOTOR_NAMES))):
            pybullet_client.resetJointState(phys_model,
                                            joint_name_to_id[name],
                                            joint_positions[i],
                                            targetVelocity=0)

    def BuildJointNameToIdDict(self, phys_model, pybullet_client):
        # TODO write a wrapper class for a1 and the reference so you can put all this information in.
        num_joints = pybullet_client.getNumJoints(phys_model)
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = pybullet_client.getJointInfo(phys_model, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        return joint_name_to_id

class AdaptiveController:
    # TODO timer class?
    # TODO a function to just accumulate the timing data
    # TODO avoid using function calls for rapid control?
    # TODO compute blend_ration, generate int joint targets
    # TODO class for latency model?
    """
    Adaptive controller for the hardware. This controller does not use sleep between each motor calls, and
    uses the maximum control bandwidth. Interpolation steps between 2 policy targets can vary.
    """
    def __init__(self, cf):
        self.cf = cf
        self.robot = cf.robot
        self.args = cf.args
        self.policy_nsteps = self.args.nsteps
        self.last_policy_dt = 0.0
        self.policy_dt_buffer = []
        self.max_policy_dt = cf.args.max_policy_dt  # in seconds.
        self.min_interpolation_steps = 2
        self.policy_loop_timer = Timer()
        self.policy_target_buffer = [Config.INI_JOINT_CONFIG]
        self.initial_blend_ratio = 1.0/300.0  # TODO evaluate that online? For now chosen very small.
        self.interpolation_counter = 0
        self.last_joint_target = None
        self.last_interpolation_origin = None
        self.last_control_dt = None
        self.control_dt_loop = []  # control dt for one joint target.
        self.control_dt_buffer = []  # control dt for all the joint targets so far.
        self.last_interpolation_steps = 100  # random init.
        self.time_left_before_new_target = None
        # Buffers.
        self.blend_ratio_buffer = []
        self.current_blend_ratio = []  # for current policy joint target
        self.interpolation_step_buffer = []
        self.interpolation_steps = []
        self.motor_torque_buffer = []
        self.contact_boolean_buffer = []
        self.joint_position_buffer = []
        self.base_position_buffer = []
        self.base_velocity_buffer = []

    def control(self):
        # TODO add a sleep between each control command sent.
        for policy_step in tqdm(range(self.policy_nsteps)):
            self.policy_loop_timer.start()
            # Sample a new action from the policy.
            obs = self.cf.observe()
            self.policy_loop_timer.checkpoint("observation")
            action_np = self.cf.policy.inference(obs)
            action_robot = self.cf.action_bridge.adapt(action_np)
            # Adds residual to nomimal configuration.
            self.last_joint_target = action_robot.flatten() + self.cf.ini_conf
            self.last_interpolation_origin = self.policy_target_buffer[-1]
            self.policy_target_buffer.append(self.last_joint_target)
            self.policy_loop_timer.checkpoint("inference")
            # Generate intermediary joint targets
            self.last_policy_dt = self.policy_loop_timer.current_deltas[-1]
            while self.last_interpolation_steps > 0:
                self.interpolation_counter += 1
                # Interpolation.
                intermediary_joint_target = self.interpolate()
                self.cf.apply_action(intermediary_joint_target)
                # Sensor reading.
                self.record_sensors()
                self.policy_loop_timer.checkpoint("interpolation")
                # Diagnosis.
                # TODO measure the inter deltas, to get the while loop duration.
                self.last_control_dt = self.policy_loop_timer.current_deltas[-1] - self.last_policy_dt
                self.sleep()
                self.control_dt_loop.append(self.last_control_dt)
                self.last_policy_dt = self.policy_loop_timer.current_deltas[-1]
                self.time_left_before_new_target = self.max_policy_dt - self.last_policy_dt
            # Compute policy time
            self.policy_loop_timer.end("policy target end loop")
            self.last_policy_dt = self.policy_loop_timer.current_deltas[-1]
            # Buffer storage.
            self.blend_ratio_buffer.append(self.current_blend_ratio)
            self.control_dt_buffer.append(self.control_dt_loop)
            self.cf.policy_dt_buffer.append(self.last_policy_dt)
            self.interpolation_step_buffer.append(self.interpolation_steps)
            # Another turn.
            self.last_policy_dt = 0.0
            self.last_interpolation_steps = 100
            self.interpolation_counter = 0
            self.current_blend_ratio = []
            self.control_dt_loop = []
            self.interpolation_steps = []
        self.save_data()

    def record_sensors(self):
        self.motor_torque_buffer.append(list(self.robot.GetTrueMotorTorques()))
        self.contact_boolean_buffer.append(self.robot.GetFootContacts())
        self.joint_position_buffer.append(self.robot.GetTrueMotorAngles())
        self.base_position_buffer.append(self.robot.GetBasePosition())
        self.base_velocity_buffer.append(self.robot.GetBaseVelocity())

    def sleep(self):
        pass

    def save_data(self):
        # Logs data
        # TODO if you want to see the data in nice column format you need to give to savetxt a list of arrays which
        # TODO has the same size. Mine are changing.
        self.cf.logger.log_now("control_dt_times", np.array(self.control_dt_buffer), fmt='%s', extension="csv")
        self.cf.logger.log_now("blend_ratios", np.array(self.blend_ratio_buffer), fmt='%s', extension="csv")
        self.cf.logger.log_now("deltas", np.array(self.policy_loop_timer.delta_history), fmt='%s', extension="csv")
        self.cf.logger.log_now("inter_deltas", np.array(self.policy_loop_timer.inter_delta_history), fmt='%s', extension="csv")
        self.cf.logger.log_now("interpolation_steps", np.array(self.interpolation_step_buffer), fmt='%s', extension="csv")
        self.cf.logger.log_now("motor_torques", np.array(self.motor_torque_buffer), fmt='%1.5f', extension="csv")
        self.cf.logger.log_now("contact_states", np.array(self.contact_boolean_buffer), fmt='%1.5f', extension="csv")
        self.cf.logger.log_now("motor_angles", np.array(self.joint_position_buffer), fmt='%1.5f', extension="csv")
        self.cf.logger.log_now("base_position", np.array(self.base_position_buffer), fmt='%1.5f', extension="csv")
        self.cf.logger.log_now("base_velocity", np.array(self.base_velocity_buffer), fmt='%1.5f', extension="csv")

    def interpolate(self):
        # TODO what current motor angles should I use? I opt for the previous policy target and not for the
        # TODO current joint angles.
        # Blend ratio.
        if self.interpolation_counter == 1:  # For the first interpolation a very small step is done.
            # Used to evaluate the control time step.
            blend_ratio = self.initial_blend_ratio
        else:  # TODO use an average of previous control dt to estimate the control dt.
            self.last_interpolation_steps = int(self.time_left_before_new_target//self.last_control_dt)
            self.interpolation_steps.append(self.last_interpolation_steps)
            if self.last_interpolation_steps < self.min_interpolation_steps:
                blend_ratio = 1.0
            else:
                blend_ratio = 1.0 / self.last_interpolation_steps
        self.current_blend_ratio.append(blend_ratio)
        # Interpolation
        if self.args.run_hdw:
            intermediary_joint_target = (1 - blend_ratio) * self.last_interpolation_origin\
                                        + blend_ratio * self.last_joint_target
        else:
            intermediary_joint_target = Config.INI_JOINT_CONFIG
        self.last_interpolation_origin = intermediary_joint_target
        return intermediary_joint_target


class FixedInterpolationController(AdaptiveController):
    """
    This adaptive controller synchronizes the sampling of the policy to a fixed frequency, and syncrhonizes also each
    interpolation steps between 2 policy targets.
    """
    def __init__(self, cf):

        super(FixedInterpolationController, self).__init__(cf)

        self.target_policy_dt = cf.args.fic_policy_dt
        self.target_low_level_control_dt = cf.args.fic_ll_dt
        self.target_interpolation_number = int(self.target_policy_dt//self.target_low_level_control_dt)
        self.interpolation_counter = 1
        self.max_control_delay = 1e-4
        self.skip_nsteps = 0
        self.reached_policy_target = False

    def __str__(self):
        return(f"Fixed interpolation controller:\n Policy dt:{self.target_policy_dt}"
               f"\n LL dt: {self.target_low_level_control_dt}"
               f"\nNumber of interpolation steps: {self.target_interpolation_number}")

    def control(self):
        # TODO add a sleep between each control command sent.
        for policy_step in tqdm(range(self.policy_nsteps)):
            self.policy_loop_timer.start()
            # Sample a new action from the policy.
            obs = self.cf.observe()
            self.policy_loop_timer.checkpoint("observation")
            action_np = self.cf.policy.inference(obs)
            action_robot = self.cf.action_bridge.adapt(action_np)
            # Adds residual to nomimal configuration.
            self.last_joint_target = action_robot.flatten() + self.cf.ini_conf
            self.last_interpolation_origin = self.policy_target_buffer[-1]
            self.policy_target_buffer.append(self.last_joint_target)
            self.policy_loop_timer.checkpoint("inference")
            # Generate intermediary joint targets
            self.last_policy_dt = self.policy_loop_timer.current_deltas[-1]
            self.time_left_before_new_target = self.target_policy_dt - self.last_policy_dt
            self.target_interpolation_number = int(self.time_left_before_new_target//self.target_low_level_control_dt)+1
            # In case the inference took too long (that can happen) then no interpolation.
            if self.target_interpolation_number == 0:
                logging.warning("Not enough time left in the policy control loop. Control in one step.")
                self.target_interpolation_number = 1
            if self.last_policy_dt > 0.005:
                logging.warning(f"[POLICY STEP {policy_step}] Inference takes longer than usual: {self.last_policy_dt} s.")
            while not self.reached_policy_target:
                self.policy_loop_timer.checkpoint("interpolation")
                # Interpolation.
                intermediary_joint_target = self.interpolate()
                self.cf.apply_action(intermediary_joint_target)
                # Sensor reading
                self.record_sensors()
                self.policy_loop_timer.checkpoint("interpolation")
                # Diagnosis.
                # TODO measure the inter deltas, to get the while loop duration.
                self.last_control_dt = self.policy_loop_timer.current_inter_deltas[-1]
                # No sleep for the final control step.
                if self.interpolation_counter < self.target_interpolation_number:
                    self.sleep()
                self.control_dt_loop.append(self.last_control_dt)
                self.last_policy_dt = self.policy_loop_timer.current_deltas[-1]
                self.time_left_before_new_target = self.target_policy_dt - self.last_policy_dt
                self.interpolation_counter = min(self.interpolation_counter + 1 + self.skip_nsteps,
                                                 self.target_interpolation_number)
            # Compute policy time
            self.policy_loop_timer.end("policy target end loop")
            self.last_policy_dt = self.policy_loop_timer.current_deltas[-1]
            self.time_left_before_new_target = self.target_policy_dt - self.last_policy_dt
            # if self.time_left_before_new_target > 0.0:
            #     time.sleep(self.time_left_before_new_target)  # synchronizes policy sampling
            # Buffer storage.
            self.blend_ratio_buffer.append(self.current_blend_ratio)
            self.control_dt_buffer.append(self.control_dt_loop)
            self.cf.policy_dt_buffer.append(self.last_policy_dt)
            self.interpolation_step_buffer.append(self.interpolation_steps)
            # Another turn.
            self.last_policy_dt = 0.0
            self.interpolation_counter = 1
            self.reached_policy_target = False
            self.current_blend_ratio = []
            self.control_dt_loop = []
            self.interpolation_steps = []
        self.save_data()

    def interpolate(self):
        blend_ratio = float(self.interpolation_counter)/float(self.target_interpolation_number)
        if blend_ratio == 1.0:
            self.reached_policy_target = True
        self.interpolation_steps.append(self.target_interpolation_number)
        self.current_blend_ratio.append(blend_ratio)
        # Interpolation
        if self.args.run_hdw:
            intermediary_joint_target = (1 - blend_ratio) * self.last_interpolation_origin\
                                        + blend_ratio * self.last_joint_target
        else:
            intermediary_joint_target = Config.INI_JOINT_CONFIG
        return intermediary_joint_target
    
    def sleep(self):
        time_to_wait = self.target_low_level_control_dt - self.last_control_dt
        if time_to_wait > 0:  # sleeps if the control was faster than the target.
            time.sleep(time_to_wait)
            self.skip_nsteps = 0
        else:  # delay in the control time step.
            self.skip_nsteps = int(-time_to_wait//self.target_low_level_control_dt)  # number of steps to skip.
            logging.warning(f"Delay in the control time step: {time_to_wait} ... skips {self.skip_nsteps} control steps.")
            self.handle_control_delay(-time_to_wait)

    def handle_control_delay(self, delay):
        time_in_step = delay % self.target_low_level_control_dt
        if time_in_step <= self.max_control_delay:  # acceptable control delay.
            pass
        else:
            if self.interpolation_counter < self.target_interpolation_number-1:
                time.sleep(self.target_low_level_control_dt - time_in_step)


class MotionImitationSimpleController:

    def __init__(self, cf):
        self.cf = cf
        self.frame_target = 0 # frame target in the motion clip
        self.control_dt = 0.02
        self.sim_dt = 0.001
        self.num_frames = self.cf.motion_clip_parser.motion_clip_sim_frames
        self.frames_not_controlled = int(self.control_dt/self.sim_dt)
        self.data = {"obs": [],
                     "action_np": [],
                     "action_robot": [],
                     }
        self.data_folder = cf.test_data_folder
        self.ref_position = None
        self.ref_orn = None
        self.ref_jp = None

    def match_initial_state(self):
        '''
        Matches initial state in simulation.
        :return:
        '''
        print(LINE)
        print(f"Matching initial robot state of the robot to:\n")
        # self.cf.reset()
        # pybullet.resetBasePositionAndOrientation(self.cf.robot.quadruped,
        #                                          self.cf.ini_com,
        #                                          self.cf.ini_orn
        #                                          )
        self.cf.set_state(self.cf.pybullet_client,
                          self.cf.robot.quadruped,
                          self.cf.ini_com,
                          self.cf.ini_orn,
                          self.cf.ini_conf)
        self.cf.robot.ReceiveObservation() ## need to call receiveObs before any sensor reading
        print(f"CoM: {self.cf.robot.GetBasePosition()}/\n Orn: {self.cf.robot.GetTrueBaseOrientation()}"
              f"\n jp: {self.cf.robot.GetTrueMotorAngles()}")
        print(f"Errors : CoM: {self.cf.ini_com-self.cf.robot.GetBasePosition()}/\n Orn: {self.cf.ini_orn-self.cf.robot.GetTrueBaseOrientation()}"
              f"\n jp: {self.cf.ini_conf-self.cf.robot.GetTrueMotorAngles()}")
        print(LINE)

    def control(self):
        self.match_initial_state()
        input("TO PROCEED TO THE MOTION TRACKING......PRESS ENTER....")
        for frame in range(20, self.num_frames, self.frames_not_controlled):  #start, stop, step
            # print(f"{frame}")
            # TODO add a control for every sim step and not just control step
            # Action inference.
            self.cf.robot.ReceiveObservation()
            obs = self.cf.obs_parser.observe(target_frame=frame)
            action_np = self.cf.policy.inference(obs, std=[0.1,0.3,0.3]*4)
            action_robot = self.cf.action_bridge.adapt(action_np)
            # self.cf.robot.ReceiveObservation()
            # current_motor_angle = np.array(self.cf.robot.GetTrueMotorAngles())
            # for k in range(self.frames_not_controlled):  # the policy is not controlling for these frames
                # only for reference visualization and smoother control
                # blend_ratio = np.minimum(k / ( self.frames_not_controlled-1), 1)
                # intermediary_joint_target = (1 - blend_ratio) * current_motor_angle + blend_ratio * action_robot
                # self.cf.apply_action(intermediary_joint_target.flatten())
                # self.update_ref_model(frameIdx=(frame+k)% self.num_frames)
                # time.sleep(self.sim_dt)
            self.cf.apply_action(action_robot.flatten())
            self.update_ref_model(frameIdx=frame% self.num_frames)
            time.sleep(self.control_dt)
            self.frame_target += self.frames_not_controlled
            self.frame_target = self.frame_target % self.num_frames
            # Save data
            self.save_data(obs, action_np, action_robot)
        self.write_data_to_csv()

    def save_data(self, obs, action_np, action_robot):
        self.data["obs"].append(list(obs.flatten()))
        self.data["action_np"].append(list(action_np.flatten()))
        self.data["action_robot"].append(list(action_robot.flatten()))

    def write_data_to_csv(self):
        folder = f"{self.data_folder}/imitation_controller"
        os.makedirs(folder)
        for data_name in self.data.keys():
            save_single_data_to_csv(np.array(self.data[data_name]), data_name, folder)

    # def match_initial_reference_state(self, pybullet_client):
    #     # todo BUT NOT ESSENTIAL
    #     pybullet_client.resetBasePositionAndOrientation(self.cf.robot, self.cf.ini_base_state[:3], self.cf.ini_base_state[3:])

    def get_frame_gc(self, frameIdx, use_rotation_matrix=False):
        # gets gc in pybullet convention
        gc = self.cf.motion_clip_parser.motion_clip["Interp_Motion_Data"][frameIdx]
        self.ref_position = gc[:3]
        if use_rotation_matrix:
            # Retrive reconstructed rotmat from motion clip quaternion
            rotmat = self.cf.motion_clip_parser.motion_clip_additional_data["Rotation_matrix"][frameIdx]
            q = raisim_rotmat_to_quat(rotmat)
            self.ref_orn = raisim_quat_to_pybullet(q)
        else:
            self.ref_orn = gc[3:3+4]
        self.ref_jp = gc[-12:]

    def update_ref_model(self, frameIdx):
        self.get_frame_gc(frameIdx, use_rotation_matrix=True)
        ref_model = self.cf.robot_ref_model
        self.cf.set_state(self.cf.pybullet_client,
                          ref_model,
                          self.ref_position,
                          self.ref_orn,
                          self.ref_jp)


class Timer:

    def __init__(self):
        self.ref_time = None
        self.current_times = []
        self.total_durations = []
        self.time_history = []
        self.current_deltas = []
        self.delta_history = []
        self.delta_label = []
        self.completed = 0  # cycle counter
        self.current_inter_deltas = []
        self.inter_delta_history = []

    def start(self):
        self.reset()
        self.ref_time = time.time()
        self.current_times.append(self.ref_time)

    def reset(self):
        # resets tmp buffers.
        self.current_inter_deltas = []
        self.current_times = []
        self.current_deltas = []

    def end(self, label):
        if self.completed == 0:
            self.delta_label.append(label)
        self.completed += 1
        self.current_times.append(time.time())
        self.total_durations.append(self.total_duration())
        self.current_deltas.append(self.total_durations[-1])
        self.delta_history.append(self.current_deltas)
        self.time_history.append(self.current_times)
        self.current_inter_deltas.append(self.current_times[-1]-self.current_times[-2])
        self.inter_delta_history.append(self.current_inter_deltas)

    def total_duration(self):
        return self.current_times[-1]-self.current_times[0]

    def checkpoint(self, label):
        if self.completed == 0:
            self.delta_label.append(label)
        self.current_times.append(time.time())
        self.current_deltas.append(self.current_times[-1]-self.ref_time)
        self.current_inter_deltas.append(self.current_times[-1]-self.current_times[-2])


class RobotModel:
    """
    Class that contains data on the robot itself.
    """
    def __init__(self):
        pass


class Policy:

    def __init__(self, args, stochastic_test=False, folder=None):
        self.stochastic_test = stochastic_test
        self.weight_path = args.weight
        self.folder = folder
        from policy import ppo_module  # net architectures.
        # Inference done on the CPU.
        # TODO compare with GPU? in time
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\nTorch device: ", self.device)
        # calculate i/o dimensions of the policy net.
        self.ob_dim = Config.INPUT_DIMS
        self.act_dim = Config.OUTPUT_DIMS
        # Load policy net.
        self.loaded_graph = ppo_module.HafnerActorModelStd(self.ob_dim, self.act_dim)
        self.loaded_graph.load_state_dict(torch.load(self.weight_path, map_location=self.device)["actor_architecture_state_dict"])
        # Actions
        self.action_ll = None
        self.action_np = None

    def inference(self, obs):
        # Inference mode context manager to remove grad computation, similar to no_grad.
        # No need of the gradient for inference.
        # TODO why am I never multiplying by the std?
        with torch.inference_mode():
            action_ll = self.loaded_graph.forward(torch.from_numpy(obs).cpu())
            mean = action_ll[:, self.act_dim//2:]
            std = action_ll[:, :self.act_dim//2]
            if self.stochastic_test:
                distribution = Normal(mean, std)
                stochastic_actions = distribution.sample()
                action_np = stochastic_actions.cpu().numpy()
            else:
                action_ll = mean
                action_np = action_ll.cpu().numpy()
            self.action_ll = action_ll
            self.action_np = action_np
        return action_np


class ImitationPolicy(Policy):

    # TODO download the config file of the training so you can set hyperparameters based on that
    def __init__(self, args, folder=None, ob_dim = 342 ):
        self.folder = folder
        if args.wandb:
            # Use wandb to download the model
            print(f"Using wandb to download the policy......")
            api = wandb.Api()
            run = api.run(args.run_path)
            self.wandb_policy_folder = f"{self.folder}/{run.id}"
            self.update = args.update
            os.makedirs(self.wandb_policy_folder)
            run.file("full_"+str(args.update)+".pt").download(root=self.wandb_policy_folder, replace=True)
            run.file("mean"+str(args.update)+".csv").download(root=self.wandb_policy_folder, replace=True)
            run.file("var"+str(args.update)+".csv").download(root=self.wandb_policy_folder, replace=True)
            self.weight_path = self.wandb_policy_folder + "/full_"+str(args.update)+".pt"
            iteration_number = self.weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
            weight_dir = self.weight_path.rsplit('/', 1)[0] + '/'
            # TODO read the policy that you will use
        else:
            print("Wandb is required for now.")
        from policy import raisim_module  # net architectures.
        # Inference done on the CPU.
        # TODO compare with GPU? in time
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\nTorch device: ", self.device)
        # calculate i/o dimensions of the policy net.
        self.ob_dim = ob_dim
        print(f"Input dimension of the policy : {self.ob_dim}")
        self.act_dim = 12
        self.architecture = [256, 256]
        # Load policy net.
        self.loaded_graph = raisim_module.MLP(self.architecture, nn.LeakyReLU, self.ob_dim, self.act_dim)
        self.loaded_graph.load_state_dict(torch.load(self.weight_path, map_location=self.device)["actor_architecture_state_dict"])
        print(self.loaded_graph)
        # Actions
        self.action_ll = None
        self.action_np = None

    def inference(self, obs, std):
        with torch.inference_mode():
            action_ll = self.loaded_graph.architecture(torch.from_numpy(obs).cpu())
            action_np = action_ll.cpu().detach().numpy()
            action_ll = np.array(action_ll.flatten(), dtype=np.float64)
            action_np = np.array(action_np, dtype=np.float64)
        action_np = action_np * np.array(std)
        self.action_ll = action_ll
        self.action_np = action_np
        return action_np


class RemotePolicyAdapter:

    def __init__(self, uri):
        self.policy = Pyro5.api.Proxy(uri)     # get a Pyro proxy to the greeting object
        logging.info(f"Remote policy proxy @ {uri}")
        print(f"Remote policy proxy @ {uri}")
        self.policy._pyroBind()
        self.policy._pyroSerializer = "marshal"  # faster communication.
        self.policy._pyroTimeout = 1.5    # 1.5 seconds

    def inference(self, obs):
        obs_comm = adapt_data_for_comm(obs)
        # TODO record the inference times on the server.
        action = self.policy.inference(obs_comm)
        return recover_data(action)


class ActionBridge:

    # TODO scaling
    # TODO Filtering
    # TODO check leg order

    def __init__(self, robot, leg_bounds=Config.LEG_JOINT_BOUND):
        self.action_policy = None
        self.action = None
        self.action_policy_buffer = []
        self.action_buffer = []
        self.leg_bounds = leg_bounds

    def adapt(self, action_policy):
        self.action_policy_buffer.append(copy.deepcopy(action_policy).flatten())
        action = copy.deepcopy(action_policy)
        self.clip(action)
        self.filter(action)
        self.action_buffer.append(copy.deepcopy(action).flatten())
        return action

    def clip(self, action):
        # TODO change the clipping for the imitation policy.
        # for j, aj in enumerate(list(action.flatten())):
        action[:, Config.HIP_INDEX] = np.array([np.clip(action[:, Config.HIP_INDEX].flatten(),
                                                        -self.leg_bounds[0], self.leg_bounds[0])])
        action[:, Config.THIGH_INDEX] = np.array([np.clip(action[:, Config.THIGH_INDEX].flatten(),
                                                        -self.leg_bounds[1], self.leg_bounds[1])])
        action[:, Config.CALF_INDEX] = np.array([np.clip(action[:, Config.CALF_INDEX].flatten(),
                                                        -self.leg_bounds[2], self.leg_bounds[2])])

    def filter(self, action):
        # TODO do an average filtering as soon as there are more than 2 action and then use a window of 5.
        if len(self.action_buffer) >= Config.FILTER_WINDOW_LENGTH:
            action += sum(self.action_buffer[-Config.FILTER_WINDOW_LENGTH+1:])
            action /= Config.FILTER_WINDOW_LENGTH


class MotionImitationActionBridge(ActionBridge):

    def __init__(self, robot, leg_bounds):
        super().__init__(robot, leg_bounds)
        print(f"Leg bounds : {leg_bounds}")
        self.mean = [-0.01, 0.75,-1.5,0.01, 0.75,-1.5,-0.01, 0.75,-1.5, 0.01, 0.75, -1.5]

    def adapt(self, action_policy):
        self.action_policy_buffer.append(copy.deepcopy(action_policy).flatten())
        action = copy.deepcopy(action_policy)
        action = self.add_mean(action)
        action = self.clip(action)
        action = self.filter(action)
        action = self.clip(action)
        self.action_buffer.append(copy.deepcopy(action).flatten())
        return action

    def clip(self, action):
        action[:, Config.HIP_INDEX] = np.array([np.clip(action[:, Config.HIP_INDEX].flatten(),
                                                        a_min=self.leg_bounds["hip"][0],
                                                        a_max=self.leg_bounds["hip"][1])])
        action[:, Config.THIGH_INDEX] = np.array([np.clip(action[:, Config.THIGH_INDEX].flatten(),
                                                          a_min=self.leg_bounds["thigh"][0],
                                                          a_max=self.leg_bounds["thigh"][1])])
        action[:, Config.CALF_INDEX] = np.array([np.clip(action[:, Config.CALF_INDEX].flatten(),
                                                         a_min=self.leg_bounds["calf"][0],
                                                         a_max=self.leg_bounds["calf"][1])])
        return action

    def set_mean(self, new_mean):
        '''
        Does not update the action mean as it is fixed to a nominal configuration for the non-residual based
        policy.
        :param new_mean:
        :return:
        '''
        pass

    def add_mean(self, action):
        return action + np.array(self.mean)

    def filter(self, action):
        # TODO do an average filtering as soon as there are more than 2 action and then use a window of 5.
        if len(self.action_buffer) >= MotionImitationConfig.FILTER_WINDOW_LENGTH-1:
            action += sum(self.action_buffer[-MotionImitationConfig.FILTER_WINDOW_LENGTH+1:])
            action /= MotionImitationConfig.FILTER_WINDOW_LENGTH
        return action


class MotionImitationResidualPolicyActionBridge(MotionImitationActionBridge):

    def __init__(self, robot, leg_bounds):
        super().__init__(robot, leg_bounds)

    def adapt(self, action_policy):
        '''
        The mean is added after filtering and clipoing in the residual policy.
        :param action_policy:
        :return:
        '''
        self.action_policy_buffer.append(copy.deepcopy(action_policy).flatten())
        action = copy.deepcopy(action_policy)
        action = self.clip(action)
        action = self.filter(action)
        action = self.clip(action)
        action = self.add_mean(action)
        self.action_buffer.append(copy.deepcopy(action).flatten())
        return action

    def set_mean(self, new_mean):
        '''
        Updates the new action for residual-based policy.
        :param new_mean:
        :return:
        '''
        self.mean = new_mean


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Source: Raisim software in RaisimGymEnv.
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class ObservationParser:

    def __init__(self, robot, args, clip_obs=10., policy_dir=Config.POLICY_DIR,
                 policy_iteration=Config.POLICY_ITERATION):
        self.args = args
        self.robot = robot
        self.current_obs = None
        self.past_obs = None
        # Buffers.
        self.motor_angles_buffer = None
        self.motor_angle_rates_buffer = None
        self.rp_buffer = None
        self.obs_buffer = []
        self.obsn_buffer = []  # normalized obs if applicable.
        # Path to dir where to store temporary data.
        self.logdir = Config.LOGDIR
        # Other va.
        self.foot_positions_in_base_frame_buffer = None
        self.measurements_std_dict = {}
        self.history_shape = None
        self.obs_shape = None
        self.motor_angles = None
        self.motor_angle_rates = None
        self.rpy = None
        self.rpy_rate = None
        self.foot_positions_in_base_frame = None
        self.obs = None
        self.obsn = None
        # Obs normalization data from policy.
        self.obs_rms = RunningMeanStd(shape=[1, Config.INPUT_DIMS])
        self.clip_obs = clip_obs
        if self.args.obs_normalization:
            self.load_scaling(policy_dir, policy_iteration)

        # Construction of the observation function
        # # TODO implement the observation fucntion
        # if self.args.obs_mode==1:
        #     pass
        # else:
        #     print("No observation mode.")
        #     sys.exit(1)

    def load_scaling(self, dir_name, iteration, count=1e5):
        print(f"Observation normalization activated .... loading scaling var and mean from {dir_name}")
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        # TODO choose from another normalization data. The csv file contains normalization data for all environment.
        # TODO check if the normalization data for other env are the same. I chose the first env for now.
        # TODO these data should be the same after enough training. Is there a way to merge them?
        # TODO you could have this file genrated from real data
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32, max_rows=1)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32, max_rows=1)

    def observe(self):
        self.motor_angles = self.robot.GetTrueMotorAngles()  # in [-\pi;+\pi]
        self.motor_angle_rates = self.robot.GetTrueMotorVelocities()
        # TODO is the angular vel here the same as the one given in Raisim, they might be using quaternions ....
        # TODO but it has 3 coordinates so I guess it is the true angular vel. Difference between ang vel returned by simulation
        # TODO and the one computed.
        # TODO is the leg order the same?
        if self.args.mode == "hdw":
            self.rpy = np.array(self.robot.GetTrueBaseRollPitchYaw())
        else:
            self.rpy = self.robot.GetTrueBaseRollPitchYaw()
        self.rpy_rate = self.robot.GetTrueBaseRollPitchYawRate()
        self.foot_positions_in_base_frame = self.robot.GetFootPositionsInBaseFrame()
        # Prepares measurements for the policy.
        if self.current_obs is not None:
            tmp = copy.deepcopy(self.current_obs)
        else:
            tmp = None
        self.current_obs = np.concatenate((self.motor_angles,
                                           self.motor_angle_rates,
                                           self.rpy[:2],
                                           self.rpy_rate,
                                           self.foot_positions_in_base_frame),
                                          axis=None)
        # float32 for pytorch.
        self.current_obs = np.array([list(self.current_obs)], dtype=np.float32)
        # Put observations array in one row for policy.
        np.reshape(self.current_obs, (1, -1))
        if self.past_obs is None:  # first time reading obs.
            self.history_shape = (1, self.current_obs.shape[1]-24)
            self.past_obs = np.zeros(self.history_shape, dtype=np.float32)
        else:
            self.past_obs = tmp[:, 24:]  # removes the joint information only.
        self.obs = np.hstack((self.current_obs, self.past_obs))
        self.obs_shape = self.obs.shape
        # Store obs.
        self.obs_buffer.append(self.obs.flatten())  # flatten for logging.
        # Obs normalization.
        if self.args.obs_normalization:
            self.obsn = self.normalize(copy.deepcopy(self.obs))
            self.obsn_buffer.append(self.obsn.flatten())  # flatten for logging.
            return self.obsn
        return self.obs

    def observe_record(self):
        """"
        Records the sensor outputs, and returns obs.
        """
        # TODO add the buffer for the body angular vel.
        obs = self.observe()
        # Buffer.
        if self.motor_angles_buffer is None:
            self.motor_angles_buffer = self.motor_angles.astype(np.float32)
            self.motor_angle_rates_buffer = self.motor_angle_rates.astype(np.float32)
            self.rp_buffer = self.rpy[:2].astype(np.float32)
            self.foot_positions_in_base_frame_buffer = self.foot_positions_in_base_frame.flatten().astype(np.float32)
        else:  # adds in buffer.
            self.motor_angles_buffer = np.vstack((self.motor_angles_buffer,
                                                  self.motor_angles.astype(np.float32)))
            self.motor_angle_rates_buffer = np.vstack((self.motor_angle_rates_buffer,
                                                       self.motor_angle_rates.astype(np.float32)))
            self.rp_buffer = np.vstack((self.rp_buffer,
                                        self.rpy[:2].astype(np.float32)))
            self.foot_positions_in_base_frame_buffer = np.vstack((self.foot_positions_in_base_frame_buffer,
                                                                  self.foot_positions_in_base_frame.flatten().astype(np.float32)))
        return obs

    def normalize(self, obs):
        """
        TODO in the tester in Raisim they use the mean and var data from the training, but what do I with the robot?
        TODO the mean and var data are not accurate for the hdw.
        :return:
        """
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs, self.clip_obs)

    # TODO implement obs filtering.
    def filter(self):
        pass

    def print_obs(self):
        print("Motor angles: {}".format(self.motor_angles))
        print("Motor angle rates: {}".format(self.motor_angle_rates))
        print("RPY: {}".format(self.rpy))
        print("Relative foot positions: {}".format(self.foot_positions_in_base_frame))
        print("Policy obs: {}".format(self.obs))

    def get_obs_std(self):
        # Computes the std for the different measurement buffer.
        if self.motor_angles_buffer is not None:
            self.measurements_std_dict["Motor-Angles"] = self.motor_angles_buffer.std(axis=0)
            self.measurements_std_dict["Motor-Angle-Rates"] = self.motor_angle_rates_buffer.std(axis=0)
            self.measurements_std_dict["RP"] = self.rp_buffer.std(axis=0)
            self.measurements_std_dict["Foot-Position"] = self.foot_positions_in_base_frame_buffer.std(axis=0)

    def print_obs_std(self):
        print(self.measurements_std_dict)


class MotionImitationObservationParser(ObservationParser):
    '''
    Class for the observation of the motion imitation policy
    '''
    # TODO
    def __init__(self, robot, args, policy, motion_clip_parser, data_folder=None, clip_obs=10.):
        super().__init__(robot, args, clip_obs, policy_dir=policy.wandb_policy_folder,
                         policy_iteration=policy.update)
        self.logdir = data_folder
        self.obs_rms = RunningMeanStd(shape=[1, policy.ob_dim])
        if self.args.obs_normalization:
            self.load_scaling(policy.wandb_policy_folder, policy.update)
        self.data_folder = data_folder
        self.motion_clip_parser = motion_clip_parser
        self.args=args
        self.motion_clip_folder =args.motion_clip_folder
        self.motion_clip_name = args.motion_clip_name
        self.get_motion_clip_data()
        self.obs_window = None
        self.rel_info = True

    def get_motion_clip_data(self):
        self.motion_clip_parser.get_single_motion_clip(
            folder=self.motion_clip_folder,
            interp_file_name=self.motion_clip_name
        )

    def set_rel_info_flag(self, rel_info_flag):
        self.rel_info = rel_info_flag

    def load_scaling(self, dir_name, iteration, count=1e5):
        print(f"Observation normalization activated .... loading scaling var and mean from {dir_name}")
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        # TODO choose from another normalization data. The csv file contains normalization data for all environment.
        # TODO check if the normalization data for other env are the same. I chose the first env for now.
        # TODO these data should be the same after enough training. Is there a way to merge them?
        # TODO you could have this file genrated from real data
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)
        print(f"Loaded obs means:{self.obs_rms.mean}")
        print(f"Loaded obs vars:{self.obs_rms.var}")

    def set_obs_window(self, new_window):
        self.obs_window = new_window

    def normalize(self, obs):
        """
        TODO in the tester in Raisim they use the mean and var data from the training, but what do I with the robot?
        TODO the mean and var data are not accurate for the hdw.
        Note: this function has changed. The action clipping is not done anymore.
        :return:
        """
        # print(f"MEAN: {self.obs_rms.mean}/VAR:{self.obs_rms.var}")
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)

    def observe(self, target_frame=0):
        # Rel ino
        # Remove rel info
        rel_info = np.zeros((1, 36*3))
        # Gets the robot data
        self.get_robot_data()
        # Gets the reference data
        self.get_reference_data(target_frame=target_frame,
                                obs_window=self.obs_window)
        # Constitutes observations
        if self.rel_info:
            self.current_obs = np.concatenate((rel_info,
                                               self.robot_data,
                                               self.reference_data),
                                               axis=None)
        else:
            self.current_obs = np.concatenate((self.robot_data,
                                               self.reference_data),
                                               axis=None)
        # float32 for pytorch.
        self.obs = np.array([list(self.current_obs)], dtype=np.float64)
        # Put observations array in one row for policy.
        np.reshape(self.obs, (1, -1))
        # Store obs.
        self.obs_buffer.append(self.obs.flatten())  # flatten for logging.
        # Obs normalization.
        if self.args.obs_normalization:
            self.obsn = self.normalize(copy.deepcopy(self.obs))
            self.obsn_buffer.append(self.obsn.flatten())  # flatten for logging.
            self.obsn = np.array(list(self.obsn), dtype=np.float32)
            return self.obsn
        self.obs = np.array(list(self.obs), dtype=np.float32)
        return self.obs

    def get_robot_data(self):
        self.robot.ReceiveObservation()
        self.motor_angles = self.robot.GetTrueMotorAngles()  # in [-\pi;+\pi]
        self.motor_angle_rates = self.robot.GetTrueMotorVelocities()
        self.rpy = np.array(self.robot.GetTrueBaseRollPitchYaw())
        self.rpy_rate = self.robot.GetTrueBaseRollPitchYawRate()
        self.com = self.robot.GetBasePosition()
        self.lin_vel = self.robot.GetBaseVelocity()
        self.rotmat = np.array(pybullet.getMatrixFromQuaternion(pybullet.getQuaternionFromEuler(self.rpy))) #TODO to check
        self.robot_data = np.concatenate((self.com,
                                          self.motor_angles,
                                          self.rotmat.flatten(),
                                          self.lin_vel,
                                          self.rpy_rate,
                                          self.motor_angle_rates),
                                          axis=None)

    def get_reference_data(self, target_frame, obs_window
                           ):
        '''

        :param current_frame: refers to the current time in the motion tracking
        :param obs_window: list of the different frame wrt to the current frame
        present in the observations.
        :return:
        '''
        reference_data_in_obs = []
        for frame in obs_window:
            frame_to_get = (frame+target_frame) % self.motion_clip_parser.motion_clip_sim_frames
            reference_data_in_obs.append(list(self.motion_clip_parser.reference_data_obs[frame_to_get]))
        # self.reference_data = np.zeros((1, 24*8))
        self.reference_data = np.array(reference_data_in_obs)


class HdwMotionImitationObservationParser(MotionImitationObservationParser):
    '''
    Adaptation of the observation parser for the imitation policy on the hdw. Using the
    remote server.
    '''
    def __init__(self, robot, args, policy, motion_clip_parser, data_folder):
        super().__init__(robot, args, policy, motion_clip_parser, data_folder)
        self.remote_robot_data = None
        self.robot_data = None

    def get_robot_data(self):
        sensor_data_list = self.robot.get_sensor_data()
        sensor_data_np = np.array(sensor_data_list, dtype=np.float32)
        self.robot_data = sensor_data_np
