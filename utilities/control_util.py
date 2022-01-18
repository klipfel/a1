import numpy as np
import copy
from absl import logging
import os
import subprocess
import inspect
import argparse
from tqdm import tqdm
import time
# Pybullet.
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

subprocess = subprocess.Popen("echo $HOME", shell=True, stdout=subprocess.PIPE)
HOME = subprocess.stdout.read()
HOME = str(HOME)
# Motion imitation wrapper
if HOME.find('unitree')!=-1:  # adds the path to the local motion_imitation wrapper installation.
    os.sys.path.append("/home/unitree/arnaud/motion_imitation")
from motion_imitation.robots import robot_config

from utilities.config import Config
from utilities.logging import Logger

import torch
from torch.distributions import Normal

LINE = "-"*100


def error(x, y):
    np.linalg.norm(np.array(x-y))


class ControlFramework:

    def __init__(self,):
        # TODO Fix other boolean parameters have to use store_true or store_false.
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--visualize", action='store_true', help='Activates the rendering in sim mode when present.')
        # TODO rack in hardware mode.
        parser.add_argument("-r", "--rack", help='rack boolean. If true the robot is considered to be on a rack. For now only in simulation', type=bool, default=True)
        parser.add_argument("-t", "--test_type", help='Type of the test: static.', type=str, default="static")
        parser.add_argument("-m", "--mode", help='sim or hdw', type=str, default="sim")
        parser.add_argument("--kp", help='Proportional for thigh and calf.', type=float, default=40.0)
        parser.add_argument("--kp_policy", help='Proportional for thigh and calf.', type=float, default=40.0)
        parser.add_argument("--kpa", help='Proportional for hip.', type=float, default=40.0)
        parser.add_argument("--kd_policy", help='Derivative for thigh and calf.', type=float, default=0.5)
        parser.add_argument("--kd", help='Derivative for thigh and calf.', type=float, default=0.5)
        parser.add_argument("--kda", help='Derivative for hip.', type=float, default=0.5)
        parser.add_argument("--dt", help="Control time step.", type=float, default=0.01)
        parser.add_argument("--dt_policy", help="Control time step for the policy test.", type=float, default=0.005)
        parser.add_argument("--time_step", help="Control time step between two repeated commands when calling the Step function.", type=float, default=0.001)
        parser.add_argument("--nsteps", help="Total control steps to reach joint position.", type=int, default=200)
        parser.add_argument("--nrepeat", help="Number of steps to control to intermediary targets between policy commands.", type=int, default=5)
        parser.add_argument("--sp", help="Smoothing percentage.", type=float, default=2/3)
        parser.add_argument("--sjt", nargs="+", help="Single joint target specification for one leg.", type=float, default=None)
        parser.add_argument("-w", "--weight", help="pre-trained weight path", type=str, default=Config.WEIGHT_PATH)
        parser.add_argument("-obsn", "--obs_normalization", help="Normalize or not observations based on the data accumulated in Raisim.", type=bool, default=Config.OBS_NORMALIZATION)
        parser.add_argument("-rh", "--run_hdw", action='store_true', help="Apply actions on hardware.")
        parser.add_argument("-arp", "--action_repeat", help="Repeats the action applied on hardware.", type=int, default=1)
        args = parser.parse_args()
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
        self.policy = Policy(args)
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
                                     control_latency=0.002)
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
                                 obs_ref=self.obs_parser.obs_buffer,
                                 action_policy_ref=self.action_bridge.action_policy_buffer,
                                 action_ref=self.action_bridge.action_buffer,
                                 policy_dt_ref=self.policy_dt_buffer,
                                 last_action_time_ref=self.last_action_time_buffer,
                                 last_state_time_ref=self.last_state_time_buffer
                                 )

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
        self.set_pd_gains(motor_kps=np.array([self.args.kp_policy] * 12),
                          motor_kds=np.array([self.args.kd_policy] * 12))
        for _ in tqdm(range(self.args.nsteps)):
            # Time measurements
            times = []
            last_action_times = []
            last_state_times = []
            action_dt = []  # time between action application
            state_dt = []  # time between obs readings.
            t0 = time.time()

            obs = self.observe()
            action_np = self.policy.inference(obs)
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
            times.append(measured_policy_dt)
            # Adds to buffer.
            self.policy_dt_buffer.append(np.array(times))
            # self.last_action_time_buffer.append(np.array(action_dt))
            # self.last_state_time_buffer.append(np.array(state_dt))
        print(LINE)

    def observe(self):
        """Returns the agent observations."""
        return self.obs_parser.observe()


class Policy:

    def __init__(self, args, stochastic_test=False):
        self.stochastic_test = stochastic_test
        self.weight_path = args.weight
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
        action_ll = self.loaded_graph.forward(torch.from_numpy(obs).cpu())
        mean = action_ll[:, self.act_dim//2:]
        std = action_ll[:, :self.act_dim//2]
        if self.stochastic_test:
            distribution = Normal(mean, std)
            stochastic_actions = distribution.sample()
            action_np = stochastic_actions.cpu().detach().numpy()
        else:
            action_ll = mean
            action_np = action_ll.cpu().detach().numpy()
        self.action_ll = action_ll
        self.action_np = action_np
        return action_np


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

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        # TODO choose from another normalization data. The csv file contains normalization data for all environment.
        # TODO check if the normalization data for other env are the same. I chose the first env for now.
        # TODO these data should be the same after enough training. Is there a way to merge them?
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


