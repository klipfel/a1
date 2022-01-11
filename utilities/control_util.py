import numpy as np
import copy
from absl import logging
import os
import inspect
import argparse
from tqdm import tqdm
import time
# Pybullet.
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

from utilities.config import Config

import torch
from torch.distributions import Normal

def error(x, y):
    np.linalg.norm(np.array(x-y))


class ControlFramework:

    def __init__(self,):
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--visualize", help='visualization boolean.', type=bool, default=True)
        # TODO rack in hardware mode.
        parser.add_argument("-r", "--rack", help='rack boolean. If true the robot is considered to be on a rack. For now only in simulation', type=bool, default=True)
        parser.add_argument("-t", "--test_type", help='Type of the test: static.', type=str, default="static")
        parser.add_argument("-m", "--mode", help='sim or hdw', type=str, default="sim")
        parser.add_argument("--kp", help='Proportional for thigh and calf.', type=float, default=40.0)
        parser.add_argument("--kpa", help='Proportional for hip.', type=float, default=40.0)
        parser.add_argument("--kd", help='Derivative for thigh and calf.', type=float, default=0.5)
        parser.add_argument("--kda", help='Derivative for hip.', type=float, default=0.5)
        parser.add_argument("--dt", help="Control time step.", type=float, default=0.01)
        parser.add_argument("--nsteps", help="Total control steps to reach joint position.", type=int, default=300)
        parser.add_argument("--sp", help="Smoothing percentage.", type=float, default=2/3)
        parser.add_argument("--sjt", nargs="+", help="Single joint target specification for one leg.", type=float, default=None)
        parser.add_argument("-w", "--weight", help="pre-trained weight path", type=str, default=Config.WEIGHT_PATH)
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
            from motion_imitation.robots import robot_config
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
            robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)
            robot.motor_kps = np.array([KPA,KP,KP] * 4)
            robot.motor_kds = np.array([KDA,KD,KD] * 4)
            print("Robot Kps: ", robot.motor_kps)
            print("Robot Kds: ", robot.motor_kds)
        # simulation using the pybullet GUI, no gym environment. Does not use tf, or any learning.
        elif args.mode == "simGui":
            from motion_imitation.robots import a1
            # TODO implement that.
            p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            num_bullet_solver_iterations = 30
            p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setPhysicsEngineParameter(numSolverIterations=30)
            simulation_time_step = 0.001
            p.setTimeStep(simulation_time_step)
            p.setGravity(0, 0, -9.8)
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
            robot = a1.A1(pybullet_client=p, action_repeat=1)
            motor_kps = np.array([KPA,KP,KP] * 4)
            motor_kds = np.array([KDA,KD,KD] * 4)
            robot.SetMotorGains(motor_kps, motor_kds)
            gains = robot.GetMotorGains()
            print("Robot Kps:", gains[0])
            print("Robot Kds:", gains[1])
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
        self.obs_parser = ObservationParser(self.robot, self.args)

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

    def observe(self):
        """Returns the agent observations."""
        pass


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
        mean = action_ll[:, self.act_dim:]
        std = action_ll[:, :self.act_dim]
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


class ActionParser:

    def __init__(self, robot):
        pass


class ObservationParser:

    def __init__(self, robot, args):
        self.args = args
        self.robot = robot
        self.current_obs = None
        self.past_obs = None
        # Buffers.
        self.motor_angles_buffer = None
        self.motor_angle_rates_buffer = None
        self.rp_buffer = None
        self.foot_positions_in_base_frame_buffer = None
        self.measurements_std_dict = {}

    def observe(self):
        self.motor_angles = self.robot.GetMotorAngles()  # in [-\pi;+\pi]
        self.motor_angle_rates = self.robot.GetMotorVelocities()
        if self.args.mode == "hdw":
            self.rpy = np.array(self.robot.GetBaseRollPitchYaw())
        else:
            self.rpy = self.robot.GetBaseRollPitchYaw()
        self.foot_positions_in_base_frame = self.robot.GetFootPositionsInBaseFrame()
        # Prepares measurements for the policy.
        if self.current_obs is not None:
            tmp = copy.deepcopy(self.current_obs)
        else:
            tmp = None
        self.current_obs = np.concatenate((self.motor_angles,
                                          self.motor_angle_rates,
                                          self.rpy[:2],
                                          self.foot_positions_in_base_frame),
                                          axis=None)
        # float32 for pytorch.
        self.current_obs = np.array([list(self.current_obs)], dtype=np.float32)
        # Put observations array in one row for policy.
        np.reshape(self.current_obs, (1, -1))
        if self.past_obs is None:  # first time reading obs.
            self.past_obs = np.zeros(self.current_obs.shape)
        else:
            self.past_obs = tmp
        self.obs = np.hstack((self.current_obs, self.past_obs))
        return self.obs

    def observe_record(self):
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
        self.measurements_std_dict["Motor-Angles"] = self.motor_angles_buffer.std(axis=0)
        self.measurements_std_dict["Motor-Angle-Rates"] = self.motor_angle_rates_buffer.std(axis=0)
        self.measurements_std_dict["RP"] = self.rp_buffer.std(axis=0)
        self.measurements_std_dict["Foot-Position"] = self.foot_positions_in_base_frame_buffer.std(axis=0)

    def print_obs_std(self):
        print(self.measurements_std_dict)


