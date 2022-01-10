import numpy as np
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
        # TODO why do the gains not have any effect in simulation?
        parser.add_argument("--kp", help='Proportional for thigh and calf.', type=float, default=100.0)
        parser.add_argument("--kpa", help='Proportional for hip.', type=float, default=100.0)
        parser.add_argument("--kd", help='Derivative for thigh and calf.', type=float, default=0.5)
        parser.add_argument("--kda", help='Derivative for hip.', type=float, default=0.5)
        parser.add_argument("--dt", help="Control time step.", type=float, default=0.01)
        parser.add_argument("--nsteps", help="Total control steps to reach joint position.", type=int, default=300)
        parser.add_argument("--sp", help="Smoothing percentage.", type=float, default=2/3)
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
            robot.motor_kps = np.array([KPA,KP,KP] * 4)
            robot.motor_kds = np.array([KDA,KD,KD] * 4)
            print("Robot Kps: ", robot.motor_kps)
            print("Robot Kds: ", robot.motor_kds)
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
