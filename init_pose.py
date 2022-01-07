"""

Convergence to an initial/single joint position.

"""

import os
import inspect
import argparse
from tqdm import tqdm
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--visualize", help='visualization boolean.', type=bool, default=True)
# TODO rack in hardware mode.
parser.add_argument("-r", "--rack", help='rack boolean. If true the robot is considered to be on a rack. For now only in simulation', type=bool, default=True)
parser.add_argument("-t", "--test_type", help='Type of the test: static.', type=str, default="static")
parser.add_argument("-m", "--mode", help='sim or hdw', type=str, default="sim")
# TODO why do the gains not have any effect in simulation?
parser.add_argument("--kp", help='Proportional for thigh and calf.', type=float, default=100.0)
parser.add_argument("--kpa", help='Proportional for hip.', type=float, default=100.0)
parser.add_argument("--kd", help='Derivative for thigh and calf.', type=float, default=1.0)
parser.add_argument("--kda", help='Derivative for hip.', type=float, default=1.0)
parser.add_argument("--dt", help="Control time step.", type=float, default=0.01)
parser.add_argument("--nsteps", help="Total control steps to reach joint position.", type=int, default=300)
args = parser.parse_args()


#from absl import app  # conflict with argpase # TODO fix it
from absl import logging
import numpy as np

# Pybullet.
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

# Motion imitation wrapper.
if args.mode == "hdw":  # adds the path to the local motion_imitation wrapper installation.
    os.sys.path.append("/home/unitree/arnaud/motion_imitation")
from motion_imitation.robots import robot_config


def main():
    logging.info("WARNING: this code executes low-level controller on the robot.")
    logging.info("Make sure the robot is hang on rack before proceeding.")
    input("Press enter to continue...")
    # Construct sim env and real robot
    is_sim_env = args.mode == "simEnv"
    is_sim_gui = args.mode == "simGui"
    is_hdw = args.mode == "hdw"
    # Gains.
    KP = args.kp
    KD = args.kd
    KPA = args.kpa
    KDA = args.kda
    # Creates a simulation using a gym environment.
    if is_sim_env:
        from motion_imitation.robots import a1
        from motion_imitation.envs import env_builder # moved it here since it also imports tensforflow. Don't need this on
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
    # HDW.
    elif is_hdw:
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
    current_motor_angle = np.array(robot.GetMotorAngles())
    print("Current joint positions:", current_motor_angle)
    desired_motor_angle = np.array([0., 1.0, -1.8] * 4)
    print("Desired initial joint positions:", desired_motor_angle)

    if is_hdw:
        # Static control to the same joint position the robot has. This is a test.
        print("TEST: Keeps current configuration for 1 sec.")
        for _ in tqdm(range(100)):
            robot.Step(current_motor_angle, robot_config.MotorControlMode.POSITION)
            time.sleep(0.01)

    print("Control starts...")
    # input("Proceed to control? Press a key.")
    for t in tqdm(range(args.nsteps)):
        blend_ratio = np.minimum(t / 200., 1)
        action = (1 - blend_ratio
                  ) * current_motor_angle + blend_ratio * desired_motor_angle
        if is_sim_env:
            env.step(action)
        elif is_hdw or is_sim_gui:
            robot.Step(action, robot_config.MotorControlMode.POSITION)
        else:
            logging.error("ERROR: unsupported mode. Either sim or hdw.")
        time.sleep(args.dt)  # the example used 0.005.

    print("Final joint positions:", np.array(robot.GetMotorAngles()))
    print("Final joint positions error:", np.linalg.norm(np.array(robot.GetMotorAngles())-desired_motor_angle))

    if is_hdw:
        robot.Terminate()


if __name__ == '__main__':
    main()
