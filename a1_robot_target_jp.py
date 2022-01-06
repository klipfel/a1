"""Apply the same action to the simulated and real A1 robot.


As a basic debug tool, this script allows you to execute the same action
(which you choose from the pybullet GUI) on the simulation and real robot
simultaneouly. Make sure to put the real robot on rack before testing.

This script was modified from the original. Here it enables the user to choose the
joint position targets with a pybullet GUI, but it does not work on the hardware unless
you are connected with a screen.

TODO how are they able to execute the actions on hardware and on simulation simultaneously? Better to keep them separate.
TODO squatting, body lowering, body rotation, lift one foot.
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
  is_sim = args.mode == "sim"
  is_hdw = args.mode == "hdw"
  if is_sim:
      from motion_imitation.robots import a1
      from motion_imitation.envs import env_builder # moved it here since it also imports tensforflow. Don't need this on
      # the hardware.
      # Create an environment for simulation.
      env = env_builder.build_regular_env(
          robot_class=a1.A1,  # robot class for simulation
          motor_control_mode=robot_config.MotorControlMode.POSITION,
          on_rack=args.rack,
          enable_rendering=args.visualize,
          wrap_trajectory_generator=False)
      robot = env.robot
  elif is_hdw:
      from motion_imitation.robots import a1_robot  # imports the robot interface in the case where the code is
      # run on hardware.
      # No environment is needed for hardware tests.
      p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
      p.setAdditionalSearchPath(pybullet_data.getDataPath())
      # Hardware class for the robot. (wrapper)
      robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)
  else:
      logging.error("ERROR: unsupported mode. Either sim or hdw.")

  robot.ReceiveObservation()

  initialize = False
  current_motor_angle = np.array(robot.GetMotorAngles())
  desired_motor_angle = np.array([-0.17, 0.75, -1.34,0.21,0.95,-1.44,-0.21,0.73,-1.57,0.17,0.50,-1.33])
  print("Target joint position: ", desired_motor_angle)
  input("Press enter to continue...")

  # Simulation loop.
  for _ in tqdm(range(10)):
    # Initialization to converge smoothly to first joint position target.
    if initialize == False:
      for t in tqdm(range(2000)):
        blend_ratio = np.minimum(t / 200., 1)
        action = (1 - blend_ratio
              ) * current_motor_angle + blend_ratio * desired_motor_angle
        if is_sim:
            env.step(action)
        elif is_hdw:
            robot.Step(action, robot_config.MotorControlMode.POSITION)
        time.sleep(0.005)
      initialize = True

    # Control to other jp.
    action =  desired_motor_angle
    if is_sim:
        env.step(action)
    elif is_hdw:
        robot.Step(action, robot_config.MotorControlMode.POSITION)


    time.sleep(0.005)

  if is_hdw:
    robot.Terminate()


if __name__ == '__main__':
    main()
