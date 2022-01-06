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
from motion_imitation.envs import env_builder
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config


def main():
  logging.info("WARNING: this code executes low-level controller on the robot.")
  logging.info("Make sure the robot is hang on rack before proceeding.")
  input("Press enter to continue...")
  # Construct sim env and real robot
  is_sim = args.mode == "sim"
  is_hdw = args.mode == "hdw"
  if is_sim:
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

  # Task specification.
  dim_action = 12
  robot_motor_angles = robot.GetMotorAngles()
  action = robot_motor_angles
  print(action)

  # Simulation loop.
  for _ in range(10000):
    if is_sim:
        env.step(action)
    elif is_hdw:
        robot.Step(action, robot_config.MotorControlMode.POSITION)

  if is_hdw:
    robot.Terminate()


if __name__ == '__main__':
    main()
