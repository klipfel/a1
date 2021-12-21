"""Apply the same action to the simulated and real A1 robot.


As a basic debug tool, this script allows you to execute the same action
(which you choose from the pybullet GUI) on the simulation and real robot
simultaneouly. Make sure to put the real robbot on rack before testing.

TODO: if you want to be able to to simulate A1 you will need to create a pybullet environment.

"""

from absl import app
from absl import logging
import numpy as np
import time
from tqdm import tqdm
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

from motion_imitation.robots import a1
from motion_imitation.robots import robot_config

import argparse

from gym import error, spaces

# Constants
CONTROL_TIME_STEP = 0.025  # TODO: how is pybullet working?

positions = np.loadtxt("joint_targets.txt")

velocities = np.zeros(positions.shape)

# Computes target joint velocities from joint positions.
for i in range(1,positions.shape[0]):
  velocities[i,:] = (positions[i,:] - positions[i-1,:])/0.008

obs_space = spaces.Box(-1*np.ones(33,),np.ones(33,))
ac_space = spaces.Box(-1*np.ones(12,),np.ones(12))

def main(args):
  logging.info(
      "WARNING: this code executes low-level controller on the robot.")
  logging.info("Make sure the robot is hang on rack before proceeding.")
  input("Press enter to continue...")
  test_type = args.test_type
  # Construct sim env and real robot
  if args.visualize:
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
  else:
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  p.setGravity(0, 0, -10)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  robot = a1.A1(pybullet_client=p, action_repeat=1, on_rack=False)  # simulation class for A1.
  # TODO: add a flag to choose from hardware or simulation.

  # Move the motors slowly to initial position
  robot.ReceiveObservation()
  current_motor_angle = np.array(robot.GetMotorAngles())
  print(current_motor_angle)
  input("Press enter to continue...")
  desired_motor_angle = positions[0,:]
  action = desired_motor_angle
  i = 1

  done = False
  initialize = False
  q = []
  rpy = []
  drpy = []
  dq = []
  fc = []
  acc = []
  timer = []

  n_steps = 2000  # number of simulation steps.

  if test_type=="static":
    for t in range(n_steps):
      print("Simulation step: {}", format(t))
      qt = np.array(robot.GetMotorAngles())
      e = np.linalg.norm(qt-action)
      print("Measured joint positions: {}", qt)
      print("Control error: {}", e)
      action = desired_motor_angle
      robot.Step(action, robot_config.MotorControlMode.POSITION)
      #time.sleep(0.005)  # TODO: find the right simulation control frequency.

  elif test_type=="single_jp_target":
    desired_motor_angle = np.array([0., 0.5, -1.8,  0., 0.5, -1.8,  0., 0.5, -1.8,  0., 0.5, -1.8])
    action = desired_motor_angle
    for t in range(n_steps):
      print("Simulation step: {}", format(t))
      qt = np.array(robot.GetMotorAngles())
      e = np.linalg.norm(qt-action)
      print("Measured joint positions: {}", qt)
      print("Control error: {}", e)
      robot.Step(action, robot_config.MotorControlMode.POSITION)
      time.sleep(CONTROL_TIME_STEP)  # TODO: find the right simulation control frequency.

  else:
    logging.error("ERROR: test configuration does not exist.")

  # while not done:
  #   if initialize == False:
  #     for t in tqdm(range(2000)):
  #       blend_ratio = np.minimum(t / 200., 1)
  #       action = (1 - blend_ratio
  #             ) * current_motor_angle + blend_ratio * desired_motor_angle
  #       robot.Step(action, robot_config.MotorControlMode.POSITION)
  #       time.sleep(0.005)
  #     initialize = True
  #   input("enter")
  #   robot.ReceiveObservation()
  #
  #   begin = time.time()
  #   #q = robot.GetMotorAngles()
  #   #o = np.concatenate((robot.GetBaseRollPitchYaw(),robot.GetBaseRollPitchYawRate(),robot.GetMotorAngles(),robot.GetMotorVelocities(),np.array([phi,left,right])))
  #   #print(o.shape)
  #
  #   #action = q + a*0.05
  #
  #   q.append(robot.GetMotorAngles())
  #   #rpy.append(robot.GetBaseRollPitchYaw())
  #   drpy.append(robot.GetTrueBaseRollPitchYawRate())
  #   #rpy.append(robot.GetBaseOrientation())
  #   rpy.append(robot._raw_state.imu.rpy)
  #   dq.append(robot.GetMotorVelocities())
  #   fc.append(robot.GetFootContacts())
  #   timer.append(time.time())
  #   acc.append(robot._raw_state.imu.accelerometer)
  #   #print(robot._raw_state.imu.accelerometer)
  #   #print("orientation",robot._raw_state.imu.rpy)
  #
  #   action =  desired_motor_angle
  #   robot.Step(action, robot_config.MotorControlMode.POSITION)
  #
  #
  #   time.sleep(0.005)
  #
  #   i+=1

  robot.Terminate()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--visualize", help='visualization boolean.', type=bool, default=False)
  parser.add_argument("-t", "--test_type", help='Type of the test: static.', type=str, default="static")
  args = parser.parse_args()
  main(args)
