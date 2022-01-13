"""

Convergence to an initial/single joint position.

"""

import os
import inspect
from tqdm import tqdm
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import logging
import numpy as np

from utilities.control_util import ControlFramework
from utilities.config import Config

cf = ControlFramework()

# Motion imitation wrapper.
if cf.args.mode == "hdw":  # adds the path to the local motion_imitation wrapper installation.
    os.sys.path.append("/home/unitree/arnaud/motion_imitation")
from motion_imitation.robots import robot_config


def main():
    # Initial setup, prepares the robot for policy.
    cf.go_to_initial_configuration()
    # Static control to the same joint position the robot has. This is a test.
    print("TEST: Keeps current configuration for 1 sec.")
    # current_motor_angle = np.array(cf.robot.GetMotorAngles())
    for _ in tqdm(range(1000)):
        obs = cf.obs_parser.observe()
        # cf.obs_parser.print_obs()
        # action = cf.policy.inference(obs)
        # print(action)
        cf.robot.Step(Config.INI_JOINT_CONFIG, robot_config.MotorControlMode.POSITION)
        time.sleep(0.01)
    # Runs policy.
    print(cf.args)
    if cf.args.run_hdw:
        cf.run()

    # cf.obs_parser.get_obs_std()
    # cf.obs_parser.print_obs_std()

    # print("Control starts...")
    # alpha = 0.2
    # for t in tqdm(range(cf.args.nsteps)):
    #     current_motor_angle = np.array(cf.robot.GetMotorAngles())
    #     action = current_motor_angle*alpha + (1-alpha)*desired_motor_angle
    #     if cf.is_sim_env:
    #         cf.env.step(action)
    #     elif cf.is_hdw or cf.is_sim_gui:
    #         cf.robot.Step(action, robot_config.MotorControlMode.POSITION)
    #     else:
    #         logging.error("ERROR: unsupported mode. Either sim or hdw.")
    #     time.sleep(cf.args.dt)  # the example used 0.005.

    print("Final joint positions:", np.array(cf.robot.GetMotorAngles()))
    # print("Final joint positions error:", np.linalg.norm(np.array(cf.robot.GetMotorAngles())-desired_motor_angle))

    cf.logger.log()

    if cf.is_hdw:
        cf.robot.Terminate()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        cf.logger.log()

