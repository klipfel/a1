# saved as greeting-client.py
import Pyro5.api
import numpy as np
import time
import argparse
import os
# Pybullet.
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

# TODO not complete, I actually don't need this code.

from tqdm import tqdm

# Robot library for hardware.
os.sys.path.append("/home/unitree/arnaud/motion_imitation")

from motion_imitation.robots import a1_robot  # imports the robot interface in the case where the code is

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--uri", help="URI of the proxy of the Policy object", type=str, default=None)
args = parser.parse_args()

# Server connection ini.
policy = Pyro5.api.Proxy(args.uri)     # get a Pyro proxy to the greeting object
policy._pyroBind()
policy._pyroSerializer = "marshal"  # faster communication.
policy._pyroTimeout = 1.5    # 1.5 seconds

# Robot setup.
# run on hardware.
# No environment is needed for hardware tests.
p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Hardware class for the robot. (wrapper)
robot = a1_robot.A1Robot(pybullet_client=p,
                         action_repeat=1,
                         time_step=0.0001,
                         control_latency=0.0)
robot.motor_kps = np.array([100.0] * 12)
robot.motor_kds = np.array([2.0] * 12)
print("Robot Kps: ", robot.motor_kps)
print("Robot Kds: ", robot.motor_kds)
robot.ReceiveObservation()

while True:
    t0 = time.time()
    action = policy.inference(obs.tolist())
    # action = np.array(action, dtype=np.float32)
    # print(action)
    delta = time.time() - t0
    print(f"Time of inference: {delta}")
    print(obs)
