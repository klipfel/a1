import os, sys
import Pyro5.server
import Pyro5.api
import logging
import numpy
import time
import argparse
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
import numpy as np

# TODO explore Pyro callbacks

'''
Single file to run on the robot.
- This file has is the server running on the robot.
- The function is :
    1. Read the robot sensors.
    2. Send the sensor reading to the laptop for inference.
    3. Apply the command to the HDW.
'''
parser = argparse.ArgumentParser()
parser.add_argument("--kp", help='Proportional gain.', type=float, default=50.0)
parser.add_argument("--kd", help='Derivative gain.', type=float, default=2.0)
parser.add_argument("--ip_host", help='Host ip address, where the Pyro deamon will be called and where the name server'
                                 'should be instanciated, it should an IP address on the host.', type=str, default="192.168.123.12")
parser.add_argument("--ns_ip_host", help='Host ip address of the Name Server in Pyro', type=str, default="192.168.123.24")
parser.add_argument("-v", "--visualize", action='store_true', help='Activates the rendering in sim mode when present.')
parser.add_argument("-nc", "--no_control", action='store_true', help='If flag is present the control command is not'
                                                                     'not sent to the Low-Level DC motors.')
parser.add_argument("-hdw", "--hardware_mode", action='store_true', help='Hardware mode for control on the robot.')
parser.add_argument("-arp", "--action_repeat", help="Repeats the action applied on hardware.", type=int, default=1)
args = parser.parse_args()

# I had gto displace these statement in the preamble as it did not like it if you imported packages inside methdos.
if args.hardware_mode:
    # imports the robot interface in the case where the code is
    os.sys.path.append("/media/f89b4767-18eb-4289-8b1c-1981706279e6/a1/motion_imitation")
    from motion_imitation.robots import a1_robot, robot_config
else:
    os.sys.path.append("/media/arnaud/arnaud/a1/motion_imitation")
    from motion_imitation.robots import a1, robot_config  # for sim


class RobotA1:

    def __init__(self):
        if args.hardware_mode:
            # run on hardware.
            # No environment is needed for hardware tests.
            p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # Hardware class for the robot. (wrapper)
            robot = a1_robot.A1Robot(pybullet_client=p,
                                     action_repeat=args.action_repeat,
                                     time_step=0.001,
                                     control_latency=0.0)
            motor_kps = np.array([args.kp] * 12)
            motor_kds = np.array([args.kd] * 12)
            robot.SetMotorGains(motor_kps, motor_kds)
            gains = robot.GetMotorGains()
            print("Robot Kps: ", robot.motor_kps)
            print("Robot Kds: ", robot.motor_kds)
        else:
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
            simulation_time_step = 0.001
            p.setTimeStep(simulation_time_step)
            p.setGravity(0, 0, -9.8)
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf")
            robot = a1.A1(pybullet_client=p,
                          on_rack=False,
                          action_repeat=args.action_repeat,
                          time_step=simulation_time_step,  # time step of the simulation
                          control_latency=0.0,
                          enable_action_interpolation=True,
                          enable_action_filter=False)
            motor_kps = np.array([args.kp] * 12)
            motor_kds = np.array([args.kd] * 12)
            robot.SetMotorGains(motor_kps, motor_kds)
            gains = robot.GetMotorGains()
            print("Robot Kps:", gains[0])
            print("Robot Kds:", gains[1])
            if args.visualize:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        # Class attribute
        self.robot = robot
        self.args = args
        self.pybullet_client = p
        # sensor data dict
        self.sensor_data = {}
        self.action = None

    @Pyro5.server.expose
    def get_sensor_data(self):      # exposed as 'proxy.attr' writable
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
        return self.robot_data.tolist()

    @Pyro5.server.expose
    def get_action(self, action):      # exposed as 'proxy.attr' writable
        # TODO ADD A FLAG TO APPLY IT OR NOT
        self.action = np.array(action)
        if not self.args.no_control:
            self.apply_action()
        else:
            print(f"NO CONTROL MODE.")

    def apply_action(self):
        self.robot.Step(self.action.flatten(), robot_config.MotorControlMode.POSITION)

    def getpid(self):
        return self.pid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting server on the robot.")
    p_server = RobotA1()
    # If testing on a single machine use the loopback ip address 127.0.0.1
    daemon = Pyro5.api.Daemon(host=f"{args.ip_host}", port=2020)             # make a Pyro daemon
    ns = Pyro5.api.locate_ns(host=args.ns_ip_host)             # find the name server
    # TODO not sure about how to register or what to register, will I have to different objects? Maybe it is better to
    # TODO create a wrapper of the policy class in the control utilities.
    uri = daemon.register(p_server)    # register the greeting maker as a Pyro object
    ns.register("robot.server", uri)   # register the object with a name in the name server
    print("Ready. Object uri =", uri)       # print the uri so we can use it in the client later
    print("WAITING FOR CALLS FROM THE REMOTE LAPTOP .... ")       # print the uri so we can use it in the client later
    daemon.requestLoop()                    # start the event loop of the server to wait for calls


