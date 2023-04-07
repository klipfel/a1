import os, sys
import Pyro5.server
import Pyro5.api
import logging  # TODO log everything in  tmp file
import numpy
import time
import argparse
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
import numpy as np
import datetime

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
parser.add_argument("-ra", "--rack", action='store_true', help='Rack in sim.')
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

# basics constants
CONTROL_SIM_RATE = 0.001 #0.0001 # sim 0.001, latency 0.001
HDW_RESET_TIME_STEP = 0.002  # used for pybullet to reset a1 to initial joint pose
CONTROL_LATENCY_SIM = 0.00
# TODO are they their own time.sleep in their code

MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
# MAX efforts in NM
MAX_INSTANTANEOUS_TORQUE_EPS = 0.0 # tolerated different in Nm
MAX_INSTANTANEOUS_TORQUE = [
    20, 55, 55,
    20, 55, 55,
    20, 55, 55,
    20, 55, 55
]
# MAX VELOCITY in rad/s
# MAX_VELOCITY = [
#     52.4, 28.6, 28.6,
#     52.4, 28.6, 28.6,
#     52.4, 28.6, 28.6,
#     52.4, 28.6, 28.6
# ]
MAX_VELOCITY = [
    40, 28.0, 28.0,
    40, 28.0, 28.0,
    40, 28.0, 28.0,
    40, 28.0, 28.0
]
def create_tmp_data_folder():
    # Folder where the test data will be saved
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_data_folder = "/tmp" + "/" + date
    if not os.path.exists(test_data_folder):
        os.makedirs(test_data_folder)
    return test_data_folder


class RobotA1:

    def __init__(self):
        self.folder_data = create_tmp_data_folder()
        if args.hardware_mode:
            # run on hardware.
            # No environment is needed for hardware tests.
            p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # Hardware class for the robot. (wrapper)
            robot = a1_robot.A1Robot(
                pybullet_client=p,
                action_repeat=args.action_repeat,
                time_step=HDW_RESET_TIME_STEP,
                control_latency=0.0,
                reset_time=-1  # prevents issues during resetting
            )
            motor_kps = np.array([args.kp] * 12)
            motor_kds = np.array([args.kd] * 12)
            robot.SetMotorGains(motor_kps, motor_kds)
            gains = robot.GetMotorGains()
            print(f"Robot gains: ", gains)
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
            p.setTimeStep(CONTROL_SIM_RATE)
            p.setGravity(0, 0, -9.8)
            p.setPhysicsEngineParameter(enableConeFriction=1)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.floor = p.loadURDF("plane.urdf")
            self.setFloorFrictions(pybullet_client=p,
                                   lateral=1.0)
            robot = a1.A1(pybullet_client=p,
                          on_rack=args.rack,
                          action_repeat=args.action_repeat,
                          time_step=CONTROL_SIM_RATE,  # time step of the simulation
                          control_latency=CONTROL_LATENCY_SIM,
                          enable_action_interpolation=True,
                          enable_action_filter=False)
            # p.changeDynamics(robot.quadruped, -1, lateralFriction=1.0)
            # robot.SetBaseMasses([10.0])
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
        self.last_action = None
        self.control_times = []
        self.measured_control_dt = None
        self.actions = []
        self.torque_over_bound = False
        self.crazy_motor_velocity_over_bound = None
        self.ref_initial_base_orientation = None
        self.ref_initial_base_position = None
        self.pybullet_client = p
        self.robot = robot

    @Pyro5.server.expose
    def get_and_set_initial_reference_base_state(self, ini_position, ini_orn):
        '''
        Gets the initial reference base state for motion matching in simulation.
        ini_position: is a list as well as ini_orn
        :return:
        '''
        self.ref_initial_base_orientation = np.array(ini_orn)
        self.ref_initial_base_position = np.array(ini_position)
        self.set_initial_reference_base_state(self.pybullet_client,
                                              self.robot.quadruped)

    def set_initial_reference_base_state(self, pybullet_client, phys_model):
        pybullet_client.resetBasePositionAndOrientation(phys_model,
                                                        self.ref_initial_base_position,
                                                        self.ref_initial_base_orientation)

    def setFloorFrictions(self, pybullet_client, lateral=1, spinning=-1, rolling=-1):
            """Sets the frictions with the plane object

            Keyword Arguments:
                lateral {float} -- lateral friction (default: {1.0})
                spinning {float} -- spinning friction (default: {-1.0})
                rolling {float} -- rolling friction (default: {-1.0})
            """
            if self.floor is not None:
                pybullet_client.changeDynamics(self.floor, -1, lateralFriction=lateral,
                                spinningFriction=spinning, rollingFriction=rolling)

    @Pyro5.server.expose
    def get_sensor_data(self):      # exposed as 'proxy.attr' writable
        # TODO they use pybind to do a bridge between robot sensor and the wrapper.
        self.robot.ReceiveObservation()  # need to call that function anytime before reading sensor
        self.motorTorques = self.robot.GetTrueMotorTorques()
        print(f"Motor torques : {self.motorTorques}")
        # Checks if the generated torque due ot the new action is withing safety ranges.
        # self.safety_check_torque()
        self.motor_angles = self.robot.GetMotorAngles()  # in [-\pi;+\pi]
        self.motor_angle_rates = self.robot.GetMotorVelocities()
        print(f"Motor velocities : {self.motor_angle_rates}")
        self.rpy = np.array(self.robot.GetBaseRollPitchYaw())
        self.rpy_rate = self.robot.GetBaseRollPitchYawRate()
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

    def safety_check_torque(self):
        # TODO make it work on HDW, I have a broadcasting issue from shape 0 to 12 on HDW
        # Safety to check if the torques are not too high
        # self.motorTorques = self.robot.GetMotorTorques()
        abs_torque = np.abs(self.motorTorques)-MAX_INSTANTANEOUS_TORQUE_EPS
        crazy_motor_torque = abs_torque > np.array(MAX_INSTANTANEOUS_TORQUE)
        if np.any(crazy_motor_torque):
            print(f"Torque too HIGH!!!! SHUTTING DOWN!!! X(")
            print(f"Torque values for each motor: {self.motorTorques}")
            self.torque_over_bound = True
            # self.robot.Terminate()
            # sys.exit(1)
        self.torque_over_bound = False

    def safety_check_joint_velocity(self):
        '''
        Checks the expected joint velocity that is going to be produeced by the new
        joint target.
        :return:
        '''
        if len(self.control_times) > 1:
            # Computes the expected joint velocity from the new policy action
            self.measured_control_dt = self.control_times[-1] - self.control_times[-2]
            joint_velocity = np.array(self.actions[-1]) - np.array(self.actions[-2])
            joint_velocity /= self.measured_control_dt
            velocity_over_bound = (joint_velocity-MAX_VELOCITY)*(joint_velocity>0) + (joint_velocity+MAX_VELOCITY)*(joint_velocity<0)
            crazy_motor_velocity = np.abs(joint_velocity) > np.array(MAX_VELOCITY)
            self.crazy_motor_velocity_over_bound = velocity_over_bound*crazy_motor_velocity
            if np.any(crazy_motor_velocity):
                # print(f"Joint velocity too HIGH!!!! SHUTTING DOWN!!! X(")
                # print(f"Joint velocity values for each motor: {joint_velocity}")
                # print(f"Joint velocity overbound for each motor: {self.crazy_motor_velocity_over_bound}")
                return True
            return False
        return False

    def clamp_action(self, bool):
        # TODO make it work, ot does not seem to have any effect on the robot
        '''
        Clamps actions when it is estimated that the policy actions will cause a too big displacement of the
        joints.
        :param bool:
        :return:
        '''
        if bool:
            self.action = self.action - self.crazy_motor_velocity_over_bound*self.measured_control_dt

    @Pyro5.server.expose
    def get_action(self, action):      # exposed as 'proxy.attr' writable
        # TODO ADD A FLAG TO APPLY IT OR NOT
        self.control_times.append(time.time())
        if len(self.control_times) > 1:
            # Computes the expected joint velocity from the new policy action
            self.measured_control_dt = self.control_times[-1] - self.control_times[-2]
            print(f"Control time : {self.measured_control_dt}")
        self.action = np.array(action)
        self.actions.append(action)
        self.last_action = self.action
        # Safety check
        # jv_issue = self.safety_check_joint_velocity()
        # self.clamp_action(jv_issue)
        # print(self.action)
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


