from __future__ import print_function
# from vicon_dssdk import ViconDataStream
import argparse
import sys
import multiprocessing
import pickle
import zlib
from typing import Any, Dict, cast
import numpy as np
import zmq
import Pyro5.server
import Pyro5.api
import time
from multiprocessing import Process as Process
import threading
import argparse
import copy
import pandas as pd
import datetime
import numpy as np
from numpy.linalg import norm

parser = argparse.ArgumentParser()
parser.add_argument("--read_only", help='Reads the mocap data only and does not publish it.', action="store_true")
parser.add_argument("--save_data", help='Saves the data.', action="store_true")
parser.add_argument("--ip_mocap_system", help='Mocap system ip where the data is published.',
                    type=str,
                    # default='tcp://192.168.1.103:9999',  # the one used in case of connection through the local network
                    default='tcp://128.61.117.245:6666' # the one using the eduroam network
                    )
                    # when connected on eduraom 128.61.117.245 is the ip adress of the mocap system
args = parser.parse_args()

# Global variables
DATA_BUFFER = []  # global va for the mocap data shared by the processes


# PyZMQ class to send information
class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods
    send_zipped_pickle is just like send_pyobj, but uses
    zlib to compress the stream before sending.
    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    """
 
    def send_zipped_pickle(
        self, obj: Any, flags: int = 0, protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        print('zipped pickle is %i bytes' % len(zobj))
        return self.send(zobj, flags=flags)
 
    def recv_zipped_pickle(self, flags: int = 0) -> Any:
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = self.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)
 
    def send_array(
        self, A: np.ndarray, flags: int = 0, copy: bool = True, track: bool = False
    ) -> Any:
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)
 
    def recv_array(
        self, flags: int = 0, copy: bool = True, track: bool = False
    ) -> np.ndarray:
        """recv a numpy array"""
        md = cast(Dict[str, Any], self.recv_json(flags=flags))
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])
        # print(md['dtype'])
        return A.reshape(md['shape'])
 
class SerializingContext(zmq.Context[SerializingSocket]):
    _socket_class = SerializingSocket
 
# Transform position from body to world frame
def pos_in_world_frame(
    global_rotation_matrix: np.ndarray,
    global_translation: np.ndarray,
    pos_b: np.ndarray = np.array([0., 0., 315.])
    )-> np.ndarray:
    pos_w = global_translation + global_rotation_matrix.T @ pos_b
    return pos_w
 
# Compute angular velocity given orientation in quaternion
def compute_ang_vel(
    q_curr: np.ndarray,
    q_prev: np.ndarray,
    dt: float
    ) -> np.ndarray:
    # Quaternions in Vicon are of the form (x, y, z, w) where w is the
    # real component. This convention is the same as that of PyBullet.
    # Need to transform this to (w, x, y, z) only for these calculations.
    q_dot = (q_curr - q_prev) / dt  # TODO fix the quaternion difference
    E = np.array([[-q_curr[1], q_curr[0], -q_curr[3], q_curr[2]],
                  [-q_curr[2], q_curr[3], q_curr[0], -q_curr[1]],
                  [-q_curr[3], -q_curr[2], q_curr[1], q_curr[0]]])
    return 2 * E @ q_dot
 
def angular_velocity(q_t, q_t_1, dt):
    """
    Compute the angular velocity between two quaternions at time t and t-1.
    
    :param q_t: Quaternion at time t
    :param q_t_1: Quaternion at time t-1
    :param dt: Time difference between t and t-1
    :return: Angular velocity in radians per second
    """
    # Compute the difference between the quaternions
    q_diff = np.multiply(q_t_1, np.conj(q_t))

    # Extract the axis and angle of rotation from the quaternion difference
    angle = 2.0 * np.arccos(q_diff[0])
    axis = q_diff[1:] / np.sin(angle / 2.0)

    # Compute the angular velocity
    angular_velocity = axis * angle / dt

    return angular_velocity

def linear_velocity(p_t, p_t_1, dt):
    """
    Compute the linear velocity between two positions at time t and t-1.
    
    :param p_t: Position at time t
    :param p_t_1: Position at time t-1
    :param dt: Time difference between t and t-1
    :return: Linear velocity in meters per second
    """
    return (p_t - p_t_1) / dt


# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument('host', nargs='?', help="Host name, in the format of server:port", default = "localhost:801")
# args = parser.parse_args()
 
# client = ViconDataStream.Client()
# i = 0


class MocapSystemClient:
    """
    Client connected and reading data from the mocap system hardware.
    """

    def __init__(self):
        # Creates the context for the pyzmq interface with the mocap system
        self.mocap_data = None
        ctx = SerializingContext()
        self.rep = ctx.socket(zmq.SUB)  # rep is short for "reply" (server side)
        self.rep.subscribe(b"")
        #self.rep.setsockopt(zmq.SUBSCRIBE, b'')
        ip_publisher = args.ip_mocap_system  # ip of the mocaps system, need to be on the same network.
        # here : TP-link  is used
        self.rep.connect(ip_publisher)
        print(f"Connecting to publisher {ip_publisher}")
        # some data buffering
        self.last_time = None
    
    def get_data(self):
        print("Receiving data ....")
        data = self.rep.recv_array()
        print(data)
        self.mocap_data = copy.deepcopy(data[0])
        self.last_time = data[0][-2]
        self.mocap_data = np.concatenate((self.mocap_data, self.estimate_velocities(self.mocap_data)))

    def estimate_velocities(self, data):
        data = np.array(data)
        lin_vel = np.zeros((3,))
        angl_vel = np.zeros((3,))
        if len(DATA_BUFFER) >= 1:
            data_t_1 = np.array(DATA_BUFFER[-1])
            dt =  (self.last_time - data_t_1[3+9+3+4])
            lin_vel = linear_velocity(data[:3], data_t_1[:3], dt)
            angl_vel =   angular_velocity(data[3+9+3:3+9+3+4], data_t_1[3+9+3:3+9+3+4], dt)
        return np.concatenate((lin_vel, angl_vel))

    def store(self):
        # gets the data
        DATA_BUFFER.append(self.mocap_data)

    def loop(self):
        while True:
            self.get_data()
            self.store()


class MocapServer:
    """
    Server that publishes the mocap data for the policy to read.
    """
    def __init__(self):
        self.daemon = Pyro5.api.Daemon(host=f"0.0.0.0", port=6235)
        ns = Pyro5.api.locate_ns()
        uri = self.daemon.register(self)    # register the greeting maker as a Pyro object
        ns.register("mocap_system.server", uri)   # register the object with a name in the name server
        print("Ready. Object uri =", uri)       # print the uri so we can use it in the client later

    def start(self):
        self.daemon.requestLoop()

    @Pyro5.server.expose
    def get_data(self):      # exposed as 'proxy.attr' writable
        data = copy.deepcopy(DATA_BUFFER[-1])  # gets the last data received from the mocap system
        # and take the data in the first row list of list ...
        print(f"Mocap server, sending data to policy: {data}")
        return data.tolist()


if __name__ == '__main__':
    # Mocap server instantiation
    mocap_server = MocapServer()
    # Mocap system instantiation
    mocap_system = MocapSystemClient()
    # Different processes
    print("Processes creating")
    thread_server = threading.Thread(target=mocap_server.start, #daemon=True
                                     )
    thread_mocap_system = threading.Thread(target=mocap_system.loop, #daemon=True
                                           )
    print("Processes created")
    try:
        # Acquires the data from the mocap system
        if args.read_only:
            # Just reads from the data stream of the mocap system.
            # The mocap system loop works fine but not when I use it as a process.
            mocap_system.loop()
        else:
            thread_mocap_system.start()
            thread_server.start()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    finally:
        print("Closing the program and saving data.")
        # stops the threads
        if not(args.read_only):
            # TODO fix this part, the threads are not stopping correclty
            #thread_mocap_system.join()
            #thread_server.join()
            sys.exit()
        # uses pandas to save the data in a folder in /tmp w
        df = pd.DataFrame(DATA_BUFFER)
        # adds columns names
        df.columns = ["x", "y", "z",
                    "R11", "R12", "R13",
                    "R21", "R22", "R23",
                    "R31", "R32", "R33",
                    # euler angles
                    "roll", "pitch", "yaw",
                    "qw", "qx", "qy", "qz",
                    "time",
                    "occlusion",
                    # linear velocity,
                    "lin_vel_x", "lin_vel_y", "lin_vel_z",
                    "ang_vel_x", "ang_vel_y", "ang_vel_z",
                    ]
        df.to_csv(f"/tmp/mocap_data_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv")
        print("Data saved")
