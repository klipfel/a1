from __future__ import print_function
# from vicon_dssdk import ViconDataStream
import argparse
import pickle
import zlib
from typing import Any, Dict, cast
import numpy as np
import zmq
import pybullet as p
import time
from multiprocessing import Process as Process
import threading

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
    q_curr = q_curr[[3, 0, 1, 2]]
    q_prev = q_prev[[3, 0, 1, 2]]
    q_dot = (q_curr - q_prev) / dt
    E = np.array([[-q_curr[1], q_curr[0], -q_curr[3], q_curr[2]],
                  [-q_curr[2], q_curr[3], q_curr[0], -q_curr[1]],
                  [-q_curr[3], -q_curr[2], q_curr[1], q_curr[0]]])
    return 2 * E @ q_dot
 
# parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument('host', nargs='?', help="Host name, in the format of server:port", default = "localhost:801")
# args = parser.parse_args()
 
# client = ViconDataStream.Client()
# i = 0


class Mocap:

    def __init__(self):
        ctx = SerializingContext()
        self.rep = ctx.socket(zmq.SUB)  # rep is short for "reply" (server side)
        self.rep.subscribe(b"")
        #self.rep.setsockopt(zmq.SUBSCRIBE, b'')
        ip_publisher = 'tcp://192.168.1.103:9999'
        self.rep.connect(ip_publisher)
        print(f"Connecting to publisher {ip_publisher}")
    
    def get_data(self):
        print("Receiving data ....")
        data = self.rep.recv_array(copy=False).copy()
        #data = self.rep.recv()
        print(data)
        # if not np.isnan(obs_).any():
        #     self.obs = obs_.copy()
        # else:
        #     time.sleep(0.001)


if __name__ == '__main__':
    mocap = Mocap()
    while True:
        mocap.get_data()
