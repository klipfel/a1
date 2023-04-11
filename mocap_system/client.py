from __future__ import print_function
# from vicon_dssdk import ViconDataStream
import argparse
import pickle
import zlib
from typing import Any, Dict, cast
import numpy as np
import zmq
import time

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
        return A.reshape(md['shape'])

class SerializingContext(zmq.Context[SerializingSocket]):
    _socket_class = SerializingSocket


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('host', nargs='?', help="Host name, in the format of server:port", default = "localhost:801")
args = parser.parse_args()

# client = ViconDataStream.Client()
try:
    ctx = SerializingContext()
    rep = ctx.socket(zmq.SUB)  # rep is short for "reply" (server side)
    rep.setsockopt(zmq.BACKLOG, 2) # add by JH for last-msg onnly(https://stackoverflow.com/questions/38256936/last-message-only-option-in-zmq-subscribe-socket)
    rep.connect('tcp://192.168.1.103:9999')
    rep.subscribe("")
    print(rep,"waiting")
    time_start = time.time()
    time_tmp = 0
    while(True):
        array = rep.recv_array()
        # print("recv_array")
        if (time.time() - time_start)//1 != time_tmp:
            print(array)
            time_tmp = (time.time() - time_start)//1
        # time.sleep(1/30)
        
        # rep.connect('tcp://*:9999')
except:
    __import__('pdb').set_trace()