import Pyro5.server
import Pyro5.api
import logging
import numpy
import time

# TODO log everything to a file using logging.


class Policy:

    def __init__(self):
        self.joint_position_target = [1]*12
        self.pid = numpy.random.randint(0,10)

    @Pyro5.server.expose
    def inference(self, obs):      # exposed as 'proxy.attr' writable
        numpy_array = numpy.array([1,2,12.6456110,54.4545154,2.5645154,5,2,5.5455451,2,5,25,4,2,54,2,5,5,56,4], dtype=numpy.float32)
        self.joint_position_target = numpy_array.tolist()
        return self.joint_position_target

    def getpid(self):
        return self.pid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting inference server.")
    p_server = Policy()
    daemon = Pyro5.api.Daemon(host="192.168.123.24", port=2020)             # make a Pyro daemon
    ns = Pyro5.api.locate_ns()             # find the name server
    # TODO not sure about how to register or what to register, will I have to different objects? Maybe it is better to
    # TODO create a wrapper of the policy class in the control utilities.
    uri = daemon.register(Policy)    # register the greeting maker as a Pyro object
    ns.register("laptop.inference", uri)   # register the object with a name in the name server
    print("Ready. Object uri =", uri)       # print the uri so we can use it in the client later
    daemon.requestLoop()                    # start the event loop of the server to wait for calls


