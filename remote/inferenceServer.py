import Pyro5.server
import Pyro5.api
import logging
import numpy
import time
import os
import torch
from torch.distributions import Normal
# Adding root folder to pythonpath to find other packages
os.sys.path.append("/home/arnaud/morphology_agnostic")
from utilities.config import Config
import argparse
from util import recover_data, adapt_data_for_comm


# TODO log everything to a file using logging.

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="pre-trained weight path", type=str, default=Config.WEIGHT_PATH)
parser.add_argument("-m", "--mode", help="inference on cpu or cuda", type=str, default="cpu")
parser.add_argument("--host", help="Host ip address.", type=str, default="192.168.123.24")
parser.add_argument("-p", "--port", help="port for Pyro deamon.", type=int, default=2020)
args = parser.parse_args()


class Policy:

    def __init__(self, stochastic_test=False):
        self.stochastic_test = stochastic_test
        self.weight_path = args.weight
        from policy import ppo_module  # net architectures.
        # Inference done on the CPU.
        # TODO compare with GPU? in time
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\nTorch device: ", self.device)
        # calculate i/o dimensions of the policy net.
        self.ob_dim = Config.INPUT_DIMS
        self.act_dim = Config.OUTPUT_DIMS
        # Load policy net.
        self.loaded_graph = ppo_module.HafnerActorModelStd(self.ob_dim, self.act_dim)
        self.loaded_graph.load_state_dict(torch.load(self.weight_path, map_location=self.device)["actor_architecture_state_dict"])

    @Pyro5.server.expose
    def inference(self, obs):
        # Inference mode context manager to remove grad computation, similar to no_grad.
        # No need of the gradient for inference.
        obs = recover_data(obs)
        with torch.inference_mode():
            # TODO why do I need to do .cpu? you can use .cudo or .to('cuda') or to('cpu')
            action_ll = self.loaded_graph.forward(torch.from_numpy(obs).cpu())
            mean = action_ll[:, self.act_dim//2:]
            action_np = mean.cpu().numpy()
        # TODO adapt numpy array to list and 1D, do I have as many row as env I trained on?
        return adapt_data_for_comm(action_np)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting inference server.")
    p_server = Policy()
    daemon = Pyro5.api.Daemon(host=args.host, port=args.port)             # make a Pyro daemon
    ns = Pyro5.api.locate_ns()             # find the name server
    uri = daemon.register(Policy)    # register the greeting maker as a Pyro object
    ns.register("laptop.inference", uri)   # register the object with a name in the name server
    print(f"Ready. Object uri:\n{uri}")       # print the uri so we can use it in the client later
    daemon.requestLoop()                    # start the event loop of the server to wait for calls
