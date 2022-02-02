import Pyro5.api
import logging
import numpy as np
import time

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting inference client.")
    # The object remotely accessed is different everytime, the server uses a new object.
    # for _ in range(4):
    #     policy = Pyro5.api.Proxy("PYRONAME:laptop.inference")
    #     action = policy.inference(1)
    #     action = np.array(action, dtype=np.float32)
    #     print(action)
    #     print(policy.getpid())
    #     time.sleep(1)
    # Now we can just use a single object from a server.
    policy = Pyro5.api.Proxy("PYRONAME:laptop.inference")
    policy._pyroSerializer = "marshal"  # faster communication.
    policy._pyroTimeout = 1.5    # 1.5 seconds
    while True:
        action = policy.inference(1)
        action = np.array(action, dtype=np.float32)
        print(action)
        print(policy.getpid())
        time.sleep(1)
