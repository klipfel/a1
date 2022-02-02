# saved as greeting-client.py
import Pyro5.api
import numpy as np
import time

uri = input("What is the Pyro uri of the greeting object? ").strip()
name = input("What is your name? ").strip()

policy = Pyro5.api.Proxy(uri)     # get a Pyro proxy to the greeting object
policy._pyroBind()
policy._pyroSerializer = "marshal"  # faster communication.
policy._pyroTimeout = 1.5    # 1.5 seconds
obs = np.random.rand(60)
print(f"obs: {obs}")
while True:
    t0 = time.time()
    action = policy.inference(obs)
    # action = np.array(action, dtype=np.float32)
    # print(action)
    delta = time.time() - t0
    print(f"Time of inference: {delta}")
