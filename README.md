Repository to transfer policy on real A1 robot.

# Installation
## Laptop packages
Packages contained in `prerequisites.txt`.
Recommended to use conda env.
```bash
pip install prerequisites.txt
```
# Launch code on the robot
Usage:
To use Ethernet connection:
1. Launch remote server on the laptop.
- Launch the Pyro name server on the laptop.
```bash
python -m Pyro5.nameserver -n 192.168.123.24
```
- Launch the Inference Serve on the laptop for remote inference. Located in the `remote` folder.
```bash
python3 remote/inferenceServer.py
```
2. Run code on the robot. Example present in `bashfiles/run.sh`, Copy paste the laptop server URI, printed in the console when launched when prompted.

# Launch code in simulation
Eveything is done on the laptop. Replace the flag `--mode` with `simGui`.
