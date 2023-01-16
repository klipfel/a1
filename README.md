Repository to transfer policy on real A1 robot.

# Installation
## Laptop packages
Packages contained in `prerequisites.txt`.
Recommended to use conda env.
```bash
pip install prerequisites.txt
```
## virtualenv and conda conflicts
If conda is initialized by default when the shell opens, and if you use virtualenv deactivate the (base) environment: run
`conda deactivate` until you go out of the conda base environment.

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
Launch:
```bash
python3 policy_test.py -w ./weights/delay-rand-m72n4bon/mass/185d07w7/full_3200.pt --mode simGui --kp_policy 40.0 --kd_policy 0.5 --nsteps 1000 --time_step 0.001 --run_hdw -fic --fic_policy_dt 0.026 --fic_ll_dt 0.004 -v
```
