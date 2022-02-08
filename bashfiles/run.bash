echo "Specify the URI of remote policy:"
read URI
sudo python3 policy_test.py -w ./weights/m72n4bon/mass/185d07w7/full_3200.pt --mode hdw --kp_policy 40.0 --kd_policy 0.5 --nsteps 1000 --time_step 0.001 --run_hdw -fic --fic_policy_dt 0.026 --fic_ll_dt 0.004 --uri $URI
