from remote.laptopClient import LaptopPolicy

input("PRESS ENTER IF YOU WANT TO START THE LAPTOP SERVER ....")
policy = LaptopPolicy()
try:
    policy.inference_loop()
    policy.write_data_to_csv()
except Exception as e:
    print(f"Exception : {e}")
    policy.write_data_to_csv()
