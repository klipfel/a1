from remote.laptopClient import LaptopPolicy

input("PRESS ENTER IF YOU WANT TO START THE LAPTOP SERVER ....")
policy = LaptopPolicy()
while True:
    policy.inference_loop()
