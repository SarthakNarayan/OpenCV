# import the necessary packages
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--name_u", "-n",
	help="name of the user" , default = "sarthak")
ap.add_argument("--age", "-a",
	help="age of the user",type = int , default = 20)
args = vars(ap.parse_args())

# display a friendly message to the user
print("Hi there {}, it's nice to meet you! your age is {}".format(args["name_u"] , args["age"]))
print(args)
