#run all the python files in the water_dimer directory
import os
import glob
import subprocess

# get a list of all the python files in the water_dimer directory
py_files = glob.glob("*.py")

# loop through the python files and run them
for py_file in py_files:
    subprocess.run(["python", py_file])

