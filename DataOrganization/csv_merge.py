import pandas as pd
import os
import csv 

"""
Use: Place in ROSBAG
"""

path_to_csvs = "./CSVs"
data_prefix = "vehicle_"

data_categories = ["steering_ang", "steering_torq", "suspension", "tire_press","acc_ped_eng", "brake_ped", "brake_torq", "gear", "imu_data_raw"]

for category in data_categories:
	master_csv = path_to_csvs + "/" + category + ".csv"
	with open(master_csv, "a") as fout:
		for input_folder in os.listdir(path_to_csvs):
			if input_folder.startswith(b'.'):
				continue
			csv_name = data_prefix + category + "-" + input_folder + ".csv"
			file_path = path_to_csvs + "/" + input_folder + "/" + csv_name
			if os.path.isfile(file_path):
				with open(file_path) as f:
					f.next()
					for line in f:
						fout.write(line)
