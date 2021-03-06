import os
import argparse
from sensor_types import *

np.set_printoptions(threshold=np.inf)

ap = argparse.ArgumentParser(description="Master CSV/Pickle generator. Example usage: python2.7 create_master_csv.py -f cole1 \
-b pressure-sensor/cole. MUST run bag2csv.py in cole1 first to create rosbag csvs to collate and synchronize.")
ap.add_argument("-f", nargs=1, dest='folder',
                help="Folder name of collated rosbags. If not specified, default is ./exp_rosbags/")
ap.add_argument("-b", nargs=1, dest='bps_name',
                help="Location of BPS sensor M and C CSVs. If not specified, default is pressure-sensor/bps-.")
ap.add_argument("-c", nargs=1, dest='csv_folder',
                help="Folder to place CSVs in. If not specified, default is ./exp_rosbags/CSVs/")
ap.add_argument("-n", action="store_true", dest='no_merge',
                help="Skip collating rosbag CSVs. If not specified, default false.")
ap.add_argument("-r", action="store_true", dest='no_bag_convert',
                help="Skip converting rosbags to CSVs. If not specified, default false.")
ap.add_argument("-p", action="store_true", dest='extra_info',
                help="Also store picture data and LIDAR data in master.csv.")
ap.add_argument("-d", action="store_false", dest='daylight_savings',
				help="If flag set, data was recorded in PDT. Otherwise, PST.")

args = ap.parse_args()

"""
Example usage:
python2.7 create_master_csv.py -f cole1 -b cole
Creates master CSV file dataset.csv containing synchronized data for all sensors in cole1.
Name of BPS sensor CSVs is overriden from default "M.csv" and "C.csv" 
to "cole_M.csv" and "cole_C.csv"
"""


def merge_rosbag_csvs():
    data_categories = ["steering_ang", "steering_torq", "suspension", "tire_press", "acc_ped_eng", \
                       "brake_ped", "brake_torq", "gear", "imu_data_raw", "twist", "wheel_speeds", "turn_sig"]
    data_prefix = "vehicle_"

    # For each data type
    for category in data_categories:
        intermediate_csv = csv_path + category + ".csv"
        with open(intermediate_csv, "a") as fout:
            folders = sorted(os.listdir(csv_path))
            # Go through rosbags sequentially to fill data_type.csv
            for rosbag_folder in folders:
                folder_path = os.path.join(csv_path, rosbag_folder)
                if rosbag_folder.startswith('.') or os.path.isfile(folder_path):
                    continue
                csv_name = data_prefix + category + "-" + rosbag_folder + ".csv"
                file_path = os.path.join(folder_path, csv_name)
                if os.path.isfile(file_path):
                    with open(file_path) as f:
                        f.readline()
                        for line in f:
                            fout.write(line)


# 11/2/2018 3:30:55.73 PM


def extract_data(filename, frame_data):
    """
    @param filename             Name of vehicle sensor data csv to parse
    @param frame_data           List of frames. Each frame corresponds to all synchronized sensor data

    @return synch_vec_data


    Outputs
    """
    sync_vec_data = []
    sync_frame_data = []
    with open(filename + '.csv') as csv_file:
        # 11/30/18: Changed i, first_ind to =1, since i = 0 is header
        csv_reader, first_ind, i, j = csv.reader(
            csv_file, delimiter=','), 1, 1, 0
        final_list = []
        for elem_temp in csv_reader:
            final_list.append(elem_temp)
        while i < len(final_list) and j < len(frame_data):
            elem = final_list[i]
            final, curr = int(float(elem[0])), frame_data[j].time
            if args.daylight_savings:
            	gmt_offset = 8 * 3600 * (10 ** 9)
            else:
            	gmt_offset = 7 * 3600 * (10 ** 9)
            diff = (curr - final + gmt_offset) / (10 ** 9)
            if diff < -0.05:
                j += 1
            elif abs(diff) < 0.05:
                sync_vec_data.append(elem)
                sync_frame_data.append(frame_data[j])
                i, j = i + 1, j + 1
            else:
                i += 1
    print("Finished synchronizing with " + file_name + ". Total data points: ")
    print(len(sync_vec_data), len(sync_frame_data))
    return sync_vec_data, sync_frame_data


files = {
    'acc_ped_eng': AcceleratorPedalFrame,
    'brake_ped': BrakePedFrame,
    'brake_torq': BrakeTorqFrame,
    'gear': GearFrame,
    'imu_data_raw': IMUFrame,
    #'lidar': LidarFrame,
    'steering_ang': SteeringAngleFrame,
    'steering_torq': SteeringTorqueFrame,
    'suspension': VehicleSuspensionFrame,
    # 'tire_press': TirePressureFrame,
    # 'turn_sig': TurnSignalFrame,
    'twist': VehicleTwistFrame,
    'wheel_speeds': VehicleWheelSpeedsFrame,
}

if not args.folder:
    print("===Warning===\nNo folder name specified; using ./CSVs as default folder location\n=============")
    rosbag_path = "./exp_rosbags/"
else:
    rosbag_path = "./" + str(args.folder[0]) + "/"

if not args.csv_folder:
    csv_path = rosbag_path + "CSVs/"
else:
    csv_path = rosbag_path + str(args.csv_folder[0])

if not args.no_merge:
    merge_rosbag_csvs()

bps_path = rosbag_path + str(args.bps_name[0]) + '_' if args.bps_name else rosbag_path + "pressure-sensor/bsp-"

info, final_frames = extract_body_pressure_sensor_m(bps_path + 'M')

extract_body_pressure_sensor_c(bps_path + 'C', final_frames)

print("Body pressure sensor C and M data stored.")
print(len(final_frames))

for file_name, class_obj in files.items():
    try:
    	rows, final_frames = extract_data(csv_path + file_name, final_frames)
    except:
    	print("Error: Data is missing " + file_name + " data. Master CSV creation halted.")
    	exit()
    for k in range(len(rows)):
        class_obj.parse(rows[k], final_frames[k])

frames = Dataset(final_frames)
frames.to_csv(csv_path + "MASTER.csv")
frames.pickler("MASTER.p")
