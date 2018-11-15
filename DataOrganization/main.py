import csv
import numpy as np
from datetime import datetime

class BodyPressureSensorFrame(object):
    def __init__(self, array):
        self.count, self.datetime, self.sum = process_title(array[0])
        self.epoch = ((self.datetime - datetime(1970, 1, 1)).total_seconds()) * (10 ** 9)
        matrix_to_make = []
        for elem in array[1:]:
            matrix_to_make.append([float(data) for data in elem])
        self.mat = np.asmatrix(np.array(matrix_to_make))
        self.cog = [0, 0]      

class AcceleratorPedalFrame(object):
	def __init__(self, throttle_rate, throttle_pc, engine_rpm):
		self.throttle_rate = throttle_rate
		self.throttle_pc = throttle_pc
		self.engine_rpm = engine_rpm

	@classmethod
	def parse(self, row, frame):
		frame.accelerator_pedal = AcceleratorPedalFrame(float(row[7]), float(row[8]), int(row[9]))
		print(row)
		

class BrakeFrame(object):
	def __init__(self, brake_torque_request, brake_torque_actual, vehicle_speed, brake_pedal_boo):
		self.brake_torque_request = brake_torque_request
		self.brake_torque_actual = brake_torque_actual
		self.vehicle_speed = vehicle_speed
		self.brake_pedal_boo = brake_pedal_boo

	@classmethod
	def parse(self, row, frame):
		#TODO FIGURE THIS OUT
		pass

class GearFrame(object):
	def __init__(self, gear):
		self.gear = gear

	@classmethod
	def parse(self, row, frame):
		frame.gear = GearFrame(int(row[8]))

class SteeringWheelFrame(object):
	def __init__(self, steering_wheel_angle, steering_wheel_torque):
		self.steering_wheel_angle = steering_wheel_angle
		self.steering_wheel_torque = steering_wheel_torque

	@classmethod
	def parse(self, row, frame):
		#TODO FIGURE THIS OUT
		frame.steering_wheel = SteeringWheelFrame()
		pass
		
class IMUFrame(object):
	def __init__(self, orientation_x, orientation_y, orientation_z, orientation_w, orientation_covariance, angular_velocity_x, angular_velocity_y, angular_velocity_z, angular_velocity_covariance, linear_acceleration_x, linear_acceleration_y, linear_acceleration_z, linear_acceleration_covariance):
		self.orientation = {
								'x': orientation_x,
								'y': orientation_y,
								'z': orientation_z,
								'w': orientation_w,
								'covariance': orientation_covariance
							}
		
		self.angular_velocity = {
									'x': angular_velocity_x,
									'y': angular_velocity_y,
									'z': angular_velocity_z,
									'covariance': angular_velocity_covariance
								}

		self.linear_acceleration = {
										'x': linear_acceleration_x,
										'y': linear_acceleration_y,
										'z': linear_acceleration_z,
										'covariance': linear_acceleration_covariance
									}

	@classmethod
	def parse(self, row, frame):
		frame.imu = IMUFrame(float(row[8]), float(row[9]), float(row[10]), float(row[11]), [float(i) for i in row[12][1:-1].split(',')], float(row[14]), float(row[15]), float(row[16]), [float(i) for i in row[17][1:-1].split(',')], float(row[19]), float(row[20]), float(row[21]), [float(i) for i in row[22][1:-1].split(',')])

class VehicleSuspensionFrame(object):
	def __init__(self, ftont, rear):
		self.front = front
		self.rear = rear

	@classmethod
	def parse(self, row, frame):
		frame.vehicle_suspension = VehicleSuspensionFrame(float(row[7]), float(row[8]))

class TirePressureFrame(object):
	def __init__(self, lf, rf, rr_orr, lr_olr, rr_irr, lr_ilr):
		self.lf = lf
		self.rf = rf
		self.rr_orr = rr_orr
		self.lr_olr = lr_olr
		self.rr_irr = rr_irr
		self.lr_ilr = lr_ilr

	@classmethod
	def parse(self, row, frame):
		frame.tire_pressure = TirePressureFrame(int(row[7]), int(row[8]), int(row[9]), int(row[10]), int(row[11]), int(row[12]), int(row[13]))

class TurnSignalFrame(object):
	def __init__(self, value):
		self.value = value

	@classmethod
	def parse(self, row, frame):
		frame.turn_signal = TurnSignalFrame(int(row[8]))
		
class VehicleTwistFrame(object):
	def __init__(self, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z):
		self.linear = {
						'x': linear_x,
						'y': linear_y,
						'z': linear_z,
					}

		self.angular = {
						'x': angular_x,
						'y': angular_y,
						'z': angular_z,
					}

	@classmethod
	def parse(self, row, frame):
		frame.vehicle_twist = VehicleTwistFrame(float(row[9]), float(row[10]), float(row[11]), float(row[13]), float(row[14]), float(row[15]))

	
class VehicleWheelSpeeds(object):
	def __init__(self, front_left, front_right, rear_left, rear_right):
		self.front_left = front_left
		self.front_right = front_right
		self.rear_left = rear_left
		self.rear_right = rear_right
	
	@classmethod
	def parse(self, row, frame):
		frame.vehicle_wheel_speeds = VehicleWheelSpeeds(float(row[7]), float(row[8]), float(row[9]), float(row[10]))

class Frame(object):


	def __init__(self, time=None, body_pressure_frame=None, accelerator_pedal_frame=None, brake_frame=None, gear_frame=None, steering_wheel_frame=None, imu_frame=None, vehicle_suspension_frame=None, tire_pressure_frame=None, turn_signal_frame=None, vehicle_twist_frame=None, vehicle_wheel_speeds_frame=None):
		
		self.time = time

		self.body_pressure = body_pressure_frame
		
		self.accelerator_pedal = accelerator_pedal_frame

		self.brake = brake_frame

		self.gear = gear_frame

		self.steering_wheel = steering_wheel_frame

		self.imu = imu_frame
		
		self.vehicle_suspension = vehicle_suspension_frame

		self.tire_pressure = tire_pressure_frame

		self.turn_signal = turn_signal_frame

		self.vehicle_twist = vehicle_twist_frame

		self.vehicle_wheel_speeds = vehicle_wheel_speeds_frame



# 11/2/2018 3:30:55.73 PM
def process_title(arr):
    frame_count = int(arr[0].split()[1])
    frame_dt = arr[3].strip().split()
    if "." in  frame_dt[1]:
        time_string = frame_dt[0] + " " + frame_dt[1][:-2] + " " + frame_dt[1][-2:]
    else:
        time_string = frame_dt[0] + " " + frame_dt[1][:-2] + ".0 " + frame_dt[1][-2:] 
    # print(time_string)
    datetime_obj = datetime.strptime(time_string , "%m/%d/%Y %I:%M:%S.%f %p")
    frame_sum = float(arr[-1])
    return frame_count, datetime_obj, frame_sum

def extract_body_pressure_sensor_M(filename):
    with open(filename + '.csv') as csv_file:
        csv_reader, i = csv.reader(csv_file, delimiter=','), 0
        dataset_info, current_frame, all_frames = {}, [], []
        for elem in csv_reader:
            if i < 32:
                if elem:
                    raw_info = elem[0].split()
                    dataset_info[raw_info[0]] = raw_info[1:]
            else:
                if elem:
                    if elem[0] != '@@':
                        current_frame.append(elem)
                    else:
                        all_frames.append(Frame(body_pressure_frame=BodyPressureSensorFrame(current_frame)))
                else:
                    all_frames.append(Frame(body_pressure_frame=BodyPressureSensorFrame(current_frame)))
                    current_frame = []
            i += 1

    for a in all_frames:
    	a.time = a.body_pressure.epoch

    return dataset_info, all_frames

def extract_body_pressure_sensor_C(filename, frame_data):
    with open(filename + '.csv') as csv_file:
        csv_reader, i, j = csv.reader(csv_file, delimiter=','), 0, 0
        for elem in csv_reader:
            if i < 30:
                if i == 11:
                    row_measure = float(elem[0].split()[1])
                if i == 12:
                    col_measure = float(elem[0].split()[1])
            else:
                if elem[0] != '@@':
                    row_curr, col_curr = float(elem[3]), float(elem[4])
                    frame_data[j].body_pressure.cog = [int(round(row_curr / row_measure)),
                        int(round(col_curr / row_measure))]
                    j += 1
            i += 1   

def extract_data(filename, frame_data):
    data_to_return = []
    with open(filename + '.csv') as csv_file:
        csv_reader, i, j = csv.reader(csv_file, delimiter=','), 0, 0
        for elem in csv_reader:
            if i > 1:
                final, curr = (int(elem[0]) - 7*3600*(10**9)), frame_data[j].time
                diff = (curr - final) / (10 ** 9)
                if abs(diff) < 0.05:
                    data_to_return.append(elem)
                    j += 1
            i += 1
    return data_to_return


folder = 'cole1/'

info, frames = extract_body_pressure_sensor_M(folder + 'cole_M')
extract_body_pressure_sensor_C(folder + 'cole_C', frames)


files = {
	'acc_ped_eng': AcceleratorPedalFrame
}

for file in files:
	rows = extract_data(folder + file, frames)

	print(rows)

	for row, frame in zip(rows, frames):
		AcceleratorPedalFrame.parse(row, frame)

