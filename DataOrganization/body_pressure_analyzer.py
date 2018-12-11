import pickle
import numpy as np
import csv
from datetime import datetime


class BodyPressureSensorFrame(object):
    selected_columns = ["epoch", "body_pressure_data", "center_of_gravity"]

    def __init__(self, array):
        '''
        :param array: format is [[Frame i, ..., date/time, ... Raw Sum][data (3, 0, 7, 18, 8, ...)][data (0, 2, 9. 17, ...)] ...]
        '''
        self.count, self.datetime, self.sum = process_title(array[0])
        #
        self.epoch = ((self.datetime - datetime(1970, 1, 1)).total_seconds()) * (10 ** 9) + 8 * 3600 * (10 ** 9)
        matrix_to_make = []
        # build a matrix containing all the data for ths frame.
        for elem in array[1:]:
            matrix_to_make.append([float(data) for data in elem])
        self.mat = np.array(np.array(matrix_to_make))
        self.cog = [0, 0]

    def to_csv_row(self):
        return [self.epoch, self.mat, self.cog]


class AcceleratorPedalFrame(object):
    selected_columns = ["throttle_rate", "throttle_pc", "engine_rpm"]

    def __init__(self, throttle_rate, throttle_pc, engine_rpm):
        self.throttle_rate = throttle_rate
        self.throttle_pc = throttle_pc
        self.engine_rpm = engine_rpm

    def to_csv_row(self):
        return [str(self.throttle_rate), str(
            self.throttle_pc), str(self.engine_rpm)]

    @classmethod
    def parse(cls, row, frame):
        frame.accelerator_pedal = AcceleratorPedalFrame(
            float(row[7]),
            float(row[8]),
            float(row[9])
        )


class BrakePedFrame(object):
    selected_columns = ["brake_pedal_boo"]

    def __init__(self, brake_pedal_boo):
        self.brake_pedal_boo = brake_pedal_boo

    def to_csv_row(self):
        return [str(i) for i in [self.brake_pedal_boo]]

    @classmethod
    def parse(cls, row, frame):
        # TODO FIGURE THIS OUT
        frame.brake_ped = cls(str(row[7]))


class BrakeTorqFrame(object):
    selected_columns = ["brake_torque_request", "brake_torque_actual", "vehicle_speed"]

    def __init__(
            self,
            brake_torque_request,
            brake_torque_actual,
            vehicle_speed):
        self.brake_torque_request = brake_torque_request
        self.brake_torque_actual = brake_torque_actual
        self.vehicle_speed = vehicle_speed

    def to_csv_row(self):
        return [
            str(i) for i in [
                self.brake_torque_request,
                self.brake_torque_actual,
                self.vehicle_speed]]

    @classmethod
    def parse(cls, row, frame):
        # TODO FIGURE THIS OUT
        frame.brake_torq = cls(float(row[7]), float(row[8]), float(row[9]))


class GearFrame(object):
    selected_columns = ["gear"]

    def __init__(self, gear):
        self.gear = gear

    def to_csv_row(self):
        return [str(self.gear)]

    @classmethod
    def parse(cls, row, frame):
        frame.gear = cls(int(row[8]))


class SteeringTorqueFrame(object):
    selected_columns = ["steering_wheel_torque"]

    def __init__(self, steering_wheel_torque):
        self.steering_wheel_torque = steering_wheel_torque

    def to_csv_row(self):
        return [str(self.steering_wheel_torque)]

    @classmethod
    def parse(cls, row, frame):
        frame.steering_torq = cls(float(row[7]))


class SteeringAngleFrame(object):
    selected_columns = ["steering_wheel_angle"]

    def __init__(self, steering_wheel_angle):
        self.steering_wheel_angle = steering_wheel_angle

    def to_csv_row(self):
        return [str(self.steering_wheel_angle)]

    @classmethod
    def parse(cls, row, frame):
        frame.steering_ang = cls(float(row[7]))


class IMUFrame(object):
    selected_columns = ["orientation_x", "orientation_y", "orientation_z", "orientation_w", "orientation_covariance",
                        "angular_velocity_x", "angular_velocity_y", "angular_velocity_z", "angular_velocity_covariance",
                        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
                        "linear_acceleration_covariance"]

    def __init__(
            self,
            orientation_x,
            orientation_y,
            orientation_z,
            orientation_w,
            orientation_covariance,
            angular_velocity_x,
            angular_velocity_y,
            angular_velocity_z,
            angular_velocity_covariance,
            linear_acceleration_x,
            linear_acceleration_y,
            linear_acceleration_z,
            linear_acceleration_covariance):
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

    def to_csv_row(self):
        return [
            str(i) for i in [
                self.orientation['x'],
                self.orientation['y'],
                self.orientation['z'],
                self.orientation['w'],
                self.orientation['covariance'],
                self.angular_velocity['x'],
                self.angular_velocity['y'],
                self.angular_velocity['z'],
                self.angular_velocity['covariance'],
                self.linear_acceleration['x'],
                self.linear_acceleration['y'],
                self.linear_acceleration['z'],
                self.linear_acceleration['covariance']]]

    @classmethod
    def parse(cls, row, frame):
        frame.imu = IMUFrame(float(row[8]),
                             float(row[9]),
                             float(row[10]),
                             float(row[11]),
                             [float(i) for i in row[12][1:-1].split(',')],
                             float(row[14]),
                             float(row[15]),
                             float(row[16]),
                             [float(i) for i in row[17][1:-1].split(',')],
                             float(row[19]),
                             float(row[20]),
                             float(row[21]),
                             [float(i) for i in row[22][1:-1].split(',')])


class LidarFrame():
    selected_columns = ["close_dots", "front_close_dots", "front_medium_dots", "front_far_dots"]

    def __init__(self, close_dots, front_close_dots, front_medium_dots, front_far_dots):
        self.close_dots = close_dots
        self.front_close_dots = front_close_dots
        self.front_medium_dots = front_medium_dots
        self.front_far_dots = front_far_dots

    def to_csv_row(self):
        return [str(self.close_dots), str(self.front_close_dots), str(self.front_medium_dots), str(self.front_far_dots)]

    @classmethod
    def parse(cls, row, frame):
        frame.lidar_frame = LidarFrame(int(row[1]), int(row[2]), int(row[3]), int(row[4]))


class VehicleSuspensionFrame(object):
    selected_columns = ["front", "rear"]

    def __init__(self, front, rear):
        self.front = front
        self.rear = rear

    def to_csv_row(self):
        return [str(self.front), str(self.rear)]

    @classmethod
    def parse(cls, row, frame):
        frame.vehicle_suspension = VehicleSuspensionFrame(
            float(row[7]), float(row[8]))


class TirePressureFrame(object):
    selected_columns = ["tire_press_lf", "tire_press_rf", "tire_press_rr_orr", "tire_press_lr_olr", "tire_press_rr_irr",
                        "tire_press_lr_ilr"]

    def __init__(self, lf, rf, rr_orr, lr_olr, rr_irr, lr_ilr):
        self.lf = lf
        self.rf = rf
        self.rr_orr = rr_orr
        self.lr_olr = lr_olr
        self.rr_irr = rr_irr
        self.lr_ilr = lr_ilr

    def to_csv_row(self):
        return [
            str(i) for i in [
                self.lf,
                self.rf,
                self.rr_orr,
                self.lr_olr,
                self.rr_irr,
                self.lr_ilr]]

    @classmethod
    def parse(cls, row, frame):
        frame.tire_pressure = TirePressureFrame(
            float(
                row[7]), float(
                row[8]), float(
                row[9]), float(
                row[10]), float(
                row[11]), float(
                row[12]))


class TurnSignalFrame(object):
    selected_columns = ["value"]

    def __init__(self, value):
        self.value = value

    def to_csv_row(self):
        return [str(self.value)]

    @classmethod
    def parse(cls, row, frame):
        frame.turn_signal = TurnSignalFrame(float(row[8]))


class VehicleTwistFrame(object):
    selected_columns = ["twist_linear_x", "twist_linear_y", "twist_linear_z", "twist_angular_x", "twist_angular_y",
                        "twist_angular_z"]

    def __init__(
            self,
            linear_x,
            linear_y,
            linear_z,
            angular_x,
            angular_y,
            angular_z):
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

    def to_csv_row(self):
        return [
            str(i) for i in [
                self.linear['x'],
                self.linear['y'],
                self.linear['z'],
                self.angular['x'],
                self.angular['y'],
                self.angular['z']]]

    @classmethod
    def parse(cls, row, frame):
        frame.vehicle_twist = VehicleTwistFrame(
            float(
                row[9]), float(
                row[10]), float(
                row[11]), float(
                row[13]), float(
                row[14]), float(
                row[15]))


class VehicleWheelSpeedsFrame(object):
    selected_columns = ["front_left", "front_right", "rear_left", "rear_right"]

    def __init__(self, front_left, front_right, rear_left, rear_right):
        self.front_left = front_left
        self.front_right = front_right
        self.rear_left = rear_left
        self.rear_right = rear_right

    def to_csv_row(self):
        return [
            str(i) for i in [
                self.front_left,
                self.front_right,
                self.rear_left,
                self.rear_right]]

    @classmethod
    def parse(cls, row, frame):
        frame.vehicle_wheel_speeds = VehicleWheelSpeedsFrame(
            float(
                row[7]), float(
                row[8]), float(
                row[9]), float(
                row[10]))


class Frame(object):

    def __init__(
            self,
            time=None,
            body_pressure_frame=None,
            accelerator_pedal_frame=None,
            brake_ped_frame=None,
            brake_torq_frame=None,
            gear_frame=None,
            steering_torq_frame=None,
            steering_ang_frame=None,
            imu_frame=None,
            vehicle_suspension_frame=None,
            # tire_pressure_frame=None,
            # turn_signal_frame=None,
            vehicle_twist_frame=None,
            vehicle_wheel_speeds_frame=None,
            lidar_frame=None):
        self.time = time

        self.body_pressure = body_pressure_frame

        self.accelerator_pedal = accelerator_pedal_frame

        self.brake_ped = brake_ped_frame

        self.brake_torq = brake_torq_frame

        self.gear = gear_frame

        self.steering_torq = steering_torq_frame

        self.steering_ang = steering_ang_frame

        self.imu = imu_frame

        self.vehicle_suspension = vehicle_suspension_frame

        # self.tire_pressure = tire_pressure_frame

        # self.turn_signal = turn_signal_frame

        self.vehicle_twist = vehicle_twist_frame

        self.vehicle_wheel_speeds = vehicle_wheel_speeds_frame

        self.lidar_frame = lidar_frame

    def to_csv_row(self):
        row = []

        # time
        row.append(str(self.body_pressure.datetime))
        # Alternative: row.append(self.time.utcnow().strftime('%Y-%m-%d
        # %H:%M:%S.%f'))
        row.extend(self.body_pressure.to_csv_row())
        row.extend(self.accelerator_pedal.to_csv_row())
        row.extend(self.brake_torq.to_csv_row())
        row.extend(self.brake_ped.to_csv_row())
        row.extend(self.gear.to_csv_row())
        row.extend(self.steering_ang.to_csv_row())
        row.extend(self.steering_torq.to_csv_row())
        row.extend(self.imu.to_csv_row())
        row.extend(self.vehicle_suspension.to_csv_row())
        # row.extend(self.tire_pressure.to_csv_row())
        # row.extend(self.turn_signal.to_csv_row())
        row.extend(self.vehicle_twist.to_csv_row())
        row.extend(self.vehicle_wheel_speeds.to_csv_row())
        row.extend(self.lidar_frame.to_csv_row())

        return row

class Dataset(object):

    def __init__(self, frames=[]):
        self.frames = frames

    def pickler(self, file_name="dataset.p"):
        with open(file_name, 'wb') as p_file:
            pickle.dump(self.frames, p_file, protocol=2)

    def column_names_row(self):
        column_names = []
        column_names.extend(["timestamp"])

        column_names.extend(BodyPressureSensorFrame.selected_columns)
        column_names.extend(AcceleratorPedalFrame.selected_columns)
        column_names.extend(BrakeTorqFrame.selected_columns)
        column_names.extend(BrakePedFrame.selected_columns)
        column_names.extend(GearFrame.selected_columns)
        column_names.extend(SteeringAngleFrame.selected_columns)
        column_names.extend(SteeringTorqueFrame.selected_columns)
        column_names.extend(IMUFrame.selected_columns)
        column_names.extend(VehicleSuspensionFrame.selected_columns)
        # column_names.extend(TirePressureFrame.selected_columns)
        # column_names.extend(TurnSignalFrame.selected_columns)
        column_names.extend(VehicleTwistFrame.selected_columns)
        column_names.extend(VehicleWheelSpeedsFrame.selected_columns)
        column_names.extend(LidarFrame.selected_columns)
        return column_names

    @classmethod
    def loader(self, filename="dataset.p"):
        return pickle.load(open(filename, "rb"))

    def to_csv(self, filename="dataset.csv"):
        with open(filename, 'w') as file:
        #with open(filename, 'wb') as file:
            file_writer = csv.writer(
                file,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)

            file_writer.writerow(self.column_names_row())
            for frame in self.frames:
                file_writer.writerow(frame.to_csv_row())
""" 
class BodyPressureSensorFrame(object):
    def __init__(self, array):
        self.count, self.datetime, self.sum = process_title(array[0])
        self.epoch = ((self.datetime - datetime(1970, 1, 1)
                       ).total_seconds()) * (10 ** 9)
        matrix_to_make = []
        for elem in array[1:]:
            matrix_to_make.append([float(data) for data in elem])
        self.mat = np.asmatrix(np.array(matrix_to_make))
        self.cog = [0, 0]

    def to_csv_row(self):
        return [str(self.mat), str(self.cog)]

Important Things to Extract:
	1) differential from frame to frame
	2) differential over 50 frames / maximum frames (<5 seconds of differential)
		> region of greatest differential over time
		> LEFT / RIGHT / NEUTRAL
	4) cell of maximum pressure (to do so reliably, would need to look at adjacent
		cells to see if that pressure is distributed over a region or if it just an
		abberant piece of information)
	5) Base state matrix of M data to calculate differentials from

"""

def extract_body_pressure_sensor_m(filename):
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
                        all_frames.append(BodyPressureSensorFrame(current_frame))
                else:
                    all_frames.append(BodyPressureSensorFrame(current_frame))
                    current_frame = []
            i += 1

    return dataset_info, all_frames

def extract_body_pressure_sensor_c(filename, frame_data):
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
                    # 12/3/2018: Changed to dividing col_curr by col measure
                    frame_data[j].cog = [int(round(row_curr / row_measure)),
                    					int(round(col_curr / col_measure))]
                    j += 1
            i += 1


# a, b = extract_body_pressure_sensor_m("cole2final_M")
# # c, d = extract_body_pressure_sensor_m("turning_M")
# # e, f = extract_body_pressure_sensor_m("turning_2_M")

# extract_body_pressure_sensor_c("cole2final_C", b)
# # extract_body_pressure_sensor_c("turning_C", d)
# # extract_body_pressure_sensor_c("turning_2_C", f)

def baseline(array_frames, count):
	sum_matrix, sum_row_cog, sum_col_cog, i = array_frames[0].body_pressure.mat, array_frames[0].body_pressure.cog[0], array_frames[0].body_pressure.cog[1], 1
	while i < count:
		sum_matrix += array_frames[i].body_pressure.mat
		sum_row_cog += array_frames[i].body_pressure.cog[0]
		sum_col_cog += array_frames[i].body_pressure.cog[1]
		i += 1
	baseline, cog_avg = sum_matrix / count, [int(sum_row_cog / count), int(sum_col_cog / count)]
	left, right = baseline[:, :cog_avg[1]], baseline[:, cog_avg[1]:]
	tp_left, tp_right = left.sum(), right.sum()
	return baseline, cog_avg, 3 * abs(tp_left - tp_right)

def check_most(lst, x, c):
	counter = c
	for elem in lst:
		if elem != x:
			counter -= 1
		if counter <= 0:
			return False
	return True

def assign_fill(lst, count, item):
	i = -1
	while abs(i) <= count:
		lst[i]= item
		i -= 1

def emphasis(curr_frame, bl_count):
	bl, cog, th = baseline(array_frames, bl_count)
	to_return, c_count = [], bl_count

	# for curr_frame in array_frames:

	mat = curr_frame.body_pressure.mat - bl
	left, right = mat[:, :cog[1]], mat[:, cog[1]:]
	tp_left, tp_right = left.sum(), right.sum()
	
	if (abs(tp_left - tp_right)) <= th:
		# to_return.append(0)
		return 0
	elif tp_left > tp_right:
		# to_return.append(1)
		return 1
	else:
		# to_return.append(-1)
		return -1
		# if len(to_return) > c_count:
		# 	if check_most(to_return[-bl_count:], -1, 1) or check_most(to_return[-bl_count:], 1, 1):
				# assign_fill(to_return, c_count, 0)
				# bl, cog, th = baseline(array_frames[-bl_count:], bl_count)

	# return to_return

def emp(curr_frame, bl, cog, th):

	# for curr_frame in array_frames:

	mat = curr_frame.body_pressure.mat - bl
	left, right = mat[:, :cog[1]], mat[:, cog[1]:]
	tp_left, tp_right = left.sum(), right.sum()
	
	if (abs(tp_left - tp_right)) <= th:
		# to_return.append(0)
		return 0
	elif tp_left > tp_right:
		# to_return.append(1)
		return 1
	else:
		# to_return.append(-1)
		return -1
		# if len(to_return) > c_count:
		# 	if check_most(to_return[-bl_count:], -1, 1) or check_most(to_return[-bl_count:], 1, 1):
				# assign_fill(to_return, c_count, 0)
				# bl, cog, th = baseline(array_frames[-bl_count:], bl_count)

	# return to_return

x = Dataset()
x.loader('MASTER.p')
bl, cog, th = baseline(x.frames, 50)

def give_tilt(frame_index):
	if frame_index < 50:
		return 0
	else:
		return emp(x.frames[frame_index], bl, cog, th)









x = (emphasis(b, 200))
# y = (emphasis(d, 300))
# z = (emphasis(f, 300))