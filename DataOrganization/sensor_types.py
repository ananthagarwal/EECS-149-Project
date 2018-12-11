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

def process_title(arr):
    '''
    Extract frame number and sum and convert timestamp into format "%m/%d/%Y %I:%M:%S.%f %p"
    :param arr: <class 'list'>: e.g. ['Frame 1', ' 0', '', ' 11/2/2018 1:46:08.83PM', '', '', 'Raw Sum=', '', '136309']
    :return: frame_count, datetime_obj, frame_sum
    '''
    frame_count = int(arr[0].split()[1])
    frame_dt = arr[3].strip().split()
    if "." in frame_dt[1]:
        time_string = frame_dt[0] + " " + \
                      frame_dt[1][:-2] + " " + frame_dt[1][-2:]
    else:
        time_string = frame_dt[0] + " " + \
                      frame_dt[1][:-2] + ".0 " + frame_dt[1][-2:]
    # print(time_string)
    datetime_obj = datetime.strptime(time_string, "%m/%d/%Y %I:%M:%S.%f %p")
    frame_sum = float(arr[-1])
    return frame_count, datetime_obj, frame_sum


def extract_body_pressure_sensor_m(filename):
    '''
    :param filename: File containing BPS data
    :return: dataset_info: header of filename (32 rows)
    all_frames: list of Frames, each corresponding to a BPS frame from filename (each .1 seconds apart)
    '''
    with open(filename + '.csv') as csv_file:
        csv_reader, i = csv.reader(csv_file, delimiter=','), 0
        dataset_info, current_frame, all_frames = {}, [], []
        intro = True
        for elem in csv_reader:
            if elem and intro:
                # Builds a dictionary structure of BPS metadata, e.g. DATA_TYPE: MOVIE, MAP_INDEX: 0
                raw_info = elem[0].split()
                dataset_info[raw_info[0]] = raw_info[1:]
                print(elem)
            else:
                if intro:
                    intro = False
                    continue
                if elem:
                    if elem[0] != '@@':
                        current_frame.append(elem)
                    else:
                        all_frames.append(
                            Frame(
                                body_pressure_frame=BodyPressureSensorFrame(current_frame)))
                else:
                    # Finished finding all rows of a frame, so add a BodyPressureSensorFrame object to the list of objects.
                    all_frames.append(
                        Frame(
                            body_pressure_frame=BodyPressureSensorFrame(current_frame)))
                    current_frame = []
            i += 1

    for a in all_frames:
        a.time = a.body_pressure.epoch

    return dataset_info, all_frames


def extract_body_pressure_sensor_c(filename, frame_data):
    '''
    Since center of gravity data is in cm, convert to row and column indices
    :param filename: CSV containing center of gravity data
    :param frame_data:
    :return:
    '''
    with open(filename + '.csv') as csv_file:
        csv_reader, i, j = csv.reader(csv_file, delimiter=','), 0, 0
        row_measure, col_measure, intro = 1.016, 1.016, True
        for elem in csv_reader:
            if elem and intro:
                continue
            else:
                if intro:
                    intro = False
                    continue
                if elem[0] != '@@':
                    row_curr, col_curr = float(elem[3]), float(elem[4])
                    # 12/3/2018: Changed to dividing col_curr by col measure

                    frame_data[j].body_pressure.cog = [int(round(row_curr / row_measure)),
                                                       int(round(col_curr / col_measure))]
                    j += 1
            i += 1

