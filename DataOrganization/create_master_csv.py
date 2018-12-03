import csv
import numpy as np
import pickle
from datetime import datetime


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


class AcceleratorPedalFrame(object):
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
            float(
                row[7]), float(
                row[8]), float(
                row[9]))


class BrakePedFrame(object):
    def __init__(self, brake_pedal_boo):
        self.brake_pedal_boo = brake_pedal_boo

    def to_csv_row(self):
        return [str(i) for i in [self.brake_pedal_boo]]

    @classmethod
    def parse(cls, row, frame):
        # TODO FIGURE THIS OUT
        frame.brake_ped = cls(str(row[7]))


class BrakeTorqFrame(object):
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
    def __init__(self, gear):
        self.gear = gear

    def to_csv_row(self):
        return [str(self.gear)]

    @classmethod
    def parse(cls, row, frame):
        frame.gear = cls(int(row[8]))


class SteeringTorqueFrame(object):
    def __init__(self, steering_wheel_torque):
        self.steering_wheel_torque = steering_wheel_torque

    def to_csv_row(self):
        return [str(self.steering_wheel_torque)]

    @classmethod
    def parse(cls, row, frame):
        frame.steering_torq = cls(float(row[7]))


class SteeringAngleFrame(object):
    def __init__(self, steering_wheel_angle):
        self.steering_wheel_angle = steering_wheel_angle

    def to_csv_row(self):
        return [str(self.steering_wheel_angle)]

    @classmethod
    def parse(cls, row, frame):
        frame.steering_wheel = cls(float(row[7]))


class IMUFrame(object):
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


class VehicleSuspensionFrame(object):
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
    def __init__(self, value):
        self.value = value

    def to_csv_row(self):
        return [str(self.value)]

    @classmethod
    def parse(cls, row, frame):
        frame.turn_signal = TurnSignalFrame(float(row[8]))


class VehicleTwistFrame(object):
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


class VehicleWheelSpeeds(object):
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
        frame.vehicle_wheel_speeds = VehicleWheelSpeeds(
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
            tire_pressure_frame=None,
            turn_signal_frame=None,
            vehicle_twist_frame=None,
            vehicle_wheel_speeds_frame=None):

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

        self.tire_pressure = tire_pressure_frame

        self.turn_signal = turn_signal_frame

        self.vehicle_twist = vehicle_twist_frame

        self.vehicle_wheel_speeds = vehicle_wheel_speeds_frame

    def to_csv_row(self):
        row = []

        # time
        row.append(str(self.body_pressure.datetime))
        # Alternative: row.append(self.time.utcnow().strftime('%Y-%m-%d
        # %H:%M:%S.%f'))
        row.extend(self.body_pressure.to_csv_row())
        row.extend(self.accelerator_pedal.to_csv_row())
        row.extend(self.brake.to_csv_row())
        row.extend(self.gear.to_csv_row())
        row.extend(self.steering_wheel.to_csv_row())
        row.extend(self.imu.to_csv_row())
        row.extend(self.vehicle_suspension.to_csv_row())
        row.extend(self.tire_pressure.to_csv_row())
        row.extend(self.turn_signal.to_csv_row())
        row.extend(self.vehicle_twist.to_csv_row())
        row.extend(self.vehicle_wheel_speeds.to_csv_row())

        return row


class Dataset(object):

    def __init__(self, frames=[]):
        self.frames = frames

    def pickler(self, filename="dataset.p"):
        return pickle.dumps(self, open(filename, 'wb'))

    @classmethod
    def loader(self, filename="dataset.p"):
        return pickle.load(open(filename, "rb"))

    def to_csv(self, filename="dataset.csv"):
        # with open("dataset.csv", 'w', newline='') as file:
        with open(filename, 'wb') as file:
            file_writer = csv.writer(
                file,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)

            for frame in self.frames:
                file_writer.writerow(frame.to_csv_row)

# 11/2/2018 3:30:55.73 PM


def process_title(arr):
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
                        all_frames.append(
                            Frame(
                                body_pressure_frame=BodyPressureSensorFrame(current_frame)))
                else:
                    all_frames.append(
                        Frame(
                            body_pressure_frame=BodyPressureSensorFrame(current_frame)))
                    current_frame = []
            i += 1

    for a in all_frames:
        a.time = a.body_pressure.epoch

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
                    frame_data[j].body_pressure.cog = [int(round(row_curr / row_measure)),
                                                       int(round(col_curr / col_measure))]
                    j += 1
            i += 1


def extract_data(filename, frame_data):
    data_to_return = []
    with open(filename + '.csv') as csv_file:
        # 11/30/18: Changed i, first_ind to =1, since i = 0 is header
        csv_reader, first_ind, i, j = csv.reader(
            csv_file, delimiter=','), 1, 1, 0
        final_list = []
        for elem_temp in csv_reader:
            final_list.append(elem_temp)
        while i < len(final_list):
            elem = final_list[i]
            final, curr = (int(elem[0]) - 7 * 3600 *
                           (10**9)), frame_data[j].time
            diff = (curr - final) / (10 ** 9)
            if i == 1 and diff < -0.05:
                j += 1
            elif abs(diff) < 0.05:
                data_to_return.append(elem)
                if i == 1:
                    print(final, curr)
                    first_ind = j
                i, j = i + 1, j + 1
            else:
                i += 1
    final_frame_data = frame_data[first_ind:j]
    return data_to_return, final_frame_data


files = {
    'acc_ped_eng': AcceleratorPedalFrame,
    'brake_ped': BrakePedFrame,
    'brake_torq': BrakeTorqFrame,
    'gear': GearFrame,
    'imu_data_raw': IMUFrame,
    'steering_ang': SteeringAngleFrame,
    'steering_torq': SteeringTorqueFrame,
    'suspension': VehicleSuspensionFrame,
    'tire_press': TirePressureFrame
}


def main():
    extract_body_pressure_sensor_c(folder + 'cole_C', bps_frames)

    for file_name, class_obj in files.items():
        rows, final_frames = extract_data(folder + file_name, bps_frames)
        # print(len(rows), len(final_frames))
        # print(rows[len(rows) - 1])
        for row, frame in zip(rows, final_frames):
            class_obj.parse(row, frame)

    frames = Dataset(final_frames)
    frames.to_csv()


folder = 'cole1/'
info, bps_frames = extract_body_pressure_sensor_m(folder + 'cole_M')
main()
