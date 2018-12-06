from sensor_types import *


def is_right_turn(frame_index):
    frame = frames[frame_index]
    left_wheel_speed = frame.vehicle_wheel_speeds.front_left
    right_wheel_speed = frame.vehicle_wheel_speeds.front_right
    return


def is_left_turn(frame_index):
    frame = frames[frame_index]
    left_wheel_speed = frame.vehicle_wheel_speeds.front_left
    right_wheel_speed = frame.vehicle_wheel_speeds.front_right
    return


def is_braking(frame_index):
    frame = frames[frame_index]
    brake_torq = frame.brake_torq.brake_torque_request
    return


def is_accel(frame_index):
    frame = frames[frame_index]
    accel = frame.accelerator_pedal.throttle_rate

    return


frames = Dataset.loader("MASTER.p")
