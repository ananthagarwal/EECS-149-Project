from sensor_types import *

LEFT_TURN_THRESHOLD = 60
RIGHT_TURN_THRESHOLD = -60
SPEED_UP_TIME = 3
MASTER_CSV_INTERVAL = 0.1


def is_right_turn(frame_index):
    frame = frames[frame_index]
    left_wheel_speed = frame.vehicle_wheel_speeds.front_left
    right_wheel_speed = frame.vehicle_wheel_speeds.front_right
    steering_angle = frame.steering_ang.steering_wheel_angle
    return steering_angle < RIGHT_TURN_THRESHOLD and (left_wheel_speed - right_wheel_speed) >= 1.0


def is_left_turn(frame_index):
    frame = frames[frame_index]
    left_wheel_speed = frame.vehicle_wheel_speeds.front_left
    right_wheel_speed = frame.vehicle_wheel_speeds.front_right
    steering_angle = frame.steering_ang.steering_wheel_angle
    return steering_angle > LEFT_TURN_THRESHOLD and (right_wheel_speed - left_wheel_speed) >= 1.0


def traffic_light_behavior(frame_index):
    """
    0 - Cautious
    1 - Aggressive
    """
    frame = frames[frame_index]
    init_throttle = frame.accelerator_pedal.throttle_rate
    init_brake = frame.brake_torq.brake_torque_request
    avg_throttle = sum([frames[frame_index + idx].accelerator_pedal.throttle_rate for idx in
                        range(SPEED_UP_TIME / MASTER_CSV_INTERVAL)]) // (SPEED_UP_TIME / MASTER_CSV_INTERVAL)
    avg_brake = sum([frames[frame_index + idx].brake_torq.brake_torque_request for idx in
                     range(SPEED_UP_TIME / MASTER_CSV_INTERVAL)]) // (SPEED_UP_TIME / MASTER_CSV_INTERVAL)

    # Default behavior: Slow down upon seeing the yellow light. Expected, cautious.
    if avg_brake > init_brake and avg_throttle <= init_throttle:
        return 0
    # If we sped up after seeing the yellow light, reckless
    elif avg_throttle > init_throttle and avg_brake <= init_brake:
        return 1
    else:
        return 1


frames = Dataset.loader("MASTER.p")
