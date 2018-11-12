
arr = []

class AcceleratorPedalFrame(obj):
	def __init__(self, throttle_rate, throttle_pc, engine_rpm):
		self.throttle_rate = throttle_rate
		self.throttle_pc = throttle_pc
		self.engine_rpm = engine_rpm

class BrakeFrame(object):
	def __init__(self, brake_torque_request, brake_torque_actual, vehicle_speed, brake_pedal_boo):
		self.brake_torque_request = brake_torque_request
		self.brake_torque_actual = brake_torque_actual
		self.vehicle_speed = vehicle_speed
		self.brake_pedal_boo = brake_pedal_boo

class GearFrame(object):
	def __init__(self, gear):
		self.gear = gear

class SteeringWheelFrame(object):
	def __init__(self, steering_wheel_angle, steering_wheel_torque):
		self.steering_wheel_angle = steering_wheel_angle
		self.steering_wheel_torque = steering_wheel_torque
		
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

class VehicleSuspensionFrame(object):
	def __init__(self, ftont, rear):
		self.front = front
		self.rear = rear

class TirePressureFrame(object):
	def __init__(self, lf, rf, rr_orr, lr_olr, rr_irr, lr_ilr):
		self.lf = lf
		self.rf = rf
		self.rr_orr = rr_orr
		self.lr_olr = lr_olr
		self.rr_irr = rr_irr
		self.lr_ilr = lr_ilr

class TurnSignalFrame(object):
	def __init__(self, value):
		self.value = value
		
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
	
class VehicleWheelSpeeds(object):
	def __init__(self, front_left, front_right, rear_left, rear_right'):
		self.front_left = front_left
		self.front_right = front_right
		self.rear_left = rear_left
		self.rear_right = rear_right
		

class Frame(obj):
	def __init__(self, time, body_pressure_frame, accelerator_pedal_frame, brake_frame, gear_frame, steering_wheel_frame, imu_frame, vehicle_suspension_frame, tire_pressure_frame, turn_signal_frame, vehicle_twist_frame, vehicle_wheel_speeds_frame):
		
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