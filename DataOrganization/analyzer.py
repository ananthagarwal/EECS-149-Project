from create_master_csv import *
import numpy as np

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

a, b = extract_body_pressure_sensor_m("cool_guy_M")

def diff(frame_first, frame_second):
	return frame_second.mat - frame_first.mat

def range_diff(array_frames, count):
	if len(array_frames) > count:
		array_frames = array_frames[(-1 * count):]
	return array_frames[-1].mat - array_frames[0].mat

def baseline(array_frames, count):
	sum_matrix, i = array_frames[0].mat, 1
	while i < count:
		sum_matrix += array_frames[i].mat
		i += 1
	return sum_matrix / count

def emphasis(mat, threshold):
	print('shape: ', mat.shape)
	l = float(mat.shape[1]) / 2
	other_half = int(l) + ((2 * l) % 2)

	left = mat[:, :int(l)]
	print('left: ', left.shape)
	right = mat[:, int(other_half):]
	print('right: ', right.shape)

	tp_left, tp_right = left.sum(), right.sum()

	if (tp_left == tp_right) or (tp_left <= threshold and tp_right <= threshold):
		return 'NEUTRAL'
	elif tp_left > tp_right:
		return 'LEFT'
	else:
		return 'RIGHT'

def bl_emphasis(array_frames, bl_count, threshold):
	bl = baseline(array_frames, bl_count)
	return [emphasis(curr_frame.mat - bl, threshold) for curr_frame in array_frames]




x = (bl_emphasis(b, 50, 50))