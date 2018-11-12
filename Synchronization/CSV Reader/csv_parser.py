import csv
import numpy as np
from datetime import datetime

class BPS:
    def __init__(self, array):
        self.count, self.datetime, self.sum = process_title(array[0])
        # print(self.datetime.time())
        matrix_to_make = []
        for elem in array[1:]:
            matrix_to_make.append([float(data) for data in elem])
        self.mat = np.asmatrix(np.array(matrix_to_make))
        self.cog = [0, 0]      

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

def extract_M(filename):
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
                        all_frames.append(BPS(current_frame))
                else:
                    all_frames.append(BPS(current_frame))
                    current_frame = []
            i += 1
    return dataset_info, all_frames

def extract_C(filename, frame_data):
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
                    frame_data[j].cog = [int(round(row_curr / row_measure)),
                        int(round(col_curr / row_measure))]
                    j += 1
            i += 1   


info, frames = extract_M('cole2_M')
extract_C('cole2_C', frames)






                
            
