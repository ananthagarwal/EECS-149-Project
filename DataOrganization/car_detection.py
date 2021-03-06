import sensor_msgs.point_cloud2 as pc2
import rosbag
import csv
import os

path = './cole-driving-downtown'


def distance_from_center(x, y, z):
    return (x ** 2 + y ** 2 + z ** 2) ** .5


with open('lidar.csv', 'w') as lidar_file:
    file_writer = csv.writer(
        lidar_file,
        delimiter=',',
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL)

    file_writer.writerow(["rosbag_time", "close", "front close", "front medium", "front far"])

    bags = []
    for filename in os.listdir(path):
        if filename.endswith('.bag'):
            bags.append(path + '/' + filename)
    bags.sort()

    bags = sorted([rosbag.Bag(b) for b in bags])

    for b in bags:
        for _, msg, t in b.read_messages(topics=['/velodyne_points']):
            close = 0
            total = 0
            front_close_dot_num, front_medium_dot_num, front_far_dot_num = 0, 0, 0
            for p in pc2.read_points(msg):
                total += 1

                if p[2] < 0 and distance_from_center(p[0], p[1], p[2]) < 1.4:
                    close += 1

                # front_close_dot_num
                if p[1] > -1.4 and p[1] < 1.4 and p[2] > -1.6 and p[2] < 0:
                    if p[0] <= 5.4 and p[0] > 1.4:
                        front_close_dot_num += 1
                    elif p[0] > 5.4 and p[0] <= 9.4:
                        front_medium_dot_num += 1
                    elif p[0] > 9.4:
                        front_far_dot_num += 1

            file_writer.writerow([t, close, front_close_dot_num, front_medium_dot_num, front_far_dot_num])
