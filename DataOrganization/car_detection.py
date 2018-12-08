import sensor_msgs.point_cloud2 as pc2
import rosbag

bag = rosbag.Bag('turn_analyze/2018-07-12-12-57-29_0.bag')


def distance_from_center(x, y, z):
    return (x**2 + y**2 + z**2)**.5


t_s = -1

for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
    if t_s == -1:
        t_s = t
    close = 0
    total = 0
    for p in pc2.read_points(msg):
        total += 1
        if p[2] < 0 and distance_from_center(p[0],p[1],p[2]) < 1.4:
            close += 1

    print t-t_s, close, total