import sensor_msgs.point_cloud2 as pc2
import rosbag

bag = rosbag.Bag('turn_analyze/2018-11-19-12-54-56_11.bag')


def distance_from_center(x, y, z):
    return (x**2 + y**2 + z**2)**.5


t_s = -1


def dot_detection():
    global t_s
    for _, msg, t in bag.read_messages(topics=['/velodyne_points']):
        if t_s == -1:
            t_s = t
        close = 0
        total = 0
        print(t)
        print(msg)
        for p in pc2.read_points(msg):
            total += 1
            if p[2] < 0 and distance_from_center(p[0],p[1],p[2]) < 1.4:
                close += 1
    return close

"""
MAKE A CSV called lidar.csv with ROSBAG EPOCH TIME, CLOSE
"""
#def to_csv():
#    open ('lidar.csv') as csv_file:

    #print(t-t_s, close, total)


dot_detection()
