import cv2
import argparse
import os

ap = argparse.ArgumentParser(description="plays all videos in a folder at the same time")
ap.add_argument("-f", nargs=1, dest='folder',
                help="Folder name of collated rosbags. If not specified, default is ./exp_rosbags/")

args = ap.parse_args()


def list_files(path):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        print(name)
        if name.endswith('.mp4'):
            files.append(path+'/'+name)
    return files


def video_joiner(folder_path):
    names = list_files(folder_path)

    print(names)

    cap = [cv2.VideoCapture(i) for i in names]

    frames = [None] * len(names)
    gray = [None] * len(names)
    ret = [None] * len(names)

    while True:

        for i, c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read()

        for i, f in enumerate(frames):
            if ret[i] is True:
                cv2.imshow(names[i], f)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for c in cap:
        if c is not None:
            c.release()

    cv2.destroyAllWindows()


video_joiner(args.folder[0])