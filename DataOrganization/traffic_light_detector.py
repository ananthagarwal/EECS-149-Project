import cv2
import argparse
import numpy as np
from time import sleep
ap = argparse.ArgumentParser(description="finds the traffic light")
ap.add_argument("-f", nargs=1, dest='file',
                help="Name of the video")

args = ap.parse_args()


def update_yellow_detector(frame):
    # Our operations on the frame come here

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    black_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([140,50,120]))
    res_black = cv2.bitwise_and(frame, frame, mask=black_mask)

    cv2.imshow('res', black_mask)

    lower_yellow = np.array([5, 50, 50])
    upper_yellow = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res_yellow = cv2.bitwise_and(frame, frame, mask=mask)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))

    mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

    res_red = cv2.bitwise_and(frame, frame, mask=mask)

    img = cv2.medianBlur(res_yellow, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('res', img)

    im2, contours, hierarchy = cv2.findContours(cimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boundary = [(500, 40), (800, 330)]

    cv2.rectangle(frame, boundary[0], boundary[1], (255, 0, 0), 2)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if x > boundary[0][0] and x + w < boundary[1][0] and y > boundary[0][1] and y + h < boundary[1][1]:
            area = cv2.contourArea(c)
            aspect_ratio = float(w) / h
            #print(area, aspect_ratio)
            cv2.drawContours(frame, [c], 0, (0, 255, 0), 3)
            if 15 < area < 50 and .75 < aspect_ratio < 1.25:
                approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
                #print("approx", len(approx))
                if (len(approx) < 5):
                    print("found", len(approx))
                    #sleep(1)
                    cv2.drawContours(frame, [c], 0, (0, 255, 0), 3)
                    sleep(10)

    cv2.imshow('detected circles', frame)

    # sleep(.05)


def update_traffic_detector(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()

            update_yellow_detector(frame)

            # Display the resulting frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)


update_traffic_detector(args.file[0])
