import cv2
import argparse
import numpy as np
from time import  sleep
ap = argparse.ArgumentParser(description="finds the traffic light")
ap.add_argument("-f", nargs=1, dest='file',
                help="Name of the video")

args = ap.parse_args()

cap = cv2.VideoCapture(args.file[0])

while(True):
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()


        # Our operations on the frame come here

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        res_yellow = cv2.bitwise_and(frame, frame, mask=mask)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
        mask2 = cv2.inRange(hsv, np.array([160,100,100]), np.array([179,255,255]))

        mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

        res_red = cv2.bitwise_and(frame, frame, mask=mask)


        img = cv2.medianBlur(res_yellow, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

        cv2.imshow('res', cimg)

        im2, contours, hierarchy = cv2.findContours(cimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        boundry = [(400,40), (900,300)]

        cv2.rectangle(frame, boundry[0], boundry[1], (255, 0, 0), 2)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if x > boundry[0][0] and x + w < boundry[1][0] and y > boundry[0][1] and y+h < boundry[1][1]:
                area = cv2.contourArea(c)
                aspect_ratio = float(w)/h
                print(aspect_ratio)
                if area > 10 and area < 50 and aspect_ratio > .9 and aspect_ratio < 1.25:
                    cv2.drawContours(frame, [c], 0, (0, 255, 0), 3)
                    sleep(1)

        cv2.imshow('detected circles', frame)

        sleep(.03)

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)
