from sensor_types import *
import cv2
import os
import numpy as np
from time import sleep

LEFT_TURN_THRESHOLD = 60
RIGHT_TURN_THRESHOLD = -60
SPEED_UP_TIME = 3
MASTER_CSV_INTERVAL = 0.1
sg_queue = []
sg_count = 0
traffic_run_queue = []
avg_throttle = 0
avg_brake = 0


def update_dot_count(frame_index):
    
    return


def update_car_count(front, back):
    face_cascade = cv2.CascadeClassifier('cars.xml')
    f_gray = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

    f_faces = face_cascade.detectMultiScale(f_gray, 1.3, 2)
    b_faces = face_cascade.detectMultiScale(b_gray, 1.3, 2)

    return len(f_faces) + len(b_faces)


def update_ads(frame_index):
    frame = frames[frame_index]

    return


def update_body_posture(frame_index):
    """
    -1 is left, 0 is center, 1 is right
    """
    return


def update_stop_go_count(frame_index):
    global sg_count, sg_queue
    last_state = sg_queue[-1][0]
    frame = frames[frame_index]
    throttle = frame.accelerator_pedal.throttle_rate
    brake = frame.brake_torq.brake_torque_request
    timestamp = frame.body_pressure.epoch

    if last_state == "ACCEL" and brake > 1:
        sg_count += 1
        sg_queue.append(("BRAKE", timestamp))
    elif last_state == "BRAKE" and throttle > 1:
        sg_count += 1
        sg_queue.append(("ACCEL", timestamp))
    elif not last_state:
        last_state = "ACCEL" if throttle > 1 else "BRAKE"
        sg_count += 1
        sg_queue.append((last_state, timestamp))

    while sg_queue:
        _, time = sg_queue[0]
        if timestamp - time >= 60:
            sg_queue.pop(0)
            sg_count -= 1
        else:
            break


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


def update_yellow_detect(video_frame):
    # Our operations on the frame come here

    hsv = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)

    black_mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([140,50,120]))
    res_black = cv2.bitwise_and(video_frame, video_frame, mask=black_mask)

    cv2.imshow('res', black_mask)

    lower_yellow = np.array([5, 50, 50])
    upper_yellow = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res_yellow = cv2.bitwise_and(video_frame, video_frame, mask=mask)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))

    mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

    res_red = cv2.bitwise_and(video_frame, video_frame, mask=mask)

    img = cv2.medianBlur(res_yellow, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('res', img)

    im2, contours, hierarchy = cv2.findContours(cimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boundary = [(500, 40), (800, 330)]

    cv2.rectangle(video_frame, boundary[0], boundary[1], (255, 0, 0), 2)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if x > boundary[0][0] and x + w < boundary[1][0] and y > boundary[0][1] and y + h < boundary[1][1]:
            area = cv2.contourArea(c)
            aspect_ratio = float(w) / h
            #print(area, aspect_ratio)
            cv2.drawContours(video_frame, [c], 0, (0, 255, 0), 3)
            if 15 < area < 50 and .75 < aspect_ratio < 1.25:
                approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
                #print("approx", len(approx))
                if (len(approx) < 5):
                    print("found", len(approx))
                    #sleep(1)
                    cv2.drawContours(video_frame, [c], 0, (0, 255, 0), 3)
                    sleep(10)

    cv2.imshow('detected circles', video_frame)

    # sleep(.05)

def update_traffic_light_behavior(frame, video_frame):
    global avg_brake, avg_throttle, throttle_queue, brake_queue
    """
    0 - Cautious
    1 - Aggressive
    """
    if not update_yellow_detect(video_frame):
        return 0

    throttle = frame.accelerator_pedal.throttle_rate
    brake = frame.brake_torq.brake_torque_request
    throttle_queue.append(throttle)
    brake_queue.append(brake)
    avg_throttle = sum(throttle_queue[1:])/(len(throttle_queue)-1) if len(throttle_queue) > 1 else 0
    avg_brake = sum(brake_queue[1:])/(len(brake_queue)-1) if len(brake_queue) > 1 else 0
    # Once we have enough frames to determine behavior
    if len(throttle_queue) >= SPEED_UP_TIME / MASTER_CSV_INTERVAL:
        # Default behavior: Slow down upon seeing the yellow light. Expected, cautious.
        if avg_brake > brake_queue[0] and avg_throttle <= throttle_queue[0]:
            brake_queue = []
            throttle_queue = []
            return 0
        # If we speed up after seeing the yellow light, reckless
        elif avg_throttle > throttle_queue[0] and avg_brake <= brake_queue[0]:
            brake_queue = []
            throttle_queue = []
            return 1
        else:
            brake_queue = []
            throttle_queue = []
            return 1
    return 0
        


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


def update_traffic_light_type(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
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

            boundary = [(400, 40), (900, 300)]

            cv2.rectangle(frame, boundary[0], boundary[1], (255, 0, 0), 2)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)

                if x > boundary[0][0] and x + w < boundary[1][0] and y > boundary[0][1] and y+h < boundary[1][1]:
                    area = cv2.contourArea(c)
                    aspect_ratio = float(w)/h
                    print(aspect_ratio)
                    if 10 < area < 50 and .9 < aspect_ratio < 1.25:
                        cv2.drawContours(frame, [c], 0, (0, 255, 0), 3)
                        sleep(1)

            cv2.imshow('detected circles', frame)

            sleep(.03)

            # Display the resulting frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)


frames = Dataset.loader("MASTER.p")

videos = [cv2.VideoCapture("cam1.mp4"), cv2.VideoCapture("cam2.mp4"), cv2.VideoCapture("cam3.mp4"), cv2.VideoCapture("cam4.mp4"),
          cv2.VideoCapture("cam5.mp4"), cv2.VideoCapture("cam6.mp4"), cv2.VideoCapture("cam7.mp4"), cv2.VideoCapture("cam8.mp4")]

def video_update():
    v_frames = [v.read() for v in videos]



"""
    avg_throttle = sum([frames[frame_index + idx].accelerator_pedal.throttle_rate for idx in
                        range(SPEED_UP_TIME / MASTER_CSV_INTERVAL)]) // (SPEED_UP_TIME / MASTER_CSV_INTERVAL)
    avg_brake = sum([frames[frame_index + idx].brake_torq.brake_torque_request for idx in
                     range(SPEED_UP_TIME / MASTER_CSV_INTERVAL)]) // (SPEED_UP_TIME / MASTER_CSV_INTERVAL)

"""
