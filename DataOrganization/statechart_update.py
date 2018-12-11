from sensor_types import *
import cv2
import os
import numpy as np
from time import sleep

from sismic.io import import_from_yaml, export_to_plantuml
from sismic.interpreter import Interpreter

frames = Dataset.loader("MASTER.p")

class StateMachine():

    LEFT_TURN_THRESHOLD = 60
    RIGHT_TURN_THRESHOLD = -60
    SPEED_UP_TIME = 3
    MASTER_CSV_INTERVAL = 0.1
    sg_queue = []
    sg_count = 0
    traffic_run_queue = []
    throttle_queue = []
    brake_queue = []
    dot_queue = []
    avg_throttle = 0
    avg_brake = 0

    def video_update(self):
        v_frames = [v.read() for v in videos]
        return v_frames

    def update_yellow_detect(self, video_frame):
        # Our operations on the frame come here

        hsv = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)

        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([140, 50, 120]))
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
                # print(area, aspect_ratio)
                #cv2.drawContours(video_frame, [c], 0, (0, 255, 0), 3)
                if 15 < area < 50 and .75 < aspect_ratio < 1.25:
                    approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
                    # print("approx", len(approx))
                    if (len(approx) < 5):
                        return 1
                        #print("found", len(approx))
                        # sleep(1)
                        #cv2.drawContours(video_frame, [c], 0, (0, 255, 0), 3)
                        #sleep(10)
        return 0
        #cv2.imshow('detected circles', video_frame)


    def update_dot_count(self, frame_index):
        frame = frames[frame_index]
        dot_count = frame.lidar_frame.close
        self.dot_queue.append(dot_count)
        if len(self.dot_queue) > 50:
            self.dot_queue.pop(0)
        return sum(self.dot_queue) / len(self.dot_queue)

    def update_car_count(self, front, back):
        face_cascade = cv2.CascadeClassifier('cars.xml')
        f_gray = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

        f_faces = face_cascade.detectMultiScale(f_gray, 1.3, 2)
        b_faces = face_cascade.detectMultiScale(b_gray, 1.3, 2)

        return len(f_faces) + len(b_faces)

    def convex_hull_pointing_up(self, ch):
        points_above_center, points_below_center = [], []
        x, y, w, h = cv2.boundingRect(ch)
        aspect_ratio = w / h

        if aspect_ratio < 0.8:
            vertical_center = y + h / 2

            for point in ch:
                if point[0][1] < vertical_center:
                    points_above_center.append(point)
                elif point[0][1] >= vertical_center:
                    points_below_center.append(point)

            left_x = points_below_center[0][0][0]
            right_x = points_below_center[0][0][0]
            for point in points_below_center:
                if point[0][0] < left_x:
                    left_x = point[0][0]
                if point[0][0] > right_x:
                    right_x = point[0][0]

            for point in points_above_center:
                if (point[0][0] < left_x) or (point[0][0] > right_x):
                    return False
        else:
            return False

        return True

    def num_traffic_cones(self, video_frame):
        img_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        img_thresh_low = cv2.inRange(img_HSV, np.array([0, 135, 135]), np.array([15, 255, 255]))
        img_thresh_high = cv2.inRange(img_HSV, np.array([159, 135, 135]), np.array([179, 255, 255]))
        img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)

        kernel = np.ones((5, 5))
        img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
        img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)

        img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

        _, contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_contours = np.zeros_like(img_edges)
        cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 2)

        approx_contours = []

        for c in contours:
            approx = cv2.approxPolyDP(c, 10, closed=True)
            approx_contours.append(approx)

        img_approx_contours = np.zeros_like(img_edges)
        cv2.drawContours(img_approx_contours, approx_contours, -1, (255, 255, 255), 1)

        all_convex_hulls = []
        for ac in approx_contours:
            all_convex_hulls.append(cv2.convexHull(ac))

        img_all_convex_hulls = np.zeros_like(img_edges)
        cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255, 255, 255), 2)

        convex_hulls_3to10 = []
        for ch in all_convex_hulls:
            if 3 <= len(ch) <= 10:
                convex_hulls_3to10.append(cv2.convexHull(ch))

        img_convex_hulls_3to10 = np.zeros_like(img_edges)
        cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255, 255, 255), 2)

        cones = []
        bounding_rects = []
        for ch in convex_hulls_3to10:
            if self.convex_hull_pointing_up(ch):
                cones.append(ch)
                rect = cv2.boundingRect(ch)
                bounding_rects.append(rect)
        return len(cones)

    def update_ads(self, video_frame):
        return self.num_traffic_cones(video_frame) >= 1


    def update_body_posture(self, frame_index):
        """
        -1 is left, 0 is center, 1 is right
        """
        return 0

    def update_stop_go_count(self, frame_index):
        # global sg_count, sg_queue
        last_state = self.sg_queue[-1][0] if self.sg_queue else None
        frame = frames[frame_index]
        throttle = frame.accelerator_pedal.throttle_rate
        brake = frame.brake_torq.brake_torque_request
        timestamp = frame.body_pressure.epoch/10**9
        # print(throttle, brake)
        if last_state == "ACCEL" and brake > 1:
            self.sg_count += 1
            self.sg_queue.append(("BRAKE", timestamp))
        elif last_state == "BRAKE" and throttle > 1:
            self.sg_count += 1
            self.sg_queue.append(("ACCEL", timestamp))
        elif not last_state:
            last_state = "ACCEL" if throttle > 1 else "BRAKE"
            self.sg_count += 1
            self.sg_queue.append((last_state, timestamp))

        while self.sg_queue:
            _, time = self.sg_queue[0]

            if timestamp - time >= 60:
                self.sg_queue.pop(0)
                self.sg_count -= 1
            else:
                break
        return self.sg_count

    def is_right_turn(self, frame_index):
        frame = frames[frame_index]
        left_wheel_speed = frame.vehicle_wheel_speeds.front_left
        right_wheel_speed = frame.vehicle_wheel_speeds.front_right
        steering_angle = frame.steering_ang.steering_wheel_angle
        return steering_angle < self.RIGHT_TURN_THRESHOLD and (left_wheel_speed - right_wheel_speed) >= 1.0

    def is_left_turn(self, frame_index):
        frame = frames[frame_index]
        left_wheel_speed = frame.vehicle_wheel_speeds.front_left
        right_wheel_speed = frame.vehicle_wheel_speeds.front_right
        steering_angle = frame.steering_ang.steering_wheel_angle
        return steering_angle > self.LEFT_TURN_THRESHOLD and (right_wheel_speed - left_wheel_speed) >= 1.0

    def update_traffic_light_behavior(self, frame_index, video_frame):
        # global avg_brake, avg_throttle, throttle_queue, brake_queue
        """
        0 - Cautious
        1 - Aggressive
        """
        frame = frames[frame_index]
        if not self.update_yellow_detect(video_frame):
            return 0

        throttle = frame.accelerator_pedal.throttle_rate
        brake = frame.brake_torq.brake_torque_request
        self.throttle_queue.append(throttle)
        self.brake_queue.append(brake)
        avg_throttle = sum(self.throttle_queue[1:])/(len(self.throttle_queue)-1) if len(self.throttle_queue) > 1 else 0
        avg_brake = sum(self.brake_queue[1:])/(len(self.brake_queue)-1) if len(self.brake_queue) > 1 else 0
        # Once we have enough frames to determine behavior
        if len(self.throttle_queue) >= self.SPEED_UP_TIME / self.MASTER_CSV_INTERVAL:
            # Default behavior: Slow down upon seeing the yellow light. Expected, cautious.
            if avg_brake > self.brake_queue[0] and avg_throttle <= self.throttle_queue[0]:
                self.brake_queue = []
                self.throttle_queue = []
                return 0
            # If we speed up after seeing the yellow light, reckless
            elif avg_throttle > self.throttle_queue[0] and avg_brake <= self.brake_queue[0]:
                self.brake_queue = []
                self.throttle_queue = []
                return 1
            else:
                self.brake_queue = []
                self.throttle_queue = []
                return 1
        return 0


    def list_files(self, path):
        # returns a list of names (with extension, without full path) of all files
        # in folder path
        files = []
        for name in os.listdir(path):
            print(name)
            if name.endswith('.mp4'):
                files.append(path+'/'+name)
        return files

    def video_joiner(self, folder_path):
        names = self.list_files(folder_path)

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


videos = [cv2.VideoCapture("cam1.mp4"), cv2.VideoCapture("cam2.mp4"), cv2.VideoCapture("cam3.mp4"), cv2.VideoCapture("cam4.mp4"),
          cv2.VideoCapture("cam5.mp4"), cv2.VideoCapture("cam6.mp4"), cv2.VideoCapture("cam7.mp4"), cv2.VideoCapture("cam8.mp4")]


with open('statechart.yml') as f:
    statechart = import_from_yaml(f)
    s = StateMachine()
interpreter = Interpreter(statechart, initial_context={'s': s, 'vframes': s.video_update()})
print(export_to_plantuml(statechart))

print(len(frames))

for i in range(len(frames)):
    interpreter.execute_once()

    print(interpreter.configuration)
    print(i)
    print(interpreter.context['avg_car_count'])
    print(interpreter.context['sg_count'])

"""
    avg_throttle = sum([frames[frame_index + idx].accelerator_pedal.throttle_rate for idx in
                        range(SPEED_UP_TIME / MASTER_CSV_INTERVAL)]) // (SPEED_UP_TIME / MASTER_CSV_INTERVAL)
    avg_brake = sum([frames[frame_index + idx].brake_torq.brake_torque_request for idx in
                     range(SPEED_UP_TIME / MASTER_CSV_INTERVAL)]) // (SPEED_UP_TIME / MASTER_CSV_INTERVAL)

"""
