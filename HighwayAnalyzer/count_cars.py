import cv2
import numpy as np

path = 'cars.jpg'

'''img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img_gray,-1,kernel)

ret,thresh = cv2.threshold(dst ,100
	,255,cv2.THRESH_BINARY)

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,255,0), 3)

cv2.imshow('img',img)
cv2.waitKey(0)'''


import numpy as np
import cv2 as cv


def count_cars(front, back):
	face_cascade = cv.CascadeClassifier('cars.xml')
	f_gray = cv.cvtColor(front, cv.COLOR_BGR2GRAY)
	b_gray = cv.cvtColor(fback, cv.COLOR_BGR2GRAY)

	f_faces = face_cascade.detectMultiScale(f_gray, 1.3, 2)
	b_faces = face_cascade.detectMultiScale(b_gray, 1.3, 2)

	return len(f_faces) + len(b_faces)
