import argparse
import imutils
import cv2

#code adapted from 
#
#	Video
#	https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# 	
#	Shape Detection
#	https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/


class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)

		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-d", "--door", required=True, help="which door")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["image"])
frame_num = 0

last_triangle = 0


if args["door"] == "right":

	cmin = 0
	cmax = 300
	ymin = 600

elif args["door"] == "left":

	cmin = 900
	cmax = 1200
	ymin = 600


while(cap.isOpened()):
	try:
		frame_num += 1
		# Capture frame-by-frame
		ret, image = cap.read()
		if image is None:
			break

		resized = imutils.resize(image, width=image.shape[0])
		ratio = image.shape[0] / float(resized.shape[0])

		# convert the resized image to grayscale, blur it slightly,
		# and threshold it
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

		

		# find contours in the thresholded image and initialize the
		# shape detector
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		sd = ShapeDetector()




		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


		# loop over the contours
		for c in cnts:
			# compute the center of the contour, then detect the name of the
			# shape using only the contour
			try:
				M = cv2.moments(c)
				cX = int((M["m10"] / M["m00"]) * ratio)
				cY = int((M["m01"] / M["m00"]) * ratio)

				shape = sd.detect(c)

				# multiply the contour (x, y)-coordinates by the resize ratio,
				# then draw the contours and the name of the shape on the image
				c = c.astype("float")
				c *= ratio
				c = c.astype("int")
				
				if shape == "triangle" and cX > 900 and cX < 1200 and cY > 600 and cv2.contourArea(c) > 250:
					cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
					cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
					last_triangle = cap.get(0)
					# show the output image
			except Exception as e:
				#print(e)
				pass
		cv2.imshow('frame',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	except:
		pass

# When everything done, release the capture
print(last_triangle)

cap.release()
cv2.destroyAllWindows()