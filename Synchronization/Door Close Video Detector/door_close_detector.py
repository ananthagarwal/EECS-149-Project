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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video", "--video", required=True,
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

		# convert the resized image to grayscale, blur it slightly,
		# and threshold it
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
		
		# find contours in the thresholded image and initialize the
		# shape detector
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]

		# loop over the contours
		for c in cnts:
			# compute the center of the contour, then detect the name of the
			# shape using only the contour
			try:
				M = cv2.moments(c)
				cX = int((M["m10"] / M["m00"]))
				cY = int((M["m01"] / M["m00"]))

				peri = cv2.arcLength(c, True)
				approx = cv2.approxPolyDP(c, 0.04 * peri, True)

				# multiply the contour (x, y)-coordinates by the resize ratio,
				# then draw the contours and the name of the shape on the image
				c = c.astype("int")
				
				if len(approx) == 3 and cX > cmin and cX < cmax and cY > ymin and cv2.contourArea(c) > 250:
					cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
					cv2.putText(image, 'triangle', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
					last_triangle = cap.get(0)
					# show the output image
			except Exception as e:
				#print(e)
				pass
		#cv2.imshow('frame',image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	except Exception as e:
		#print(e)
		pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print(last_triangle)
