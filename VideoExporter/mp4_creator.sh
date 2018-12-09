#!/usr/bin/env bash


#Declare camera feeds to export
declare camType="/image_raw/compressed"
declare vidType=".mp4"

#Declare default settings for sound processing
declare timeLength=120
declare peakThresh=0.6666667

##Export 8 camera feeds to mp4s


for rosbag_name in *.bag
do
	rm $rosbag_name"-cam"*".mp4"
	for cam in `seq 1 8`
	do
		echo $rosbag_name"-cam"$cam$vidType
		python2.7 rosbag2video.py --fps 10 -o $rosbag_name"-cam"$cam$vidType -t "/usb_cam"$cam$camType $rosbag_name 1> output.txt
		#python2.7 rosbag2video.py --fps 10 -o $rosbag_name"-cam"$cam$vidType -t "/usb_cam"$cam$camType $rosbag_name
	done
done
