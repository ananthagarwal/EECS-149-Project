#!/usr/bin/env bash

echo "Input rosbag name:"
read rosbag_name

#Declare camera feeds to export
declare camType="/image_raw/compressed"
declare vidType=".mp4"

#Declare default settings for sound processing
declare timeLength=120
declare peakThresh=0.6666667

##Export 8 camera feeds to mp4s
rm *.bag-cam*.mp4
for cam in `seq 4 5`
do
	echo $rosbag_name"-cam"$cam$vidType
	python2.7 rosbag2video.py --fps 10 -o $rosbag_name"-cam"$cam$vidType -t "/usb_cam"$cam$camType $rosbag_name 1> output.txt
	epoch=$(grep "Epoch time:" output.txt)
	echo $epoch
	epoch=${epoch##*:}
	echo $epoch
	#python2.7 rosbag2video.py --fps 10 -o $rosbag_name"-cam"$cam$vidType -t "/usb_cam"$cam$camType $rosbag_name
done

##Use left (cam4) and right (cam5) camera video feeds to determine door time
echo $rosbag_name$"-cam4"$vidType
python2.7 door_close_detector.py -v $rosbag_name"-cam4"$vidType --door "left" 1> left.txt
left_door_time=$(grep "ERROR:" left.txt)
if [$left_door_time = ""]; then
	left_door_time=`cat left.txt`
fi
python2.7 door_close_detector.py -v $rosbag_name"-cam5"$vidType --door "right" 1> right.txt
right_door_time=$(grep "ERROR:" right.txt)
if [$right_door_time = ""]; then
	right_door_time=`cat right.txt`
fi
echo $left_door_time
echo $right_door_time
last_door_time=$(( left_door_time > right_door_time ? left_door_time : right_door_time ))
echo $max
#Use audio detector to determine audio time
echo "Input inside mic wav file:"
read inside_mic
echo "Input outside mic wav file:"
read outside_mic

python2.7 sound_processing.py $inside_mic $timeLength $peakThresh 1> inside.txt
python2.7 sound_processing.py $outside_mic $timeLength $peakThresh 1> outside.txt

inside_mic_time=$(grep "ERROR:" inside.txt)
outside_mic_time=$(grep "ERROR:" outside.txt)
if [$inside_mic_time = ""]; then
	inside_mic_time=`cat inside.txt`
fi
if [$outside_mic_time = ""]; then
	outside_mic_time=`cat outside.txt`
fi

ros_time=$((epoch + last_door_time))
echo "Final Results:"
echo $"ROS timestamp of "ros_time", inside mic time of "$inside_mic_time" s, outside mic time of "$outside_mic_time" s, are the same."

if [ "$right_door_time" == "$last_door_time" ]; then
	echo $ros_time" corresponds to "$((right_door_time/10**9))" s on the right camera, cam5."
fi

if [ "$left_door_time" == "$last_door_time" ]; then
	echo $ros_time" corresponds to "$((left_door_time/10**9))" s on the left camera, cam4."
fi
