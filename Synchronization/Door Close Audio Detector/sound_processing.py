import numpy as np
import sys
import scipy.io.wavfile as wav

"""
To run:

python sound_processing.py </path/to/internal/audio> </path/to/external/audio> <time> <frac>
(recommended values of time and frac are 120 and 0.666667)

Time represents the number of seconds of the video clip being taken. So 120 represents that the
first 120 seconds of the clip (first 2 minutes) are taken. 'Frac' refers to the fraction of peak
volume below which all sound should be disregarded as noise. With a value of 'frac' as 0.66667,
any sound with a volume below 2/3rds of the peak sound is eliminated by the z_filter function.

"""

# reads the .wav file and converts to an array
def read_file(fname):
    fn = wav.read(fname)
    sampling_rate = fn[0]
    data = fn[1]
    left = []
    for frame in data:
        left.append(frame[0])
    return left, sampling_rate

# converts seconds to units of sampling frequency
def s(time, sampling_rate):
    return int(sampling_rate * time)
    # sampling_rate is 44100 Hz for the files we are using

# converts units of sampling frequency to seconds
def t(index, sampling_rate):
    return index / sampling_rate
    # sampling_rate is 44100 Hz for the files we are using

# a 'zero' filter that eliminates values less than frac * peak value
def z_filter(x, frac):
    f = float(frac)
    peak = float(max(x))
    to_return = []
    for element in x:
        relative = element/peak
        if relative - f >= 0:
            to_return.append(relative)
        else:
            to_return.append(0)
    return to_return

# a 'range' filter that eliminates values except those between indices 'start' and 'end'
def r_filter(x, start, end):
    to_return, counter = [], 0
    while counter < len(x):
        if counter > start and counter < end:
            to_return.append(x[counter])
        else:
            to_return.append(0)
        counter += 1
    return to_return

# a moving average function
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    to_ret = ret[n - 1:] / n
    return np.concatenate((np.array([0] * (n - 1)), np.array(to_ret)), axis=0)

# a function that finds the index of the final peak
def find_last_peak(audio):
    reverse, index = audio[::-1], 0
    while index < len(reverse):
        if reverse[index] > 0:
            break
        index += 1
    return len(audio) - (index + 1)

def return_peak(path, time, frac):
    sound, sr = read_file(path)
    sound_copy, car_start = sound[s(0, sr):s(time, sr)], False
    sz_filtered = z_filter(sound_copy, float(frac))
    sound_data, index = moving_average(sz_filtered, sr), 0 

    while index < len(sound_copy):
        if car_start:
            sound_copy[index] = 0
        elif sound_data[index] >= 0.02:
            """
            assumptions made: there is at least a second between the door closing 
            and the car beginning to move. this zero-ing process is to eliminate 
            the increasing volume to the peak that wouldn't be registered within
            this window of half a second.
            """
            for i in range(int(sr / 2)):
                sound_copy[index - i] = 0
            car_start = True
        index += 1

    final_sound = z_filter(sound_copy, float(frac))

    # The last peak on the final graph represents the door closing.
    return t(float(find_last_peak(final_sound)), sr)

def execute():
    try:
        # input arguments
        path_to_internal_audio = sys.argv[1]
        path_to_external_audio = sys.argv[2]
        time_v, frac_v = float(sys.argv[3]), float(sys.argv[4])
    except Exception:
        print("Sorry! You need path to internal audio, external audio path, time and then frac.")
        print("Recommended values for time and frac are 120 and 0.6667.")
        return [None, None]

    inside_peak = return_peak(path_to_internal_audio, time_v, frac_v)
    outside_peak = return_peak(path_to_external_audio, time_v, frac_v)
    difference = inside_peak - outside_peak

    print("The last peak of external audio is at " + str(outside_peak) + " seconds.")
    print("The last peak of internal audio is at " + str(inside_peak) + " seconds.\n")
    print("The difference is " + str(difference) + " seconds.")

    return [inside_peak, outside_peak, difference]

execute()





