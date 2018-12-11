import cv2
import pickle

states = pickle.load(open('fsm_state.p', 'rb'))


d = ['D','N']
a = ['A','C']
e = ['A','C','N','O']

names = []

for i in d:
	for j in a:
		for k in e:
			names.append(i + j + k)

images = {n: cv2.imread('fsm/'+n+'.png') for n in names}

height,width,layers=images['DAA'].shape
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

video=cv2.VideoWriter('video.avi',fourcc,10.0,(width,height))


for s in states:
	video.write(images[s])

video.release()
