import cv2

states = ['DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'DAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA', 'NAA']

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

video=cv2.VideoWriter('video.avi',-1,10.0,(width,height))


for s in states:
	video.write(images[s])

video.release()