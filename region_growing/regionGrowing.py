#Implement region growing segmentation. 
#Allow user to set an initial seed and then segment
#this region according to similarity of colors and or intensity.


import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as color
import time 



def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def growRegion(y, x):
	global points

	value = int(grayImage[y][x])

	k = 0
	prev = 0

	while len(points) > 0:
		if prev == len(points):
			k += 1

		prev = len(points)
		for i in range(len(points)):
			try:
				checkNeighbours(points[i][0], points[i][1], value)

				grayImage[(points[i][0], points[i][1])] = 0
				points.remove(points[i])

			except IndexError:
				pass

		if k == 2:
			points = []
			break


	final(grayImage)


def checkNeighbours(y, x, value):
	global points

	if (np.abs((grayImage[y-1][x] - value)) < threshold) and (grayImage[y-1][x]):
		grayImage[y-1][x] = 0
		points.append((y-1, x))		



	if (np.abs((grayImage[y][x-1] - value)) < threshold) and (grayImage[y][x-1]):
		grayImage[y][x-1] = 0
		points.append((y, x-1))


	if (np.abs((grayImage[y][x+1] - value)) < threshold) and (grayImage[y][x+1]):
		grayImage[y][x+1] = 0
		points.append((y, x+1))


	if (np.abs((grayImage[y+1][x] - value)) < threshold) and (grayImage[y+1][x]):
		grayImage[y+1][x] = 0
		points.append((y+1, x))




def final(gray):
	global start
	print("\n~~~ %0.2f Seconds ~~~ " % (time.time() - start))
	plt.title('Final Image')
	plt.imshow(gray)
	plt.savefig('output_regionGrowing.png')
	plt.show()


def onclick(event):
	global start
	start = time.time()

	x, y = int(event.xdata), int(event.ydata)

	points.append((y, x))

	growRegion(y, x)



start = 0

points    = []
threshold = 10

file = 'seg1'

fig = plt.figure()

try:
	image = plt.imread(file + '.png')

except IOError:
	image = plt.imread(file + '.jpg')


Ny, Nx, _ = image.shape

try:
	hsvImage = color.rgb_to_hsv(image)
	valImage = hsvImage[...,2]

except ValueError:
	valImage = image


grayImage = np.round(rgb2gray(image))



fig.add_subplot(111)
plt.title('Input Image')

plt.imshow(image)
plt.set_cmap('gray')
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()