import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as color
import time 


def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



def grow_region():
	global points

	while len(points) > 0:
		check_neighbours(points.pop())
		
	show_final()



def check_neighbours(point):
	global points, value

	y = point[0]
	x = point[1]

	try:
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

	except:
		pass



def show_final():
	global start
	elapsed_time = "%0.5f Seconds" % (time.time() - start)
	plt.title('Final Image ... 2D')
	plt.text(5, -12, elapsed_time, fontsize=10, bbox=dict(facecolor='yellow', alpha=1))

	plt.imshow(grayImage)
	plt.savefig('output_regionGrowing.png')
	plt.show()



def onclick(event):
	global start, value

	start = time.time()

	points.append((int(event.ydata), int(event.xdata)))

	value = grayImage[points[0]]

	grow_region()



start = 0

value = 0

points    = []
threshold = 10

file = 'seg3'

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

fig = plt.figure()

fig.add_subplot(111)
plt.title('Input Image')

plt.imshow(image)
plt.set_cmap('gray')
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()