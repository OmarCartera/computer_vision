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
	global points, value, image

	y = point[0]
	x = point[1]

	try:
		if (np.abs((grayImage[y-1][x] - value)) < threshold) and (grayImage[y-1][x]):
			grayImage[y-1][x] = 0
			image[y-1, x][3] = 0
			points.append((y-1, x))		


		if (np.abs((grayImage[y+1][x] - value)) < threshold) and (grayImage[y+1][x]):
			grayImage[y+1][x] = 0
			image[y+1, x][3] = 0
			points.append((y+1, x))


		if (np.abs((grayImage[y][x-1] - value)) < threshold) and (grayImage[y][x-1]):
			grayImage[y][x-1] = 0
			image[y, x-1][3] = 0
			points.append((y, x-1))


		if (np.abs((grayImage[y][x+1] - value)) < threshold) and (grayImage[y][x+1]):
			grayImage[y][x+1] = 0
			image[y, x+1][3] = 0
			points.append((y, x+1))

	except IndexError as e:
		# print e
		pass



def show_final():
	global start, image, background

	elapsed_time = "%0.5f Seconds" % (time.time() - start)
	plt.title('          Magic Tool')
	plt.text(5, -12, elapsed_time, fontsize=10, bbox=dict(facecolor='yellow', alpha=1))

	plt.imshow(background)
	plt.imshow(image)
	plt.savefig('output_regionGrowing.png')
	plt.show()



def onclick(event):
	global start, value, image, grayImage

	start = time.time()

	if(event.button == 1):
		image_stack.append(image.copy())
		gray_stack.append(grayImage.copy())

		points.append((int(event.ydata), int(event.xdata)))

		# print image[int(event.ydata), int(event.xdata)]

		value = grayImage[points[0]]

		grow_region()

	elif(event.button == 3):
		try:
			image = image_stack.pop().copy()
			grayImage = gray_stack.pop().copy()

			show_final()

		except IndexError:
			print 'empty stacks :)'



# def onkey(event):
# 	print('you pressed', event.key, event.xdata, event.ydata)


image_stack = []
gray_stack = []

start = 0

value = 0

points    = []
threshold = 10

file = 'conan1'

try:
	image = plt.imread(file + '.png')

except IOError as e:
	# print e
	image = plt.imread(file + '.jpg')

background = plt.imread('base.png')

Ny, Nx, _ = image.shape

grayImage = np.round(rgb2gray(image))

fig = plt.figure()

fig.add_subplot(111)
plt.title('Input Image')

plt.imshow(background)
plt.imshow(image)
plt.set_cmap('gray')

cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
# cid2 = fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()