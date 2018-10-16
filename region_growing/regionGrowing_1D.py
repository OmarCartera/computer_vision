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



def check_neighbours(edited_y):
	global points, value

	try:
		# up
		if (np.abs((grayImage[edited_y - Nx] - value)) < threshold) and (grayImage[edited_y - Nx]):
			grayImage[edited_y - Nx] = 0
			points.append(edited_y - Nx)

		# down
		if (np.abs((grayImage[edited_y + Nx] - value)) < threshold) and (grayImage[edited_y + Nx]):
			grayImage[edited_y + Nx] = 0
			points.append(edited_y + Nx)

		# right
		if (np.abs((grayImage[edited_y + 1] - value)) < threshold) and (grayImage[edited_y + 1]):
			grayImage[edited_y + 1] = 0
			points.append(edited_y + 1)

		# left
		if (np.abs((grayImage[edited_y - 1] - value)) < threshold) and (grayImage[edited_y - 1]):
			grayImage[edited_y - 1] = 0
			points.append(edited_y - 1)

	except:
		pass



def show_final():
	global start, grayImage
	elapsed_time = "%0.5f Seconds" % (time.time() - start)
	plt.title('Final Image ... 1D')
	plt.text(5, -12, elapsed_time, fontsize=10, bbox=dict(facecolor='yellow', alpha=1))

	final_image = grayImage.reshape((Ny, Nx))

	plt.imshow(final_image)
	plt.savefig('output_regionGrowing.png')
	plt.show()


def onclick(event):
	global start, value, grayImage

	start = time.time()

	y, x = int(event.ydata), int(event.xdata)

	points.append(y*Nx + x)

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
	try:
		image = plt.imread(file + '.jpg')

	except:
		image = plt.imread(file + '.jpeg')


Ny, Nx, _ = image.shape

try:
	hsvImage = color.rgb_to_hsv(image)
	valImage = hsvImage[...,2]

except ValueError:
	valImage = image


grayImage = np.round(rgb2gray(image))

grayImage = grayImage.reshape((1, Ny*Nx))[0]

fig = plt.figure()

fig.add_subplot(111)
plt.title('Input Image')

plt.imshow(image)
plt.set_cmap('gray')
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()