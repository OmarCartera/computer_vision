import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from skimage import feature
import time


def houghCircle(image):
    Ny, Nx = image.shape
    R_max = np.round(np.sqrt(Ny**2 + Nx**2))

    R = 13

    R_interval = np.arange(1, R_max)
    thetas = np.deg2rad(np.arange(360))

    # computing sin and cosine
    cosine = np.cos(thetas)
    sine   = np.sin(thetas)

    accumulator = np.zeros((Ny, Nx, R))

    # Now Start Accumulation
    for col in range(Nx):
        for row in range(Ny):
            if (image[row][col]):
                for i in range(360):
                    for r in range(1, R):
                        a = int(round(row - r * cosine[i]))
                        b = int(round(col - r * sine[i]))

                        try:
                            accumulator[a, b, r] += 1

                        except IndexError:
                            pass

    return accumulator, Nx, Ny


def detectCircles(image, accumulator, threshold, Nx, Ny):
    detectedCircles = np.where(accumulator >= threshold * accumulator.max())

    ys = detectedCircles[0]
    xs = detectedCircles[1]
    rs = detectedCircles[2]


    plotCircles(image, ys, xs, rs, Nx, Ny)
        

def plotCircles(image, ys, xs, rs, Nx, Ny):
    fig = plt.figure('Circles')

    plt.imshow(image)

    circle = []

    for i in range(len(ys)):
        circle.append(plt.Circle((xs[i], ys[i]), rs[i], color=(1,0,0), fill=True))
        fig.add_subplot(111).add_artist(circle[-1])

    plt.xlim(0, Nx)
    plt.ylim(Ny, 0)

    plt.savefig('output_houghCircle.png')


    
if __name__ == '__main__':
    ###################
    start_time = time.time()

    # Load the image
    image = plt.imread('coins.jpg')

    try:
        # Get value Channel (intensity)
        hsvImage = color.rgb_to_hsv(image)
        valImage = hsvImage[...,2]

    except ValueError:
        valImage = image

    # Edge detection (canny)
    edgeImage = feature.canny(valImage,sigma=1.4, low_threshold=40, high_threshold=150)    
    
    # Show original image
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    
    # Show edge image
    plt.figure('Edge Image')
    plt.imshow(edgeImage)
    plt.set_cmap('gray')
    
    # build accumulator    
    accumulator, Nx, Ny = houghCircle(edgeImage)

    # Detect and superimpose lines on original image
    detectCircles(image, accumulator, 0.5, Nx, Ny)
    

    print("--- %0.3f seconds ---" % (time.time() - start_time))
    ####################

    plt.show()