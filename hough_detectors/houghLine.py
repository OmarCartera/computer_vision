import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
import matplotlib.colors as color
import time


def houghLine(image, Nx, Ny):
    # Max diatance is diagonal one 
    maxdist = int(np.round((Nx*Nx + Ny*Ny)**0.5)) 

    # Range of radius, a line of 2*maxdist samples
    rs = np.linspace(-maxdist, maxdist, 2 * maxdist)

    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90, 1))

    
    # computing sin and cosine
    cosine = np.cos(thetas)
    sine   = np.sin(thetas)


    # initialize accumulator array to zeros
    accumulator = np.zeros((2 * maxdist, len(thetas)))

    # Now Start Accumulation
    for col in range(Nx):
        for row in range(Ny):
            if (image[row][col]):
                for k in range(len(thetas)):
                    r = col * cosine[k] + row * sine[k]
                    accumulator[int(r) + maxdist, k] += 1

    return accumulator, thetas, rs


def detectLines(image, accumulator, threshold, rhos, thetas, Ny, Nx):
    detectedLines = np.where(accumulator >= threshold * accumulator.max())

    detectedRadius = rhos[detectedLines[0]]
    detectedThetas = thetas[detectedLines[1]]

    # Now plot detected lines in image
    plotLines(image, detectedRadius, detectedThetas, Ny, Nx)


def plotLines(image, detectedRadius, detectedThetas, Ny, Nx):
    plt.figure('Lines')
    plt.imshow(image)

    x = np.linspace(0, Nx)

    cosine = np.cos(detectedThetas)
    sine = np.sin(detectedThetas)

    cotan = cosine/sine
    ratio = detectedRadius/sine

    for i in range(len(detectedRadius)):
        # if thete is not 0
        if (detectedThetas[i]):
            plt.plot(x, (-x * cotan[i]) + ratio[i])

        # if theta is 0
        else:
            plt.axvline(detectedRadius[i])

    plt.xlim(0, Nx)
    plt.ylim(Ny, 0)


    plt.savefig('output_houghLine.png')

   


if __name__ == '__main__':
    ###################
    start_time = time.time()

    # Load the image
    image = plt.imread('Lines.jpg')

    Ny, Nx, _ = image.shape

    # Get value Channel (intensity)
    hsvImage = color.rgb_to_hsv(image)
    valImage = hsvImage[...,2]
    
    # Detect edges using canny 
    edgeImage = feature.canny(valImage, sigma = 1.4, low_threshold = 40, high_threshold = 150)

    # Show original image
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    
    # Show edge image
    plt.figure('Edge Image')
    plt.imshow(edgeImage)
    plt.set_cmap('gray')
        
    # build accumulator    
    accumulator, thetas, rhos = houghLine(edgeImage, Nx, Ny)
    


    # Visualize hough space
    plt.figure('Hough Space', figsize = (5,5))
    plt.imshow(accumulator)
    plt.set_cmap('gray')
    
    # Detect and superimpose lines on original image
    detectLines(image, accumulator, 0.3, rhos, thetas, Ny, Nx)
    
    print("--- %0.3f seconds ---" % (time.time() - start_time))
    ####################

    plt.show()