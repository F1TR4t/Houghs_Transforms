# Our Imports
import matplotlib as plt
import numpy as np
import scipy.signal

from PIL import Image

# Method Area [For creating methods]

def hough_transform():

    # ------------------ Using Parameter Space (m, c) --------------------------

    # Quantize a Parameter Space, Map Input -> Output
    # Requires data (Our Image?, a set of points from every edge?)


    # Loop through the parameter space
    # Note, we have edge points (x0, y0) -> (xn, yn)
    # Use formula c = (-xi)m + yi
    # P(c, m) should be incremented


    # Search for Local Maximas Throughout our Parameter Space
    # Probably take that point, and use to turn into a line in the image
    # Like, y = (mi)x + ci, for x belonging in our Image Space

    # ------------------- Better Version that doesn't involve Infinity ----------------

    # Instead of (m, c), use p = x * cos(theta) + y * sin(theta)
    # Theta is from the Gradient, so now
    # Quantize to a parameter Space of A[theta_min, theta_max][p_min, p_max]
    # Choose Increments for Theta + p such that Noise isn't an issue and we don't miss any info

    # Loop through the parameter space
    # Note, we have edge points (x0, y0) -> (xn, yn)
    # Use formula p = x * cos(theta) + y * sin(theta)
    # A(theta, p) should be incremented


    # Search for Local Maximas Throughout our Parameter Space

    return None

def load_image(filepath):
    # Loads an image into a numpy array.
    # Note: image will have 3 color channels [r, g, b].
    img = Image.open(filepath)
    return (np.asarray(img).astype(np.float)/255)[:, :,:3]

# Global Area [Just the area with we'll run our Code]

i_x = [[1, 0, -1],[1, 0, -1],[1, 0, 1]]
i_y = [[1, 1, 1],[0, 0, 0],[-1, -1, -1]]

# Grab our Test Images and convert into Numpy Arrays
lines1 = load_image("test_images\lines1.png")

# Take derivatives of the image to extract the Image Gradient
lines_i_x = scipy.signal.convolve2d(lines1[:][:][0], i_x, mode='same')
#lines1_i_x = scp.convolve2d(lines1, i_y)

# Pass down necesssary arguments to hough_transform()
# grab what it returns 

# Draw Lines then add that layer ontop of the original image
fig = plt.figure(figsize=(3, 3), dpi=300)
plt.imshow(lines_i_x, cmap='gray')