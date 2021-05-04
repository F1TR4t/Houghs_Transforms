# Our Imports
# Starter code (run this first)
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal
from PIL import Image, ImageFilter

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
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = Image.open(filepath)
    return (np.asarray(img).astype(np.float64)/255)[:, :, :3]

def load_image_gray(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = Image.open(filepath)
    img = img.convert("L")
    img.save("out_images/grey_lines1.png")
    img = img.filter(ImageFilter.FIND_EDGES)
    img.save("out_images/edge_lines1.png")
    return (np.asarray(img).astype(np.float64)/255)[:, :]

# Go through our Inputted Image Matrix and extract all edges (white locations)
def extract_edges(img):
    out_x = []
    out_y = []

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[y, x] == 1:
                out_x.append(x)
                out_y.append(y)
    return out_x, out_y

# Global Area [Just the area with we'll run our Code]

# Grab our Test Images, and get out Edge Detected Image
lines1 = Image.open("test_images/lines1.png")
l1_edge = lines1.convert("L")
l1_edge = l1_edge.filter(ImageFilter.FIND_EDGES)
l1_ed_mat = np.asarray(l1_edge).astype(np.float64)/255

# Locate all edges within the Image, Edges should be white
edges_x, edges_y = extract_edges(l1_ed_mat)
# Pass down necesssary arguments to hough_transform()
# grab what it returns 



# Draw Lines then add that layer ontop of the original image, save into out_images/
fig = plt.figure(figsize=(3, 3), dpi=300)
plt.axis('off')
plt.imshow(lines1, cmap='gray')
plt.savefig("out_images/lines1", bbox_inches="tight")
plt.plot(edges_x, edges_y, 'o', markersize=0.1, color='firebrick', fillstyle="none")
plt.savefig("out_images/lines1_edge", bbox_inches="tight")