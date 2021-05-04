# Our Imports
# Starter code (run this first)
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal
from PIL import Image, ImageFilter

# Method Area [For creating methods]

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
    out = []

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[y, x] > 0.55:
                out_x.append(x)
                out_y.append(y)
                out.append([x, y])
    return out_x, out_y, out

def hough_transform(edges, theta):

    # ------------------ Using Parameter Space (m, c) --------------------------

    # Quantize a Parameter Space, Map Input -> Output
    # Requires data (Our Image?, a set of points from every edge?)

    accumulator = np.zeros((601, 19)) # Theta goes from 0 -> 180, p will have a max of 599 because of our image sizes

    # Instead of (m, c), use p = x * cos(theta) + y * sin(theta)
    # Theta is from the Gradient, so now
    # Quantize to a parameter Space of A[theta_min, theta_max][p_min, p_max]
    # Choose Increments for Theta + p such that Noise isn't an issue and we don't miss any info
    for i in range(len(edges)): # I believe the for loop works and designs our sinusoids
        for th in range(19):
            x = (edges[i])[0]
            y = (edges[i])[1]
            p = round(x * math.cos(math.pi*(th*10)/180) + y * math.sin(math.pi*(th*10)/180))
            # Since we're keeping it from theta 0 -> 180 degrees, and p 0 -> 600, must keep it inside our window
            if ( p > 0 and p <= 600 ):
                accumulator[th][p] = accumulator[th][p] + 1
            


    # Loop through the parameter space
    # Note, we have edge points (x0, y0) -> (xn, yn)
    # Use formula p = x * cos(theta) + y * sin(theta)
    # A(theta, p) should be incremented


    # Search for Local Maximas Throughout our Parameter Space

    return accumulator

# Global Area [Just the area with we'll run our Code]
s_x = [[1, 0, -1],[2, 0, -2],[1, 0, -1]]
s_y = [[1, 2, 1,],[0, 0, 0],[-1, -2, -1]]

# Grab our Test Images
lines1 = Image.open("test_images/lines1.png")

# Gray Scale it
l1_edge = lines1.convert("L")

# Create our gradient
l1_x = scipy.signal.convolve2d(l1_edge, s_x)
l1_y = scipy.signal.convolve2d(l1_edge, s_y)
l1_grad = l1_x + l1_y

# Our Edge detection
l1_edge = l1_edge.filter(ImageFilter.FIND_EDGES)
l1_ed_mat = np.asarray(l1_edge).astype(np.float64)/255

# Locate all edges within the Image, Edges should be white
edges_x, edges_y, edges = extract_edges(l1_ed_mat)

# Pass down necesssary arguments to hough_transform()
# grab what it returns 
output = hough_transform(edges, l1_grad)


# Draw Lines then add that layer ontop of the original image, save into out_images/
fig = plt.figure(figsize=(3, 3), dpi=300)
plt.axis('off')
plt.imshow(output, cmap='gray')
plt.savefig("out_images/lines1_output", bbox_inches="tight")
#plt.plot(edges_x, edges_y, 'o', markersize=0.1, color='firebrick', fillstyle="none")
#plt.savefig("out_images/lines1_edge", bbox_inches="tight")