# Our Imports
# Starter code (run this first)
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage.filters as filters
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

    for x in range(2, img.shape[0]-2):
        for y in range(2, img.shape[1]-2):
            if img[y, x] > 0.55:
                out_x.append(x)
                out_y.append(y)
                out.append([x, y])
    return out_x, out_y, out

def get_local_maxima(data, threshold, do_return_values=False):
    # See: https://stackoverflow.com/a/9113227/3672986
    neighborhood_size = 10

    data_region_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_region_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    maxima[data < threshold] = 0

    labeled, num_objects = scipy.ndimage.label(maxima)
    slices = scipy.ndimage.find_objects(labeled)
    x, y, r = [], [], []
    for dy, dx in slices:
        x_center = int(round((dx.start + dx.stop - 1)/2))
        x.append(x_center)
        y_center = int(round((dy.start + dy.stop - 1)/2))   
        y.append(y_center)
        r.append(data[y_center, x_center])
        
    if do_return_values:
        return x, y, r
    else:
        return x, y

def hough_transform(edges):

    # ------------------ Using Parameter Space (m, c) --------------------------

    # Quantize a Parameter Space, Map Input -> Output
    # Requires data (Our Image?, a set of points from every edge?)
    # Instead of (m, c), use p = x * cos(theta) + y * sin(theta)
    # Quantize to a parameter Space of A[theta_min, theta_max][p_min, p_max]
    # Choose Increments for Theta + p such that Noise isn't an issue and we don't miss any info

    #accumulator = np.zeros((301, 181)) # Theta goes from 0 -> 180, p will have a max of 300 because of our image sizes
    accumulator = np.zeros((601, 181))

    # 0 -> 299 represents -300 -> -1
    # 300 represents 0
    # 301 -> 600 represents 1 -> 300

    # 0 -> 179 represents - 180 -> -1
    # 180 represents 0
    # 181 -> 360 represents 1 -> 180

    # Loop through the parameter space
    # Note, we have edge points (x0, y0) -> (xn, yn)
    # Use formula p = x * cos(theta) + y * sin(theta)
    # A(theta, p) should be incremented
    for i in range(len(edges)): # I believe the for loop works and designs our sinusoids
        for th in range(181):
            x = (edges[i])[0]
            y = (edges[i])[1]
            p = round(x * math.cos(math.pi*(th)/180) + y * math.sin(math.pi*(th)/180))
            # Since we're keeping it from theta 0 -> 180 degrees, and p 0 -> 600, must keep it inside our window
            if ( p >= -300 and p <= 300 ):
                accumulator[p+300][th] = accumulator[p+300][th] + 1

    return accumulator

def convert_to_image(th, p):

    x = []
    y = []

    for i in range(len(th)):
        th_i = th[i]
        p_i = p[i]
        
        if ( th_i == 0 or th_i == 180):
            if (p_i < 0):
                p_i = p_i * -1
            for ii in range(300):
                y.append(ii)
                x.append(p_i)
            continue

        m = math.cos(math.pi * (th_i/180)) / math.sin(math.pi * (th_i/180))
        b = p_i / math.sin(math.pi * (th_i/180))

        for ii in range(300):
            j = round((m * ii) + b)
            if ( j >= 0 and j < 300 ):
                y.append(j)
                x.append(ii)

    return x, y

# Global Area [Just the area with we'll run our Code]
name = input("Which image would you like to load: ")

# Grab our Test Images
#lines1 = Image.open("test_images/lines1.png")
box = Image.open("test_images/" + name + ".png")

# Gray Scale it
#l1_edge = lines1.convert("L")
box_edge = box.convert("L")

# Our Edge detection
#l1_edge = l1_edge.filter(ImageFilter.FIND_EDGES)
#l1_ed_mat = np.asarray(l1_edge).astype(np.float64)/255

box_edge = box_edge.filter(ImageFilter.FIND_EDGES)
box_ed_mat = np.asarray(box_edge).astype(np.float64)/255

fig = plt.figure(figsize=(3, 3), dpi=300)
plt.axis('off')
plt.imshow(box_edge, cmap='gray')
plt.savefig("out_images/" + name + "_edge.png", bbox_inches="tight")

# Locate all edges within the Image, Edges should be white
edges_x, edges_y, edges = extract_edges(box_ed_mat)

# Pass down necesssary arguments to hough_transform()
# grab what it returns 
output = hough_transform(edges)

# Grab the local maximas coords of our accumulator
trhd = int(input("Please enter the threshold: "))
x, y = get_local_maxima(output, threshold=trhd)

fig = plt.figure(figsize=(3, 3), dpi=300)
plt.axis('off')
plt.imshow(output)
plt.plot(x, y, 'o', markersize=0.5, color='firebrick', fillstyle="none")
plt.savefig("out_images/" + name + "_sin.png", bbox_inches="tight")

for i in range(len(y)):
    y[i] = y[i] - 300

xi, yi = convert_to_image(x, y)
# Take our points and convert them into our image space as lines

# Draw Lines then add that layer ontop of the original image, save into out_images/
fig = plt.figure(figsize=(3, 3), dpi=300)
plt.axis('off')
plt.imshow(box)
plt.plot(xi, yi, 'o', markersize=0.5, linewidth=1, color='firebrick', fillstyle="none")
plt.savefig("out_images/" + name + "_ht.png", bbox_inches="tight")