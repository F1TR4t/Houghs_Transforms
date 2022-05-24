# Computer Science 482 - Computer Vision
A school project where a group took what they learned from the class and attempted to implement various Hough transforms, a shape detector. Only line detection was implementing.

This project was implementing by going through research papers and conceptual videos on Hough transforms.

# Requirements
Python 3.9.1 or higher
Packages inside requirements.txt

# Setup
Ensure you have Python installed, and it is recommended to either use anaconda or venv to create a virtual environment.

Regardless, install packages by running "pip install -r requirements.txt" to install all required packages.

# Running locally
In the terminal, run either of these commands "python hough.py" or "python3 hough.py"

You will be prompted for a file name. Make sure said file is a png and saved inside the test_images/ directory. Enter only the name. For example, for test_images/box.png, I would type box. The program can handle the long names + extensions, so to make life easier: KEEP PNG FILES INSIDE test_images/

You will be prompted again for a threshold value. This will depend on your image. Type a positive integer.

Once the program finishes running, your output files will be inside out_images/. You should get FILE_out.png and FILE_sin.png (where FILE is your input file's name used previously). The former will provide to you the original input, the edge detection performed onto it, then the resulting lines found. The latter shows a neat sinusoids and its peaks.
