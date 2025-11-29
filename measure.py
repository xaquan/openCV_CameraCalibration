import math
import re
import pandas as pd
import cv2
import numpy as np
import os
import random
from datetime import datetime


#Default grid size 200mm x 200mm
WIDTH_MM = 200
HEIGHT_MM = 200

COLS = 21
ROWS = 21
PHOTO_FILE = "photos\chessboard_1cm\IMG_0811.JPEG"

CELL_WIDTH = 10  # pixels


POLYNOMIAL_DEGREE = 9

COEFF_X = [
        5.03522E-30, -2.08032E-25, 1.66169E-21, -6.15523E-18, 1.29511E-14, -1.66958E-11, 1.34937E-08, -6.74901E-06, 0.001970621, -0.188274592, 2.405780108
]

COEFF_Y = [1.34567E-29, -3.44717E-25, 2.50912E-21, -8.94711E-18, 1.8471E-14, -2.35801E-11, 1.89715E-08, -9.48017E-06, 0.002779543, -0.311524821, 8.775590104
    ]

OUTLINE_COLOR = (0, 0, 255)
POINT_COLOR = (0, 255, 0)

LIMIT_POINTS = 99

FILENAMES_1CM = []
FILENAMES_2CM = []

#print version of cv2
print("Opencv version:", cv2.__version__)

frame = cv2.imread(PHOTO_FILE)

samples = []
samples_reference = []
poly_coefficients = None
sample_filename = ""

points = []
circles = []

mapping_points = [] #[[173, 167], [1865, 179], [1871, 1858], [177, 1860]]  # Initialize with 4 points
corner_points = [[0, 0], [1, 0], [1, 1], [0, 1]] # Normalized coordinates

grid = None

is_mapped = False

# Function convert x in pixel to x mm

def get_pos_mm_from_px(px_x, px_y):
    x = px_x
    y = px_y

    n_x = len(COEFF_X) - 1
    n_y = len(COEFF_Y) - 1

    x_mm = sum(coef * (x ** (n_x - i)) for i, coef in enumerate(COEFF_X))    

    y_mm = sum(coef * (y ** (n_y - i)) for i, coef in enumerate(COEFF_Y))

    return (x_mm, y_mm)

#print x and y in mm from pixel using the 2d polynomial
def print_position_in_mm_from_px(px_x, px_y):
    point_mm = get_pos_mm_from_px(px_x, px_y)

    point_mm2 = get_polynomial_value((px_x, px_y))

    print(f"(Checking) Position in mm ({point_mm[0]}, {point_mm[1]}) from px: ({px_x}, {px_y})")
    print(f"(Polynomial) Position in mm ({point_mm2[0]}, {point_mm2[1]}) from px: ({px_x}, {px_y})")
    return (point_mm[0], point_mm[1])

#Function draw a point when left mouse button is clicked
def mouse_click_action(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        #if there is LIMIT_POINTS points, pop the first one and remove the circle
        if len(points) == LIMIT_POINTS:
            points.pop(0)
        point = (x, y)
        points.append(point)
        draw_point(point, POINT_COLOR)
        # save_point_to_csv(x, y)

def draw_point(point, color, mm_point=None, frame=frame):
    cv2.circle(frame, point, 2, color, -1)

    # Apply the inverse perspective transform to get the position in mm
    if mm_point is None:
        mm_point = convert_point_px_to_mm(point)

    cv2.putText(frame, f"mm: {mm_point[0]:.2f}, {mm_point[1]:.2f}", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# Function saving a point append to the cvs file with parameter x and y
def save_point_to_csv(px_x, px_y, x_mm=None, y_mm=None):
    # Generate a single random filename per session (so multiple saves go to the same file)
    if not hasattr(save_point_to_csv, "filename"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_point_to_csv.filename = f"points_{timestamp}.csv"
    filename = save_point_to_csv.filename

    if x_mm is None or y_mm is None:
        point_mm = convert_point_px_to_mm((px_x, px_y))
        x_mm = round(point_mm[0], 2)
        y_mm = round(point_mm[1], 2)

    real_point_mm = get_conner_real_position_in_mm((x_mm, y_mm))
    real_point_center_mm = get_center_real_position_in_mm((x_mm, y_mm))


    x_mm_center = x_mm - WIDTH_MM/2
    y_mm_center = y_mm - HEIGHT_MM/2

    print_position_in_mm_from_px(px_x, px_y)

    real_x_center = real_point_center_mm[0]
    real_y_center = real_point_center_mm[1]


    # Create the file if it does not exist
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("px_x,px_y, x, y, x_mm,y_mm,x_center, y_center, x_mm_center,y_mm_center\n")
    with open(filename, "a") as f:
        f.write(f"{px_x},{px_y},{real_point_mm[0]},{real_point_mm[1]},{x_mm},{y_mm},{real_x_center},{real_y_center},{x_mm_center},{y_mm_center}\n")
        

#Distance betweens 2 points in pixels
def pixels_between(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

#Mapping 4 points as the rectangle of 160mm x 220mm, use 4 center of circles except the circle at the middle
def map_points(points):
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to map a rectangle.")
    
    print("Mapping points:")
    print(points)
    # Define the source points (the points clicked by the user)
    src_pts = np.array(points, dtype="float32")
    # Define the destination points (the corners of the output rectangle)
    dst_pts = np.array([[0, 0], [WIDTH_MM, 0], [0, HEIGHT_MM], [WIDTH_MM, HEIGHT_MM]], dtype="float32")
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    return M

# Function convert a point in pixel to mm using the grid mapping
def convert_point_px_to_mm(point_px):
    point_homogeneous = np.array([point_px[0], point_px[1], 1], dtype="float32")
    mm_point = grid @ point_homogeneous
    mm_point /= mm_point[2]  # Normalize
    return mm_point[:2]

def get_conner_real_position_in_mm(point):

    x_mm = int(round(point[0]/CELL_WIDTH)) * CELL_WIDTH
    x_mm = x_mm if x_mm <= WIDTH_MM else WIDTH_MM

    y_mm = int(round(point[1]/CELL_WIDTH)) * CELL_WIDTH
    y_mm = y_mm if y_mm <= HEIGHT_MM else HEIGHT_MM
    
    # print(f"Real Position in mm: ({x_mm}, {y_mm})")
    return (x_mm, y_mm)

def get_center_real_position_in_mm(point):
    x_mm = int(math.floor((point[0] - WIDTH_MM/2)/CELL_WIDTH)) * CELL_WIDTH
    x_mm = x_mm if x_mm <= WIDTH_MM/2 else WIDTH_MM/2

    y_mm = int(math.floor((point[1] - HEIGHT_MM/2)/CELL_WIDTH)) * CELL_WIDTH
    y_mm = y_mm if y_mm <= HEIGHT_MM/2 else HEIGHT_MM/2

    # print(f"Real Centered Position in mm: ({x_mm - WIDTH_MM/2}, {y_mm - HEIGHT_MM/2})")
    return (x_mm, y_mm)

def detect_chessboard(img):
    # convert the input image to a grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    panel_size = (COLS, ROWS)

    global samples
    global samples_reference
    global sample_filename

    # Find the chess board corners
    ret, samples = cv2.findChessboardCorners(gray, panel_size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ADAPTIVE_THRESH)
    print(f"Chessboard corners found: {ret}")

    # if chessboard corners are detected
    if ret == True:
        
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, panel_size, corners,ret)
        # cv2.imshow('Chessboard',img)
        
        #save corners to csv file, add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_filename = f"chessboard_corners_{timestamp}.csv"
        with open(sample_filename, "w") as f:
            header = "x, y, x_pixel, y_pixel\n"
            f.write(header)
            for i, corner in enumerate(samples):
                x = (i % COLS) * CELL_WIDTH
                y = (i // COLS) * CELL_WIDTH

                samples_reference.append((x, y))

                # Add 4 corners position to mapping points
                if i in [0, COLS -1, COLS * (ROWS -1), COLS * ROWS -1]:
                    mapping_points.append((round(corner[0][0]), round(corner[0][1])))
                                        
                    draw_point((round(corner[0][0]), round(corner[0][1])), OUTLINE_COLOR, (x, y))
                    
                    if len(mapping_points) == 4:
                        global grid
                        grid = map_points(mapping_points)

                f.write(f"{x}, {y}, {corner[0][0]}, {corner[0][1]}\n")
    return ret

def gnerate_regression_polynomial():
    global samples
    global samples_reference

    x = np.array([samples_reference[i][0] for i in range(len(samples_reference))])
    y = np.array([samples_reference[i][1] for i in range(len(samples_reference))])

    x_px = np.array([corner[0][0] for corner in samples])
    y_px = np.array([corner[0][1] for corner in samples])

    print(f"x={x[0]}, x_px={x_px[0]}")

    coeff_x = np.polyfit(x_px, x, POLYNOMIAL_DEGREE)
    coeff_y = np.polyfit(y_px, y, POLYNOMIAL_DEGREE)
    print(coeff_x)  # coefficients a6 ... a0
    print(coeff_y)

    return (coeff_x, coeff_y)

def get_polynomial_value(point_px):
    global poly_coefficients
    x = point_px[0]
    y = point_px[1]
    coeff_x, coeff_y = poly_coefficients
    x_mm = sum(coef * (x ** (len(coeff_x) - 1 - i)) for i, coef in enumerate(coeff_x))
    y_mm = sum(coef * (y ** (len(coeff_y) - 1 - i)) for i, coef in enumerate(coeff_y))
    return (x_mm, y_mm)

def convert_samples_point_to_mm():
    global samples
    global sample_filename
        
    # x_px = np.array([corner[0][0] for corner in samples])
    # y_px = np.array([corner[0][1] for corner in samples])

    list_x_mm = []
    list_y_mm = []

    for i in range(len(samples)):
        point_px = (samples[i][0][0], samples[i][0][1])
        point_mm = get_polynomial_value(point_px)

        x_mm = point_mm[0]
        y_mm = point_mm[1]
        # print(f"Sample point {i}: px({x_px[i]}, {y_px[i]}) -> mm({x_mm}, {y_mm})")
        list_x_mm.append(x_mm)
        list_y_mm.append(y_mm)

    df = pd.read_csv(sample_filename)

    # Make sure the list length matches the number of rows
    df['x_mm'] = list_x_mm
    df['y_mm'] = list_y_mm

    df.to_csv(sample_filename, index=False)  


is_chessboard_detected = detect_chessboard(frame)

if is_chessboard_detected:
    poly_coefficients = gnerate_regression_polynomial()
    convert_samples_point_to_mm()

while is_chessboard_detected and grid is not None:
    cv2.imshow("Frame", frame)
    # cv2.setMouseCallback("Frame", draw_point)    

    cv2.setMouseCallback("Frame", mouse_click_action)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break


# cap.release()
cv2.destroyAllWindows()