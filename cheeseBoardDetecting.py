import math
import re
import cv2
import numpy as np
import os
import random
from datetime import datetime


#Default grid size 200mm x 200mm
WIDTH_MM = 180
HEIGHT_MM = 180

SAMPLE_INCREASEMENT = 10  # pixels

OUTLINE_COLOR = (0, 0, 255)
POINT_COLOR = (0, 255, 0)

LIMIT_POINTS = 99

PHOTO_FILE = "photos\DSCF3016.jpg"
#print version of cv2
print("Opencv version:", cv2.__version__)



points = []
circles = []

mapping_points = [] #[[173, 167], [1865, 179], [1871, 1858], [177, 1860]]  # Initialize with 4 points
corner_points = [[0, 0], [1, 0], [1, 1], [0, 1]] # Normalized coordinates

grid = None

is_mapped = False

# Function convert x in pixel to x mm
def get_x_in_mm(x):
    # -1E-06x2 + 0.1628x - 39.033
    # return 2e-12*x**6 - 7e-9*x**5 + 3e-6*x**4 - 0.0005*x**3 + 0.0339*x**2 + 0.3737*x + 0.6167
    # return -2e-12*x**6 - 6e-9*x**5 - 5e-8*x**4 + 9e-5*x**3 + 0.0002*x**2 + 0.7302*x + 1.0377
    return  -4e-12*x**6 - 1e-9*x**5 + 6e-8*x**4 + 2e-6*x**3 - 0.0001*x**2 + 1.1166*x - 1.4974

def get_x_in_mm_from_px(px_x, px_y):
    x = px_x
    # return 3.38764e-24*x**8 - 4.70163e-20*x**7 + 2.32388e-16*x**6 - 5.77896e-13*x**5 + 8.12383e-10*x**4 - 6.55728e-07*x**3 + 0.0002782*x**2 + 0.0531651*x - 10.78725247
    return -1.56886e-25*x**9 + 1.43764e-21*x**8 - 5.60114e-18*x**7 + 1.21003e-14*x**6 - 1.58555e-11*x**5 + 1.29586e-08*x**4 - 6.53472e-06*x**3 + 0.001919707*x**2 - 0.181896763*x + 2.090817372

def get_y_in_mm_from_px(px_x, px_y):
    y = px_y
    # return 1.0462e-23*y**8 - 1.09379e-19*y**7 + 4.5822e-16*y**6 - 1.01344e-12*y**5 + 1.29803e-09*y**4 - 9.74801e-07*y**3 + 0.000397155*y**2 + 0.034221332*y - 10.66767497
    return -2.08196e-25*y**9 + 1.91136e-21*y**8 - 7.46785e-18*y**7 + 1.61952e-14*y**6 - 2.13239e-11*y**5 + 1.75257e-08*y**4 - 8.89544e-06*y**3 + 0.002638562*y**2 - 0.293480449*y + 7.856661846


def get_pos_mm_from_px(ponts):
    x_coefs = [
        -1.56886e-25,
        1.43764e-21,
        5.60114e-18,
        1.21003e-14,
        1.58555e-11,
        1.29586e-08,
        6.53472e-06,
        0.001919707,
        0.181896763,
        2.090817372
    ]

    y_coefs = [
        -2.08196e-25,
        1.91136e-21,
        -7.46785e-18,
        1.61952e-14,
        -2.13239e-11,
        1.75257e-08,
        -8.89544e-06,
        0.002638562,
        -0.293480449,
        7.856661846
    ]

    x = sum(coef * (point[0] ** (9 - i)) for i, coef in enumerate(x_coefs))
    y = sum(coef * (point[1] ** (9 - i)) for i, coef in enumerate(y_coefs))

    return (x, y)

# Function convert y in pixel to y mm
def get_y_in_mm(y):
    # -2E-06x2 + 0.1323x - 5.6461
    # return 1e-12*y**6 - 7e-9*y**5 + 3e-6*y**4 - 0.0005*y**3 + 0.0335*y**2 + 0.3647*y + 0.5962
    # return -1e-12*y**6 - 6e-9*y**5 - 3e-8*y**4 + 8e-5*y**3 + 0.0002*y**2 + 0.7347*y + 0.6523
    return -2e-12*y**6 - 1e-9*y**5 + 4e-8*y**4 + 2e-6*y**3 - 6e-5*y**2 + 1.1156*y - 0.9085

# Print x and y in mm
def print_position_in_mm(point):
    x_mm = get_x_in_mm(point[0])
    y_mm = get_y_in_mm(point[1])
    print(f"(Checking) Position in mm: ({x_mm}, {y_mm})")
    return (x_mm, y_mm)

#print x and y in mm from pixel using the 2d polynomial
def print_position_in_mm_from_px(point):
    point_mm = get_pos_mm_from_px(point)
    x_mm = point_mm[0]
    y_mm = point_mm[1]

    print(f"(Checking) Position in mm ({x_mm}, {y_mm}) from px: ({point[0]}, {point[1]})")
    return (x_mm, y_mm)

#Function draw a point when left mouse button is clicked
def draw_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        #if there is LIMIT_POINTS points, pop the first one and remove the circle
        if len(points) == LIMIT_POINTS:
            points.pop(0)
        point = (x, y)
        points.append(point)
        save_point_to_csv(x, y)


# Function saving a point append to the cvs file with parameter x and y
def save_point_to_csv(px_x, px_y, x_mm=None, y_mm=None):
    # Generate a single random filename per session (so multiple saves go to the same file)
    if not hasattr(save_point_to_csv, "filename"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_point_to_csv.filename = f"points_{timestamp}.csv"
    filename = save_point_to_csv.filename

    if x_mm is None or y_mm is None:
        point_mm = get_position_in_mm((px_x, px_y))
        x_mm = round(point_mm[0], 2)
        y_mm = round(point_mm[1], 2)

    real_point_mm = get_conner_real_position_in_mm((x_mm, y_mm))
    real_point_center_mm = get_center_real_position_in_mm((x_mm, y_mm))


    x_mm_center = x_mm - WIDTH_MM/2
    y_mm_center = y_mm - HEIGHT_MM/2


    print_position_in_mm((x_mm_center, y_mm_center))
    print_position_in_mm_from_px((px_x, px_y))

    real_x_center = real_point_center_mm[0]
    real_y_center = real_point_center_mm[1]


    # Create the file if it does not exist
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("px_x,px_y, x, y, x_mm,y_mm,x_center, y_center, x_mm_center,y_mm_center\n")
    with open(filename, "a") as f:
        f.write(f"{px_x},{px_y},{real_point_mm[0]},{real_point_mm[1]},{x_mm},{y_mm},{real_x_center},{real_y_center},{x_mm_center},{y_mm_center}\n")
        
    #Print saved point related to conner
    print(f"Saved point to {filename}: px({px_x}, {px_y})")
    print(f"mm: ({x_mm}, {y_mm}), real mm: ({real_point_mm[0]},{real_point_mm[1]})")
    print(f"Centered mm: ({real_x_center},{real_y_center}),({x_mm_center},{y_mm_center})")


def set_mapping_points(event, x, y, flags, param):
    global mapping_points
    i = len(mapping_points)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mapping point set at: ({x}, {y})")
        save_point_to_csv(x, y, corner_points[i][0] * WIDTH_MM, corner_points[i][1] * HEIGHT_MM)
        mapping_points.append((x, y))
        if len(mapping_points) > 4:
            mapping_points.pop(0)
        

#Distance betweens 2 points in pixels
def pixels_between(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

#Mapping 4 points as the rectangle of 160mm x 220mm, use 4 center of circles except the circle at the middle
def map_points(points):
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to map a rectangle.")
    # Define the source points (the points clicked by the user)
    src_pts = np.array(points, dtype="float32")
    # Define the destination points (the corners of the output rectangle)
    dst_pts = np.array([[0, 0], [WIDTH_MM, 0], [WIDTH_MM, HEIGHT_MM], [0, HEIGHT_MM]], dtype="float32")
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M

def get_position_in_mm(point_px):
    point_homogeneous = np.array([point_px[0], point_px[1], 1], dtype="float32")
    mm_point = grid @ point_homogeneous
    mm_point /= mm_point[2]  # Normalize
    return mm_point[:2]

def get_conner_real_position_in_mm(point):

    x_mm = int(round(point[0]/SAMPLE_INCREASEMENT)) * SAMPLE_INCREASEMENT
    x_mm = x_mm if x_mm <= WIDTH_MM else WIDTH_MM

    y_mm = int(round(point[1]/SAMPLE_INCREASEMENT)) * SAMPLE_INCREASEMENT
    y_mm = y_mm if y_mm <= HEIGHT_MM else HEIGHT_MM
    
    print(f"Real Position in mm: ({x_mm}, {y_mm})")
    return (x_mm, y_mm)

def get_center_real_position_in_mm(point):
    x_mm = int(math.floor((point[0] - WIDTH_MM/2)/SAMPLE_INCREASEMENT)) * SAMPLE_INCREASEMENT
    x_mm = x_mm if x_mm <= WIDTH_MM/2 else WIDTH_MM/2

    y_mm = int(math.floor((point[1] - HEIGHT_MM/2)/SAMPLE_INCREASEMENT)) * SAMPLE_INCREASEMENT
    y_mm = y_mm if y_mm <= HEIGHT_MM/2 else HEIGHT_MM/2

    print(f"Real Centered Position in mm: ({x_mm - WIDTH_MM/2}, {y_mm - HEIGHT_MM/2})")
    return (x_mm, y_mm)

def detect_chessboard(img):
    # convert the input image to a grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nCols = 19
    nRows = 19
    square_size = 10  # in mm
    panel_size = (nCols, nRows)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, panel_size, flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ADAPTIVE_THRESH)
    print(f"Chessboard corners found: {ret}")

    # if chessboard corners are detected
    if ret == True:
        
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, panel_size, corners,ret)
        # cv2.imshow('Chessboard',img)
        
        #save corners to csv file, add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chessboard_corners_{timestamp}.csv"
        with open(filename, "w") as f:
            header = "x, y, x_pixel, y_pixel\n"
            f.write(header)
            for i, corner in enumerate(corners):
                x = (i % nCols) * square_size
                y = (i // nCols) * square_size

                # Add 4 corners position to mapping points
                if i in [0, nCols -1, nCols * (nRows -1), nCols * nRows -1]:
                    mapping_points.append((round(corner[0][0]), round(corner[0][1])))
                px_point = (round(corner[0][0]), round(corner[0][1]))
                f.write(f"{x}, {y}, {corner[0][0]}, {corner[0][1]}\n")
    return ret


def check_error(point_px, expected_x_mm, expected_y_mm):
    point_mm = get_position_in_mm(point_px)
    error_x = point_mm[0] - expected_x_mm
    error_y = point_mm[1] - expected_y_mm
    print(f"Position error for point {point_px}: ({error_x:.2f} mm, {error_y:.2f} mm)")
    return (error_x, error_y)


if mapping_points is not None and len(mapping_points) == 4:
    grid = map_points(mapping_points)

frame = cv2.imread(PHOTO_FILE)
is_chessboard_detected = detect_chessboard(frame)

while is_chessboard_detected:
    # ret, frame = cap.read()
    # frame = cv2.imread(PHOTO_FILE)
    # frame = detect_circle(frame)

    for point in mapping_points:
        cv2.circle(frame, point, 2, OUTLINE_COLOR, -1)
        if grid is not None:
            # Apply the inverse perspective transform to get the position in mm
            mm_point = get_position_in_mm(point)
            cv2.putText(frame, f"mm: {mm_point[0]:.2f}, {mm_point[1]:.2f}", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, OUTLINE_COLOR, 1)

    if is_mapped == True:
        # cv2.polylines(frame, [np.array(mapping_points)], isClosed=True, color=OUTLINE_COLOR, thickness=2)
        grid = map_points(mapping_points)  


    for point in points:
        cv2.circle(frame, point, 2, POINT_COLOR, -1)

        #show position in mm based on grid mapping
        if grid is not None:
            # Apply the inverse perspective transform to get the position in mm
            mm_point = get_position_in_mm(point)
            cv2.putText(frame, f"mm: {mm_point[0]:.2f}, {mm_point[1]:.2f}", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, POINT_COLOR, 1)

    # if len(points) == 2:
    #         distance = pixels_between(points[0], points[1])
    #         cv2.putText(frame, f"Distance: {distance:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, POINT_COLOR, 2)

    cv2.imshow("Frame", frame)
    # cv2.setMouseCallback("Frame", draw_point)    

    if is_mapped == False:
        cv2.setMouseCallback("Frame", set_mapping_points)
        if len(mapping_points) == 4:
            is_mapped = True
    else:
        cv2.setMouseCallback("Frame", draw_point)



    # If two points are clicked, draw a line between them and calculate the distance
    # if len(points) == 2:
    #     cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
    #     distance = ((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2) ** 0.5
    #     cv2.putText(frame, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break


# cap.release()
cv2.destroyAllWindows()