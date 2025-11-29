import datetime
import math
import cv2
import numpy as np
from datetime import datetime
import os
import polynomialRegression


samples = []
samples_reference = []
sample_filename = ""
mapping_points = []
grid = None
image = None

# Declare a class to hold detected board information with samples, mapping points, etc.
class DetectedBoard:
    def __init__(self, board_id, board_size: tuple, cell_width: float, origin: tuple):
        self.board_id = board_id
        self.board_size = board_size
        self.cell_width = cell_width
        self.origin = origin
        self.samples = []
        self.samples_real_coordinates = []
        self.regression_points = []
        self.mapping_points = []
        self.grid = None
        self.regression_model = None

#Mapping 4 points as the rectangle , use 4 center of circles except the circle at the middle
def transfrom_detected_board(points, cols, rows, cell_width):

    width_mm = cols * cell_width
    height_mm = rows * cell_width

    if len(points) != 4:
        raise ValueError("Exactly 4 points are required to map a rectangle.")
    
    # print("Mapping points:")
    # print(points)
    # Define the source points (the points clicked by the user)
    src_pts = np.array(points, dtype="float32")
    # Define the destination points (the corners of the output rectangle)
    dst_pts = np.array([[0, 0], [width_mm, 0], [0, height_mm], [width_mm, height_mm]], dtype="float32")
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M

# Function to detect chessboard corners in an image
def detect_chessboard(img, board_id, board_size: tuple, cell_width, origin, point_color=(0, 0, 255), show_detected_points = False):
    
    print(f"Detecting chessboard of size {board_size} with cell width {cell_width} at origin {origin}")

    board = DetectedBoard(board_id, board_size, cell_width, origin)
    # convert the input image to a grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try different flag combinations for better detection
    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_FAST_CHECK,
        None  # No flags
    ]

    res = False
    samples = None
        
    for flags in flags_list:
        if flags is None:
            res, samples = cv2.findChessboardCorners(gray, board_size)
        else:
            res, samples = cv2.findChessboardCorners(gray, board_size, flags=flags)
        
        if res:
            print(f"Chessboard detected with flags: {flags}")
            break
    
    print(f"Chessboard corners found: {res}")

    # if chessboard corners are detected
    if res == True:   

        board.samples = samples
        #save corners to csv file, add timestamp to filename
        # save_sampling(board_size, cell_width, samples)

        board.mapping_points = mapping_corners(board_size[0], board_size[1], samples)
        board.grid = transfrom_detected_board(board.mapping_points, board_size[0], board_size[1], cell_width)

        # Display the image with drawn points if requested
        if show_detected_points:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, board_size, samples,res)
            
    return res, board


def mapping_corners(cols, rows, samples):
    res = []
    for i in [0, cols -1, cols * (rows -1), cols * rows -1]:
        point = samples[i]
        res.append((point[0][0], point[0][1]))
    return res
    # Call transform function

