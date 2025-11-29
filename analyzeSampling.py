

from datetime import datetime
import os
import numpy as np
from chessboardProcess import DetectedBoard
from loadRunOrder import Run
import polynomialRegression

# Function to analyze sampling data and save to file
# filename is the file to save the analyzed data
# filename: optional, if not provided, generate timestamped filename
def analyze_sampling_data(detected_board: DetectedBoard, run: Run, filename=None):

   
    detected_board = prepare_sampling(detected_board, run)

    detected_board.regression_model = generate_regression_by_degree(detected_board, run.Polynomial_Order)
    detected_board.regression_points = detected_board.regression_model['pred_x'], detected_board.regression_model['pred_y']
    # print(detected_board.regression_model)
    # Regression points
    # for point in detected_board.samples:
    #     pred_point = polynomialRegression.predict_real_coordinates(detected_board.regression_model, point[0][0], point[0][1])
    #     detected_board.regression_points.append(pred_point)

    save_sampling(detected_board, run, filename)

    return detected_board.regression_model

# Function to prepare sampling based on sampling density
# If density is 10, use all samples
# If density is 20, reduce samples by half (remove every second sample in each 10-sample block)
def prepare_sampling(detected_board: DetectedBoard, run: Run):

    for i, point in enumerate(detected_board.samples):
        x_real, y_real = get_real_coordinate_a_chess_cell(detected_board.board_size, detected_board.cell_width, i, detected_board.origin)
        detected_board.samples_real_coordinates.append((x_real, y_real))

     # If density is 20 then remove odd index of every row
    if run.Sampling_Density == 20:
        filtered_samples = []
        filtered_real_coordinates = []

        for i in range(detected_board.board_size[0] * detected_board.board_size[1]):  # for each row
            cur_row = i % detected_board.board_size[0]
            if cur_row % 2 == 0:
                # remoce this sample
                filtered_samples.append(detected_board.samples[i])
                filtered_real_coordinates.append(detected_board.samples_real_coordinates[i])

        detected_board.samples = filtered_samples
        detected_board.samples_real_coordinates = filtered_real_coordinates

        print(f"Reduced samples for density {run.Sampling_Density}, total samples now: {len(detected_board.samples)}, and real coordinates: {len(detected_board.samples_real_coordinates)}")
    return detected_board
# Function to save sampling data to CSV file
def save_sampling(detected_board: DetectedBoard, run: Run, filename=None):
    
    sample_filename = filename or f'data/{generate_timestamped_filename()}'
    # if directory "data" does not exist, create it
    if not os.path.exists("data"):
        os.makedirs("data")

    with open(sample_filename, "w") as f:
        header = "x_pixel, y_pixel, x_real, y_real, distored_x, distored_y, pred_x, pred_y, error_x, error_y\n"
        f.write(header)
        for i, point in enumerate(detected_board.samples):
            x_real, y_real = detected_board.samples_real_coordinates[i]
            distored_x, distored_y = convert_point_px_to_mm((point[0][0], point[0][1]), detected_board.grid)

            pred_point = detected_board.regression_points[0][i], detected_board.regression_points[1][i]
            error_x = x_real - pred_point[0]
            error_y = y_real - pred_point[1]

            f.write(f"{point[0][0]}, {point[0][1]}, {x_real}, {y_real}, {distored_x}, {distored_y}, {pred_point[0]}, {pred_point[1]}, {error_x}, {error_y}\n")
    print(f"Saved sampling to {sample_filename}")

# Function to plot samples with real_x, real_y and line using regression coefficients
def plot_regression_coefficients(detected_board: DetectedBoard):
    import matplotlib.pyplot as plt

    x_real = [coord[0] for coord in detected_board.samples_real_coordinates]
    y_real = [coord[1] for coord in detected_board.samples_real_coordinates]

    x_px = [point[0][0] for point in detected_board.samples]
    y_px = [point[0][1] for point in detected_board.samples]

    distorted_x = []
    distorted_y = []
    for point in detected_board.samples:
        dx, dy = convert_point_px_to_mm((point[0][0], point[0][1]), detected_board.grid)
        distorted_x.append(dx)
        distorted_y.append(dy)

    regr_x = []
    regr_y = []
    for i in range(len(x_px)):
        regr_point = polynomialRegression.run_regression_polynomial(detected_board.regression_model, x_px[i], y_px[i])
        regr_x.append(regr_point[0])
        regr_y.append(regr_point[1])

    plt.figure(num=detected_board.board_id, figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(x_real, y_real, color='blue', label='Real Coordinates')
    plt.scatter(regr_x, regr_y, color='red', label='Regression Fit')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Regression Fit vs Real Coordinates')
    plt.legend()
    plt.axis('equal')

    # Plot distorted points vs real points same figure
    plt.subplot(2, 2, 2)
    plt.scatter(distorted_x, distorted_y, color='purple', label='Distorted Points')
    plt.scatter(x_real, y_real, color='blue', label='Real Coordinates')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Distorted Points vs Regression Fit vs Real Coordinates')
    plt.legend()
    plt.axis('equal')

    
    plt.subplot(2, 2, 3)
    errors_x = [x_real[i] - regr_x[i] for i in range(len(x_real))]
    errors_y = [y_real[i] - regr_y[i] for i in range(len(y_real))]
    plt.scatter(errors_x, errors_y, color='green')
    plt.xlabel('Error in X (mm)')
    plt.ylabel('Error in Y (mm)')
    plt.title('Regression Errors')
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


def generate_regression_by_degree(detected_board: DetectedBoard, degree: int):
    return polynomialRegression.gnerate_regression_polynomial(detected_board.samples_real_coordinates, detected_board.samples, degree)

# Function to generate timestamped filename
def generate_timestamped_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"sampling_{timestamp}.csv"

# Function to get real world coordinates based on index
def get_real_coordinate_a_chess_cell(size, cell_width, cell_index, origin):

    x_real = (cell_index % size[0]) * cell_width
    y_real = (cell_index // size[0]) * cell_width

    if str.lower(origin) == 'center':
        offset_x = - (size[0] -1) * cell_width / 2
        offset_y = - (size[1] -1) * cell_width / 2
        x_real += offset_x
        y_real += offset_y

    return x_real,y_real

# Function convert a point in pixel to mm using the grid mapping
def convert_point_px_to_mm(point_px, grid):
    point_homogeneous = np.array([point_px[0], point_px[1], 1], dtype="float32")
    mm_point = grid @ point_homogeneous
    mm_point /= mm_point[2]  # Normalize
    return mm_point[:2]

def get_best_model(results):
    """
    Get the best model based on average R² score.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results from compare_models()
    
    Returns:
    --------
    tuple
        (best_degree, best_result)
    """
    best_degree = max(results.keys(), 
                     key=lambda d: (results[d]['r2_x'] + results[d]['r2_y'])/2)
    return best_degree, results[best_degree]