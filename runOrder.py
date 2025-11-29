# Get run order from loadRunOrder.py and provide as Run objects
import random
from analyzeSampling import analyze_sampling_data
import checkErrorDisplay
import loadRunOrder
import loadPhotoFiles
import chessboardProcess
import cv2
from loadRunOrder import Run

chessboard_paths = loadPhotoFiles.load_1cm_chessboard_filenames()

chessboard_size = (21, 21)  # columns, rows
cell_width = 10  # in mm


# Function to run experiment based on Run object
def run_experiment(run : Run):
    # run_id = f"{run.Std}_{run.Order}_{run.Origin}_{run.Polynomial_Order}_{run.Sampling_Density}"
    print(f"Running experiment with Std: Order: {run.Order}, Origin: {run.Origin}, Polynomial_Order: {run.Polynomial_Order}, Sampling_Density: {run.Sampling_Density}")

    # load random chessboard photo from the chessboard_paths
    ran_i = random.randint(0, len(chessboard_paths) - 1)
    photo_filename = chessboard_paths[ran_i]

    # Set cell width for the board cell size
    run.Cell_Width = cell_width

    # Load photo
    img = loadPhotoFiles.load_photo(photo_filename)
    print(f"Loaded image: {photo_filename}")

    # Chess board detection using chessboardProcess module
    res, detected_board = chessboardProcess.detect_chessboard(img, run.Id, chessboard_size, run.Cell_Width, run.Origin, show_detected_points=False)

    if res is False:
        print(f"Chessboard detection failed for image: {photo_filename}")
        return

    data_filename = f"data/sampling_{run.Id}.csv"

    regression_model = analyze_sampling_data(detected_board, run, data_filename)
    log_regression_model(run, regression_model)

    # checkErrorDisplay.show_chessboard_detection(img, detected_board)

# Function log the regression model to newline to csv file 
# Format:
# 'poly': poly,
# 'model_x': model_x,
# 'model_y': model_y,
# 'pred_x': pred_x,
# 'pred_y': pred_y,
# 'r2_x': r2_x,
# 'r2_y': r2_y,
# 'mae_x': mae_x,
# 'mae_y': mae_y,
# 'rmse_x': rmse_x,
# 'rmse_y': rmse_y,
# 'degree': poly_degree
# 'r2_mean': r2_mean
# Skip pred_x, pred_y, model_x and model_y
def log_regression_model(run: Run, regression_model):
    filename = "data/0.result_regression_model.csv"
    header = "run_id, degree,r2_x,r2_y,mae_x,mae_y,rmse_x,rmse_y, r2_mean, mae_mean, rmse_mean\n"

    r2_mean = (regression_model['r2_x'] + regression_model['r2_y']) / 2
    mae_mean = (regression_model['mae_x'] + regression_model['mae_y']) / 2
    rmse_mean = (regression_model['rmse_x'] + regression_model['rmse_y']) / 2

    # if file does not exist, create it and write header
    try:
        with open(filename, "x") as f:
            f.write(header)
    except FileExistsError:
        pass

    with open(filename, "a") as f:
        f.write(f"{run.Id}, {regression_model['degree']}, {regression_model['r2_x']}, {regression_model['r2_y']}, {regression_model['mae_x']}, {regression_model['mae_y']}, {regression_model['rmse_x']}, {regression_model['rmse_y']}, {r2_mean}, {mae_mean}, {rmse_mean}\n")

# Function to execute all runs
def execute_all_runs():
    runs = loadRunOrder.get_all_runs()
    print(f"Total runs to execute: {len(runs)}")
    for run in runs:
        run_experiment(run)

    # Run 1th and 3rd experiments for testing
    # run_experiment(runs[6])
    # run_experiment(runs[10])
    # run_experiment(runs[4])

        

# Function load all necessary data and execute experiments
if __name__ == "__main__":
    execute_all_runs()