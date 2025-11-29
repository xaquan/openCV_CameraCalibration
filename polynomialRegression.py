from unittest import result
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import testing_concept

LIBRARY = "numpy"  # "sklearn" or "numpy"

def gnerate_regression_polynomial(samples_real_coordinates, samples, polinomial_degree):

    # print("Generating regression polynomial... degree:", polinomial_degree)
    # print(f"samples_reference length: {len(samples_reference)}")
    # print(f"samples length: {len(samples)}")

# More pythonic way to create the arrays
    pixels = np.array(samples, dtype=float).reshape(-1, 2)
    reals   = np.array(samples_real_coordinates, dtype=float).reshape(-1, 2)

    X_pixels = pixels
    y_real   = reals


    # if LIBRARY == "sklearn":
    #     res = regression_sklearn(polinomial_degree, x, y, x_px, y_px)
    # else:
    #     res = regression_numpy(polinomial_degree, x, y, x_px, y_px)

    # callTest(X_pixels, y_real)

    calibration_model = generate_regression_sklearn(X_pixels, y_real, poly_degree=polinomial_degree)  

    print(f"\n{'='*70}")
    print(f"POLYNOMIAL DEGREE {polinomial_degree}")
    print(f"{'='*70}")
    print(f"X-coordinate:")
    print(f"  R² Score:  {calibration_model['r2_x']:.6f}")
    print(f"  MAE:       {calibration_model['mae_x']:.4f} units")
    print(f"  RMSE:      {calibration_model['rmse_x']:.4f} units")
    print(f"\nY-coordinate:")
    print(f"  R² Score:  {calibration_model['r2_y']:.6f}")
    print(f"  MAE:       {calibration_model['mae_y']:.4f} units")
    print(f"  RMSE:      {calibration_model['rmse_y']:.4f} units")
    print(f"\nAverage R²: {(calibration_model['r2_x'] + calibration_model['r2_y'])/2:.6f}")
    print(f"Average MAE: {(calibration_model['mae_x'] + calibration_model['mae_y'])/2:.4f} units")

    # print(f"x={x[0]}, x_px={x_px[0]}")

    return calibration_model

# Call frnction form testing_concept.py to test the regression model
def callTest(X_pixels, y_real):
    
    # 2. Compare different polynomial degrees
    degrees_to_test = [1, 3, 6]
    results = testing_concept.compare_models(X_pixels, y_real, degrees_to_test)
    
    # 3. Get best model
    best_degree, best_result = testing_concept.get_best_model(results)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    print(f"{'='*70}")
    print(f"Best model: Polynomial Degree {best_degree}")
    print(f"Average R²: {(best_result['r2_x'] + best_result['r2_y'])/2:.6f}")
    print(f"This model provides the best accuracy for camera calibration.")
    
    # 4. Example prediction
    print(f"\n{'='*70}")
    print("EXAMPLE PREDICTION:")
    print(f"{'='*70}")
    test_x_px, test_y_px = 500, 500
    pred_x, pred_y = testing_concept.predict_real_coordinates(best_result, test_x_px, test_y_px)
    print(f"Pixel coordinates: ({test_x_px}, {test_y_px})")
    print(f"Predicted real coordinates: ({pred_x:.2f}, {pred_y:.2f})")
    
    # 5. Visualizations
    testing_concept.plot_comparison(y_real, results, degrees_to_test)
    testing_concept.plot_2d_scatter(y_real, results, degrees_to_test)
    testing_concept.plot_error_analysis(y_real, best_result)


def generate_regression_sklearn(X_pixels, y_real, poly_degree=2):
    """
    Train a camera calibration model with specified polynomial degree.
    
    Parameters:
    -----------
    X_pixels : array-like, shape (n_samples, 2)
        Pixel coordinates [x_pixel, y_pixel]
    y_real : array-like, shape (n_samples, 2)
        Real-world coordinates [x_real, y_real]
    poly_degree : int, default=2
        Degree of polynomial features
    
    Returns:
    --------
    dict
        Dictionary containing trained models and metrics
    """
    # Transform features
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X_pixels)
    
    # Separate models for X and Y coordinates
    model_x = LinearRegression()
    model_y = LinearRegression()
    
    model_x.fit(X_poly, y_real[:, 0])
    model_y.fit(X_poly, y_real[:, 1])
    
    # Predictions
    pred_x = model_x.predict(X_poly)
    pred_y = model_y.predict(X_poly)
    
    # Calculate metrics
    r2_x = r2_score(y_real[:, 0], pred_x)
    r2_y = r2_score(y_real[:, 1], pred_y)
    mae_x = mean_absolute_error(y_real[:, 0], pred_x)
    mae_y = mean_absolute_error(y_real[:, 1], pred_y)
    rmse_x = np.sqrt(mean_squared_error(y_real[:, 0], pred_x))
    rmse_y = np.sqrt(mean_squared_error(y_real[:, 1], pred_y))

    r2_mean = (r2_x + r2_y) / 2
    
    return {
        'poly': poly,
        'model_x': model_x,
        'model_y': model_y,
        'pred_x': pred_x,
        'pred_y': pred_y,
        'r2_x': r2_x,
        'r2_y': r2_y,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'degree': poly_degree,
        'r2_mean': r2_mean
    }


def predict_real_coordinates(model, x_px, y_px):
    """
    Predict real-world coordinates from pixel coordinates.
    
    Parameters:
    -----------
    model : dict
        Trained calibration model
    x_px : float or array-like
        Pixel X coordinate(s)
    y_px : float or array-like
        Pixel Y coordinate(s)
    
    Returns:
    --------
    tuple
        (predict_x, predict_y) - predicted real-world coordinates
    """
    # Handle single point or multiple points
    x_px = np.atleast_1d(x_px)
    y_px = np.atleast_1d(y_px)
    
    # Prepare input
    X_pixels = np.column_stack([x_px, y_px])
    X_poly = model['poly'].transform(X_pixels)
    
    # Predict
    predict_x = model['model_x'].predict(X_poly)
    predict_y = model['model_y'].predict(X_poly)
    
    # Return single values if input was single point
    if len(predict_x) == 1:
        return predict_x[0], predict_y[0]
    else:
        return predict_x, predict_y