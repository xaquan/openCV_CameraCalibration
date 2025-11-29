import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def load_calibration_data(csv_path):
    """
    Load calibration data from CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing calibration data
    
    Returns:
    --------
    tuple
        (X_pixels, y_real) - pixel coordinates and real-world coordinates
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    X_pixels = df[['x_pixel', 'y_pixel']].values
    y_real = df[['x_real', 'y_real']].values

    return X_pixels, y_real


def train_calibration_model(X_pixels, y_real, poly_degree=2):
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
        'degree': poly_degree
    }


def compare_models(X_pixels, y_real, degrees_to_test=[2, 3, 6]):
    """
    Train and compare models with different polynomial degrees.
    
    Parameters:
    -----------
    X_pixels : array-like, shape (n_samples, 2)
        Pixel coordinates [x_pixel, y_pixel]
    y_real : array-like, shape (n_samples, 2)
        Real-world coordinates [x_real, y_real]
    degrees_to_test : list of int
        List of polynomial degrees to test
    
    Returns:
    --------
    dict
        Dictionary mapping degree to model results
    """
    results = {}
    
    print("="*70)
    print("CAMERA CALIBRATION: Pixel to Real-World Coordinate Mapping")
    print("="*70)
    print(f"Data points: {len(X_pixels)}")
    print(f"Pixel range X: [{X_pixels[:,0].min():.1f}, {X_pixels[:,0].max():.1f}]")
    print(f"Pixel range Y: [{X_pixels[:,1].min():.1f}, {X_pixels[:,1].max():.1f}]")
    print(f"Real range X: [{y_real[:,0].min():.1f}, {y_real[:,0].max():.1f}]")
    print(f"Real range Y: [{y_real[:,1].min():.1f}, {y_real[:,1].max():.1f}]")
    
    for degree in degrees_to_test:
        result = train_calibration_model(X_pixels, y_real, poly_degree=degree)
        results[degree] = result
        
        print(f"\n{'='*70}")
        print(f"POLYNOMIAL DEGREE {degree}")
        print(f"{'='*70}")
        print(f"X-coordinate:")
        print(f"  R² Score:  {result['r2_x']:.6f}")
        print(f"  MAE:       {result['mae_x']:.4f} units")
        print(f"  RMSE:      {result['rmse_x']:.4f} units")
        print(f"\nY-coordinate:")
        print(f"  R² Score:  {result['r2_y']:.6f}")
        print(f"  MAE:       {result['mae_y']:.4f} units")
        print(f"  RMSE:      {result['rmse_y']:.4f} units")
        print(f"\nAverage R²: {(result['r2_x'] + result['r2_y'])/2:.6f}")
        print(f"Average MAE: {(result['mae_x'] + result['mae_y'])/2:.4f} units")
    
    return results


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
    
    # Print the coefficients of the best model
    best_result = results[best_degree]
    print(f"\n{'='*70}")
    print("BEST MODEL DETAILS:")
    print(f"{'='*70}")
    print(f"Polynomial Degree: {best_degree}")
    print(f"X-coordinate Coefficients: {best_result['model_x'].coef_}")
    print(f"Y-coordinate Coefficients: {best_result['model_y'].coef_}")

    return best_degree, results[best_degree]


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


def plot_comparison(y_real, results, degrees_to_test):
    """
    Plot comparison of different polynomial degrees.
    
    Parameters:
    -----------
    y_real : array-like, shape (n_samples, 2)
        Actual real-world coordinates
    results : dict
        Dictionary of model results
    degrees_to_test : list of int
        List of degrees that were tested
    """
    fig, axes = plt.subplots(2, len(degrees_to_test), figsize=(6*len(degrees_to_test), 12))
    fig.suptitle('Polynomial Regression Comparison: Pixel to Real-World Coordinates', 
                 fontsize=14, fontweight='bold')
    
    for idx, degree in enumerate(degrees_to_test):
        res = results[degree]
        
        # X-coordinate plot
        ax1 = axes[0, idx]
        ax1.scatter(y_real[:, 0], res['pred_x'], alpha=0.5, s=20)
        ax1.plot([y_real[:, 0].min(), y_real[:, 0].max()], 
                 [y_real[:, 0].min(), y_real[:, 0].max()], 
                 'r--', linewidth=2, label='Perfect fit')
        ax1.set_xlabel('Actual X Real')
        ax1.set_ylabel('Predicted X Real')
        ax1.set_title(f'Degree {degree}: X-coord\nR²={res["r2_x"]:.4f}, MAE={res["mae_x"]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Y-coordinate plot
        ax2 = axes[1, idx]
        ax2.scatter(y_real[:, 1], res['pred_y'], alpha=0.5, s=20)
        ax2.plot([y_real[:, 1].min(), y_real[:, 1].max()], 
                 [y_real[:, 1].min(), y_real[:, 1].max()], 
                 'r--', linewidth=2, label='Perfect fit')
        ax2.set_xlabel('Actual Y Real')
        ax2.set_ylabel('Predicted Y Real')
        ax2.set_title(f'Degree {degree}: Y-coord\nR²={res["r2_y"]:.4f}, MAE={res["mae_y"]:.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_2d_scatter(y_real, results, degrees_to_test):
    """
    Plot 2D scatter of actual vs predicted coordinates.
    
    Parameters:
    -----------
    y_real : array-like, shape (n_samples, 2)
        Actual real-world coordinates
    results : dict
        Dictionary of model results
    degrees_to_test : list of int
        List of degrees that were tested
    """
    fig, axes = plt.subplots(1, len(degrees_to_test), figsize=(6*len(degrees_to_test), 6))
    fig.suptitle('2D Scatter: Actual vs Predicted Real Coordinates', 
                 fontsize=14, fontweight='bold')
    
    if len(degrees_to_test) == 1:
        axes = [axes]
    
    for idx, degree in enumerate(degrees_to_test):
        res = results[degree]
        
        ax = axes[idx]
        ax.scatter(y_real[:, 0], y_real[:, 1], color='b', 
                  label='Actual Real Coordinates', alpha=0.5, s=30)
        ax.scatter(res['pred_x'], res['pred_y'], color='r', 
                  label='Predicted Real Coordinates', alpha=0.5, s=30)
        ax.set_xlabel('Real X')
        ax.set_ylabel('Real Y')
        ax.set_title(f'Degree {degree}\nR²={((res["r2_x"]+res["r2_y"])/2):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_error_analysis(y_real, best_result):
    """
    Plot error distribution for the best model.
    
    Parameters:
    -----------
    y_real : array-like, shape (n_samples, 2)
        Actual real-world coordinates
    best_result : dict
        Best model result
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Error Analysis (Degree {best_result["degree"]})', 
                 fontsize=14, fontweight='bold')
    
    # X errors
    errors_x = y_real[:, 0] - best_result['pred_x']
    axes[0].hist(errors_x, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error (Real - Predicted)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'X-coordinate Errors\nMean: {errors_x.mean():.4f}, Std: {errors_x.std():.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # Y errors
    errors_y = y_real[:, 1] - best_result['pred_y']
    axes[1].hist(errors_y, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Error (Real - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Y-coordinate Errors\nMean: {errors_y.mean():.4f}, Std: {errors_y.std():.4f}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # 1. Load data
    csv_path = 'data/sampling_3_21_Corner_3_10.csv'
    X_pixels, y_real = load_calibration_data(csv_path)

    # 2. Compare different polynomial degrees
    degrees_to_test = [1, 3, 6]
    results = compare_models(X_pixels, y_real, degrees_to_test)
    
    # 3. Get best model
    best_degree, best_result = get_best_model(results)
    
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
    pred_x, pred_y = predict_real_coordinates(best_result, test_x_px, test_y_px)
    print(f"Pixel coordinates: ({test_x_px}, {test_y_px})")
    print(f"Predicted real coordinates: ({pred_x:.2f}, {pred_y:.2f})")
    
    # 5. Visualizations
    plot_comparison(y_real, results, degrees_to_test)
    plot_2d_scatter(y_real, results, degrees_to_test)
    plot_error_analysis(y_real, best_result)
    
    # 6. Save model (optional)
    print(f"\n{'='*70}")
    print("SAVE MODEL:")
    print(f"{'='*70}")
    print("""
import pickle

# Save the best model
with open('best_calibration_model.pkl', 'wb') as f:
    pickle.dump(best_result, f)

# Load and use later
with open('best_calibration_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
pred_x, pred_y = predict_real_coordinates(model, x_px, y_px)
    """)