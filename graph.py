import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1️⃣ Define your model function
def growth_model(t, A, k, n):
    return A * (1 - np.exp(-k * t**n))

# Example usage:
# t_data, y_data = load_data_from_csv('data.csv')


# 2️⃣ Example experimental data
# Replace these with your actual measurements

t_data = []
y_data = []



#load experimental data cvs file
def load_data_from_csv():
    t = []
    y = []
    with open("chessboard_corners_20251108_044331.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header if there is one
        for row in reader:
            t.append(float(row[2]))
            y.append(float(row[0]))
    return np.array(t), np.array(y)

t_data, y_data = load_data_from_csv()

# print length of data
print(f"Loaded {len(t_data)} data points.")

# 3️⃣ Initial guesses for A, k, n
initial_guess = [100, 0.001, 1]

# 4️⃣ Fit the model
popt, pcov = curve_fit(growth_model, t_data, y_data, p0=initial_guess)

# Extract fitted parameters
A_fit, k_fit, n_fit = popt
print(f"A = {A_fit:.4f}, k = {k_fit:.6f}, n = {n_fit:.4f}")

# 5️⃣ Compute fitted curve
t_fit = np.linspace(0, 2000, 200)
y_fit = growth_model(t_fit, *popt)

# 6️⃣ Plot
plt.scatter(t_data, y_data, color='red', label='Data')
plt.plot(t_fit, y_fit, 'b-', label='Fit: A(1-exp(-k*t^n))')
plt.title('Nonlinear Curve Fit: A(1 - exp(-k*t^n))')
plt.xlabel('x (pixel)')
plt.ylabel('x (mm)')
plt.legend()
plt.grid(True)
plt.show()