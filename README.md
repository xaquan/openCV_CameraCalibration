# 📷 OpenCV Camera Calibration — Polynomial Regression & DOE Analysis

This repository contains the full dataset, scripts, and tools used to evaluate camera lens distortion using **2D polynomial regression** and a structured **Design of Experiments (DOE)** workflow. The project includes automated experiment execution, visualization tools, uncertainty analysis, and an interactive measurement UI.

---

## 📁 Repository Structure

```plaintext
openCV_CameraCalibration/
│
├── data/                      # Output of DOE runs, regression results, CSV files
│   ├── *.csv                  # Saved outputs for each run
│   ├── corner_points/         # Extracted corner data
│   └── ...
│
├── photos/                    # 20 distorted calibration images
│   └── *.jpg / *.png
│
├── run_order.csv              # DOE run list (polynomial degree, origin, sampling)
├── runOrder.py                # Automated experiment runner for DOE
├── measure.py                 # Interactive UI for measuring distortion in mm
├── testing_concept.py         # Regression + visualization tool
│
└── README.md                  # Project documentation
```

---

## 🎯 Project Purpose

This project investigates how well **1st, 3rd, and 6th-degree** 2D polynomial models can correct camera lens distortion.  
A DOE approach varies:

- **Polynomial Degree:** 1, 3, 6  
- **Sampling Density:** 10 mm and 20 mm  
- **Origin Selection:** Corner or Center

The goal is to find the most accurate and stable polynomial model for converting pixel coordinates → real-world coordinates (mm), and to analyze residual distortion and measurement uncertainty.

---

## 🚀 Scripts and Their Functions

### ✔ runOrder.py — Automated DOE Runner

Automatically executes all calibration experiments defined in `run_order.csv`.

**What it does:**
1. Reads DOE parameters  
2. Loads a random photo from `/photos/`  
3. Extracts corners using OpenCV  
4. Fits the polynomial regression model  
5. Saves results to `/data/`  
6. Records metrics and coefficients  

Run it:
```bash
python runOrder.py
```

---

### ✔ measure.py — Interactive Distortion Measurement UI

A user interface that allows you to inspect distortion manually.

**Features:**
- Click any point in the image  
- View pixel coordinates  
- Convert to real-world (mm) using the best-fit polynomial  
- Useful for debugging or demonstration  
- Helps visualize distortion at arbitrary points  

Run it:
```bash
python measure.py
```

---

### ✔ testing_concept.py — Regression & Visualization Tool

Loads DOE output files and generates visualizations.

**What it produces:**
- Actual vs. predicted plots  
- Residual error scatter plots  
- R², MAE, RMSE summaries  
- Error heatmaps  
- Polynomial surface/extrapolation visualizations  
- Comparison across DOE settings  

Run it:
```bash
python testing_concept.py
```

This tool is used to generate figures and tables for the thesis.

---

## 📸 Example Data

The `/photos` folder contains 20 distorted chessboard images used for:

- Corner detection  
- Polynomial fitting  
- UI measurement  
- Reproducing the calibration process  

---

## 📊 Data Outputs

Each row in `run_order.csv` produces:

- Polynomial coefficients  
- Predicted vs. actual grid points  
- R², MAE, RMSE  
- Error statistics  
- CSV output saved in `/data/`  

These files serve as the basis for thesis results:

- Sampling effect  
- Origin effect  
- Polynomial order comparison  
- Error heatmaps  
- Uncertainty calculations  

---

## 🧩 Requirements

Install dependencies:

```bash
pip install opencv-python numpy pandas matplotlib scipy
```

---

## ▶️ How to Run the Project

### Run all DOE experiments:
```bash
python runOrder.py
```

### Run only the measurement UI:
```bash
python measure.py
```

### Visualize results:
```bash
python testing_concept.py
```

### Run a specific DOE row:
```bash
python runOrder.py
```

---

## 🔁 Reproducibility

This repository includes:

- Raw distorted images  
- DOE parameter list  
- All code for regression, visualization, and uncertainty  
- All experiment outputs in `/data/`  

Every figure used in the thesis can be regenerated from the scripts here.

---

## 📄 License

MIT License
