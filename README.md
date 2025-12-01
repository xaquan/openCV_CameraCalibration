openCV_CameraCalibration/
│
├── data/                  # Output folder for regression results and logs
│   ├── *.csv              # Saved outputs from each DOE run
│   ├── corner_points/     # Example detected corner data
│   └── ...                # Supporting data files
│
├── photos/                # 20 distorted calibration board images
│   └── *.jpg / *.png
│
├── run_order.csv          # DOE run order: origin, degree, sampling, etc.
├── runOrder.py            # Main experiment runner (reads run_order.csv)
│
└── README.md              # Project documentation (this file)
