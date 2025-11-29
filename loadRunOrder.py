
import os
import csv
SVC_FOLDER_PATH = "Run_Order.csv"

# Object of a run with Std, Order, Origin, Polynomial_Order, Sampling_Density properties
class Run:
    def __init__(self, Id, Std, Order, Origin, Polynomial_Order, Sampling_Density, board_cell_width = 10):
        self.Id = Id
        self.Std = Std
        self.Order = Order
        self.Origin = Origin
        self.Polynomial_Order = Polynomial_Order
        self.Sampling_Density = Sampling_Density
        self.Cell_Width = board_cell_width  

# Function load run order from a csv file use csv library
def load_run_order():
    try:
        with open(SVC_FOLDER_PATH, 'r') as file:
            reader = csv.reader(file)
            run_order = [row for row in reader if row]
            return run_order
    except FileNotFoundError:
        print(f"File not found: {SVC_FOLDER_PATH}")
        return []

# Function to parse run order into list of Run objects. Skips header row.  
def parse_run_order(run_order):
    runs = []
    print(f"Parsing {len(run_order)} entries from run order")
    for entry in run_order:
        if entry[0].strip().lower() == 'std':
            continue  # Skip header row
        if len(entry) >= 5:
            Std = entry[0].strip()
            Order = entry[1].strip()
            Origin = entry[2].strip()
            Polynomial_Order = int(entry[3].strip())
            Sampling_Density = int(entry[4].strip())
            Id = f'{Std}_{Order}_{Origin}_{Polynomial_Order}_{Sampling_Density}'
            run = Run(Id, Std, Order, Origin, Polynomial_Order, Sampling_Density)
            runs.append(run)
    return runs

# Function to get all runs ordered as Run objects
def get_all_runs():
    res = parse_run_order(load_run_order())
    print(f"Loaded {len(res)} runs from {SVC_FOLDER_PATH}")
    return res