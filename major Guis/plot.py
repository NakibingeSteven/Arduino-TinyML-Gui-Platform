import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Style, Treeview
from everywhereml.sklearn.ensemble import RandomForestClassifier
import tkinter.messagebox as messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.ttk import Combobox
import random
import csv
from decimal import Decimal
import numpy as np

class MLGui:
    def __init__(self):
        # ... Your existing code ...

    # ... Your other functions ...

    def generate_ultrasonic_csv(self, numvalues):
        data = []
        self.numValues = numvalues
        data.append(["Distance", "Command"])

        for i in range(self.numValues):
            distance = Decimal(random.uniform(10, 50)).quantize(Decimal("0.00"))
            if distance < 20:
                command = "bad"
            elif distance < 29:
                command = "good"
            else:
                command = "safe"
            data.append([distance, command])

        with open(self.filename + ".csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"CSV file '{self.filename + '.csv'}' generated successfully!")

    def doubleLinearRegressData(self):
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 2 * X + 3 + np.random.randn(100, 1) * 2
        df = pd.DataFrame({"X": X.flatten(), "Y": y.flatten()})
        df.to_csv(self.filename + ".csv", index=False)

    def tripleLinearRegressData(self):
        np.random.seed(42)
        X = np.random.rand(100, 3) * 10
        y = (X[:, 0] + 2 * X[:, 1] - 3 * X[:, 2] + np.random.randn(100) * 2)
        df = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2], "y": y})
        df.to_csv(self.filename + ".csv", index=False)

    def add_data_generators_to_menu(self):
        data_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="Generate Synthetic Data", command=self.make_data)
        data_menu.add_command(label="Generate Ultrasonic Data", command=self.generate_ultrasonic_data)
        data_menu.add_command(label="Generate Two Column Number Data", command=self.doubleLinearRegressData)
        data_menu.add_command(label="Generate Three Column Number Data", command=self.tripleLinearRegressData)

    # ... Your other functions ...

if __name__ == "__main__":
    ml_gui = MLGui()
    ml_gui.add_data_generators_to_menu()  # Add data generators to the menu
    ml_gui.run()
