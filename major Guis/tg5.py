import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Style, Treeview
from sklearn.datasets import make_blobs

# from everywhereml.sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import tkinter.messagebox as messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.ttk import Combobox
import csv
import random
from decimal import Decimal
import numpy as np
import os
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Add this import
import joblib
from micromlgen import port


class MLGui:
    def __init__(self):
        self.topLevel = tk.Tk()
        self.topLevel.geometry("800x600")
        self.topLevel.title("Machine Learning Trainer gUI")

        style = Style()
        style.configure("TButton", font=("Helvetica", 12))

        # Initialize the classifier
        self.classifier = None

        # Create a list of available classifiers with their corresponding hyperparameters
        # Create a list of available classifiers with their corresponding hyperparameters
        self.classifiers = {
            "Random Forest Classifier": {
                "model": RandomForestClassifier(n_estimators=10),
                "hyperparameters": {
                    "n_estimators": 10,
                    "max_depth": None,
                    "min_samples_split": 2,
                },
            },
            "Random Forest Regressor": {
                "model": RandomForestRegressor(n_estimators=10),
                "hyperparameters": {
                    "n_estimators": 10,
                    "max_depth": None,
                    "min_samples_split": 2,
                },
            },
            "Decision Tree Classifier": {
                "model": DecisionTreeClassifier(),
                "hyperparameters": {"max_depth": None, "min_samples_split": 2},
            },
            "Decision Tree Regressor": {
                "model": DecisionTreeRegressor(),
                "hyperparameters": {"max_depth": None, "min_samples_split": 2},
            },
            "SVM": {
                "model": SVC(kernel="linear"),
                "hyperparameters": {"kernel": "linear", "C": 1.0},
            },
            "K-Nearest Neighbors Classifier": {
                "model": KNeighborsClassifier(n_neighbors=5),
                "hyperparameters": {"n_neighbors": 5, "weights": "uniform"},
            },
            "K-Nearest Neighbors Regressor": {
                "model": KNeighborsRegressor(n_neighbors=5),
                "hyperparameters": {"n_neighbors": 5, "weights": "uniform"},
            },
            # Add more classifiers with their hyperparameters here
        }

        # for holding data
        self.data = None

        # for splittting data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # for breakinf down he data
        self.X = None
        self.y = None
        self.classifier = None

        # exporting arduino code data variable
        self.arduinoModel = None
        self.numValues = 1000

        # the model parameters
        self.model = None

        # Initialize label encoders
        self.label_encoders = {}

        self.create_menubar()
        self.create_gui()

    def create_menubar(self):
        menubar = tk.Menu(self.topLevel)
        self.topLevel.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="IMport CSV file", command=self.open_csv)
        file_menu.add_command(
            label="Data Statistics", command=self.show_data_statistics
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.topLevel.quit)

        # visualisations menus
        visualizations_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualizations", menu=visualizations_menu)
        visualizations_menu.add_command(label="Plot Data", command=self.plot_data)
        visualizations_menu.add_command(label="Plot Model", command=self.plot_model)

        # data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="Generate Synthetic Data", command=self.make_data)
        data_menu.add_command(
            label="Generate Ultrasonic Data", command=self.generate_ultrasonic_csv
        )
        data_menu.add_command(
            label="Generate Ultrasonic Data 2", command=self.generate_ultrasonic4_csv
        )

        data_menu.add_command(
            label="Generate Linear Regression Data(2)",
            command=self.doubleLinearRegressData,
        )
        data_menu.add_command(
            label="Generate Linear Regression Data(3)",
            command=self.tripleLinearRegressData,
        )

        # preparation menu
        preparation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Preparation", menu=preparation_menu)
        preparation_menu.add_command(
            label="Encode String Data", command=self.encode_data
        )
        preparation_menu.add_command(
            label="Decode String Data", command=self.decode_data
        )
        preparation_menu.add_separator()
        preparation_menu.add_command(
            label="Train-Test Split", command=self.train_test_split_data
        )
        preparation_menu.add_command(
            label="Show Train Data", command=self.show_train_data
        )
        preparation_menu.add_command(
            label="Show Test data", command=self.show_test_data
        )

        # model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        # Add a "Load Model" menu option
        model_menu.add_command(label="Load Model", command=self.load_model)
        # Add a "Load Label Encoders" menu option
        model_menu.add_command(
            label="Load Label Encoders(joblib)", command=self.load_label_encoders
        )
        model_menu.add_command(
            label="Load Label Encoders(Dict txt)",
            command=self.load_label_encoder_dict_txt,
        )

        model_menu.add_separator()
        model_menu.add_command(label="Train Model", command=self.train_model)
        model_menu.add_command(
            label="Set Model Parameters (Into SQL DB)",
            command=self.set_model_parameters,
        )

        # microntorllers
        export_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(
            label="Export Model for Arduino", command=self.convert_save_arduino_code
        )
        export_menu.add_command(
            label="Export Trained Model", command=self.convert_save_model
        )
        export_menu.add_command(
            label="Export Label Encoders", command=self.export_label_encoders
        )  # New menu option
        export_menu.add_command(
            label="Export Label Encoders(Dictionary)",
            command=self.export_label_encoder_dict_txt,
        )  # New menu option

    def create_gui(self):
        frame = tk.Frame(self.topLevel)
        frame.pack(padx=20, pady=20)

        title_label = tk.Label(
            frame, text="Machine Learning Trainer", font=("Helvetica", 16)
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Create a Combobox for selecting the classifier
        self.classifier_var = tk.StringVar()
        self.classifier_combobox = Combobox(
            self.topLevel, textvariable=self.classifier_var
        )
        self.classifier_combobox["values"] = list(self.classifiers.keys())
        self.classifier_combobox.current(0)  # Set the default classifier
        self.classifier_combobox.pack(padx=10, pady=5)

        # Create a button to open the prediction window
        predict_button = tk.Button( 
            self.topLevel, text="Make Predictions", command=self.open_prediction_window
        )
        predict_button.pack(padx=10, pady=5)

    def open_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                # Load the CSV data into a DataFrame
                self.data = pd.read_csv(file_path)

                # Display the data in a new window or widget
                self.display_csv_data(self.data)
            except Exception as e:
                messagebox.showerror("Error", f"Error loading CSV file: {str(e)}")

    def open_prediction_window(self):
        # Create a new window for predictions
        prediction_window = tk.Toplevel(self.topLevel)
        prediction_window.title("Prediction Window")
        prediction_window.geometry("400x300")

        # Create a frame for the input form
        input_frame = tk.Frame(prediction_window)
        input_frame.pack(padx=20, pady=10)

        input_label = tk.Label(input_frame, text="Enter Data for Prediction:")
        input_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.input_entries = {}
        
        # Create a subframe for input entries using grid
        input_entries_frame = tk.Frame(input_frame)
        input_entries_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(input_entries_frame, text=column)
            label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
            entry = tk.Entry(input_entries_frame)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.input_entries[column] = entry

        predict_button = tk.Button(input_frame, text="Predict", command=self.predict_data)
        predict_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # Adjust row and column weights to make the form expandable
        input_frame.grid_rowconfigure(1, weight=1)
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=1)


    
    
    def predict_data(self):
        if self.model and all(entry.get() for entry in self.input_entries.values()):
            # Get user input from the input form
            user_input = [float(entry.get()) for entry in self.input_entries.values()]

            # Use the trained model to make predictions
            prediction = self.model.predict([user_input])

            # Display the prediction
            messagebox.showinfo("Prediction Result", f"Predicted class: {prediction[0]}")
        else:
            messagebox.showwarning("Warning", "Model or input data is missing or incomplete. Load a model and enter data for prediction.")
    
    
    def show_data_statistics(self):
        if self.data is not None:
            try:
                # Compute basic statistics on self.data
                data_statistics = self.data.describe()

                # Display the statistics in a messagebox
                statistics_message = data_statistics.to_string()
                messagebox.showinfo("Data Statistics", statistics_message)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Error calculating data statistics: {str(e)}"
                )
        else:
            messagebox.showwarning(
                "Warning",
                "No data available for statistics. Generate or load data first.",
            )

    def display_csv_data(self, data):
        # Create a new window to display the CSV data
        csv_window = tk.Toplevel(self.topLevel)
        csv_window.title("CSV Data")
        csv_window.geometry("800x600")  # Set the dimensions as needed

        # Create a text widget to display the CSV data
        text_widget = tk.Text(csv_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Insert the data or  DataFrame's data into the text widget
        if isinstance(data, pd.DataFrame):
            # If it's a DataFrame, use to_string
            text_widget.insert(tk.END, data.to_string())
        elif isinstance(data, dict):
            # If it's a dictionary, convert to a string and insert
            data_str = "\n".join(f"{key}: {value}" for key, value in data.items())
            text_widget.insert(tk.END, data_str)
        else:
            text_widget.insert(tk.END, "Unsupported data type")

        # Add a scrollbar to the text widget
        scrollbar = tk.Scrollbar(text_widget)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

    def make_data(self):
        print("Making data is starting....")
        self.X, self.y = make_blobs(
            n_samples=100, centers=3, n_features=2, random_state=0
        )
        # print("Making data is done....")
        messagebox.showinfo("Info", "Data generation is complete.")

        # Check if self.data is already set and unset it
        if self.data is not None:
            self.data = None

        # Create a DataFrame from the X and y data
        self.data = pd.DataFrame({"X": self.X[:, 0], "y": self.X[:, 1]})
        self.display_csv_data(self.data)

    def generate_ultrasonic_csv(self):
        data = []
        data.append(["Distance", "Command"])
        for i in range(self.numValues):
            distance = Decimal(random.uniform(10, 50)).quantize(
                Decimal("0.00")
            )  # Random distance between 10 and 50 cm
            if distance < 20:
                command = "bad"
            elif distance < 29:
                command = "good"
            else:
                command = "safe"
            data.append([distance, command])

        # Create a DataFrame from the generated data
        self.data = pd.DataFrame(data[1:], columns=data[0])

        # Save the DataFrame as a CSV file
        self.data.to_csv("Ultrasonic Data.csv", index=False)

        # Display the data using the display_csv_data function
        self.display_csv_data(self.data)

        # Display the data using the display_csv_data function
        messagebox.showinfo(
            "Data Generation", "Ultrasonic data generation is complete."
        )
        print(f"CSV file '{'Ultrasonic Data.csv'}' generated successfully!")

    def generate_ultrasonic4_csv(self):
        data = []
        data.append(["Distance1", "Distance2", "Distance3", "Distance4", "Command"])

        def generate_command(distance):
            if distance < 20:
                return "bad"
            elif distance < 30:
                return "good"
            else:
                return "excellent"

        for i in range(self.numValues):
            distance1 = Decimal(random.uniform(10, 50)).quantize(Decimal("0.00"))
            distance2 = Decimal(random.uniform(10, 50)).quantize(Decimal("0.00"))
            distance3 = Decimal(random.uniform(10, 50)).quantize(Decimal("0.00"))
            distance4 = Decimal(random.uniform(10, 50)).quantize(Decimal("0.00"))
            command1 = generate_command(distance1)
            data.append([distance1, distance2, distance3, distance4, command1])

        # Create a DataFrame from the generated data
        self.data = pd.DataFrame(data[1:], columns=data[0])

        # Save the DataFrame as a CSV file
        self.data.to_csv("Ultrasonic Data.csv", index=False)

        # Display the data using the display_csv_data function
        self.display_csv_data(self.data)

        # Display the data using the display_csv_data function
        messagebox.showinfo(
            "Data Generation", "Ultrasonic data generation is complete."
        )
        print(f"CSV file '{'Ultrasonic Data.csv'}' generated successfully!")

    # Generate sample data
    def doubleLinearRegressData(self):
        np.random.seed(42)  # For reproducibility
        X = np.random.rand(100, 1) * 10
        y = 2 * X + 3 + np.random.randn(100, 1) * 2  # y = 2*X + 3 + noise
        self.data = pd.DataFrame({"X": X.flatten(), "Y": y.flatten()})
        self.data.to_csv("2 digit Linear.csv", index=False, float_format="%.2f")

        # Display the data using the display_csv_data function
        self.display_csv_data(self.data)
        messagebox.showinfo(
            "Info", "Two-column linear regression data generation is complete."
        )

    def tripleLinearRegressData(self):
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        X = np.random.rand(100, 3) * 10
        y = (
            X[:, 0] + 2 * X[:, 1] - 3 * X[:, 2] + np.random.randn(100) * 2
        )  # y = X1 + 2*X2 - 3*X3 + noise
        self.data = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2], "y": y})
        self.data.to_csv("3 digit Linear.csv", index=False, float_format="%.2f")

        # Display the data using the display_csv_data function
        self.display_csv_data(self.data)

        messagebox.showinfo(
            "Info", "Three-column linear regression data generation is complete."
        )

    def set_model_parameters(self):
        selected_classifier = self.classifier_combobox.get()
        if selected_classifier in self.classifiers:
            # Get the selected classifier's hyperparameters
            hyperparameters = self.classifiers[selected_classifier]["hyperparameters"]

            # Create a dialog box for setting hyperparameters
            param_window = tk.Toplevel(self.topLevel)
            param_window.title(f"Set {selected_classifier} Hyperparameters")

            # Create input fields for hyperparameters
            param_entries = {}
            for param, default_value in hyperparameters.items():
                param_label = tk.Label(param_window, text=param)
                param_label.pack(padx=10, pady=5)
                param_entries[param] = tk.Entry(param_window)
                param_entries[param].insert(0, str(default_value))
                param_entries[param].pack(padx=10, pady=5)

            def set_hyperparameters():
                # Retrieve hyperparameters from the input fields
                new_hyperparameters = {
                    param: float(param_entries[param].get())
                    for param in hyperparameters
                }

                # Update the selected classifier's hyperparameters
                self.classifiers[selected_classifier][
                    "hyperparameters"
                ] = new_hyperparameters

                # Close the parameter setting dialog
                param_window.destroy()

            # Create a button to confirm the hyperparameters
            param_button = tk.Button(
                param_window, text="Set Hyperparameters", command=set_hyperparameters
            )
            param_button.pack(padx=10, pady=10)

        else:
            messagebox.showerror("Error", "Invalid classifier selection.")

    def encode_data(self):
        if self.data is not None:
            data_updated = False
            for column in self.data.columns:
                if self.data[column].dtype == "object":
                    le = LabelEncoder()
                    self.data[column] = le.fit_transform(self.data[column])
                    self.label_encoders[column] = le
                    data_updated = True
            if data_updated:
                self.display_csv_data(self.data)
                self.display_csv_data(self.label_encoders)
                messagebox.showinfo("Info", "Data encoded successfully.")
            else:
                messagebox.showinfo("Info", "No string data found in the dataset.")

    def decode_data(self):
        if self.data is not None:
            data_updated = False
            for column, label_encoder in self.label_encoders.items():
                self.data[column] = label_encoder.inverse_transform(self.data[column])
                data_updated = True

            if data_updated:
                self.display_csv_data(self.data)
                messagebox.showinfo("Info", "Data decoded successfully.")
            else:
                messagebox.showinfo(
                    "Info", "No label encoders found. Data was not decoded."
                )

    def train_model(self):
        if self.X is not None and self.y is not None:
            selected_classifier = self.classifier_combobox.get()
            if selected_classifier in self.classifiers:
                # Get the selected classifier and its hyperparameters
                classifier_data = self.classifiers[selected_classifier]
                classifier_model = classifier_data["model"]
                hyperparameters = classifier_data["hyperparameters"]

                # Update the classifier model with the user-defined hyperparameters
                classifier_model.set_params(**hyperparameters)

                # Use the classifier with the updated hyperparameters
                self.classifier = classifier_model
                self.model = self.classifier.fit(self.X, self.y)
                print(
                    f"Training {selected_classifier} model is done with hyperparameters: {hyperparameters}"
                )

            else:
                messagebox.showerror("Error", "Invalid classifier selection.")
        else:
            print("No data to train on. Generate data first.")

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.joblib")])
        if model_path:
            try:
                loaded_model = joblib.load(model_path)
                if "model" in loaded_model:
                    self.model = loaded_model["model"]  # Store in self.model
                    print("Model loaded successfully from:", model_path)
                else:
                    print("No valid model found in the file.")
            except Exception as e:
                print(f"Error loading the model: {str(e)}")

    def load_label_encoders(self):
        encoder_path = filedialog.askopenfilename(
            filetypes=[("Label Encoder Files", "*.joblib")]
        )
        if encoder_path:
            try:
                loaded_encoders = joblib.load(encoder_path)
                if "label_encoders" in loaded_encoders:
                    self.label_encoders = loaded_encoders["label_encoders"]
                    print("Label encoders loaded successfully from:", encoder_path)
                else:
                    print("No valid label encoders found in the file.")
            except Exception as e:
                print(f"Error loading label encoders: {str(e)}")

    def load_label_encoder_dict_txt(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                loaded_encoders = {}
                with open(file_path, "r") as file:
                    lines = file.readlines()
                    i = 0
                    while i < len(lines):
                        label = lines[i].strip().replace("Label: ", "")
                        encoder = lines[i + 1].strip().replace("Encoder: ", "")
                        loaded_encoders[label] = encoder
                        i += 2
                self.label_encoders = loaded_encoders
                print("Label encoders loaded from text dictionary file:", file_path)
            except Exception as e:
                print(
                    f"Error loading label encoders from text dictionary file: {str(e)}"
                )

    def export_label_encoders(self):
        if self.label_encoders:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=[("Joblib Files", "*.joblib"), ("All files", "*.*")],
                title="Save Label Encoders",
            )
            if file_path:
                label_encoders = {"label_encoders": self.label_encoders}
                joblib.dump(label_encoders, file_path)
                messagebox.showinfo("Info", "Label encoders exported successfully.")
        else:
            messagebox.showwarning("Warning", "No label encoders to export.")

    def export_label_encoder_dict_txt(self):
        if self.label_encoders:
            # Save label encoders as a text file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                title="Save Label Encoders (Text File)",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            if file_path:
                with open(file_path, "w") as file:
                    for label, encoder in self.label_encoders.items():
                        file.write(f"Label: {label}\n")
                        file.write(f"Encoder: {encoder}\n")
                print("Label encoders saved as a text file to:", file_path)

    def convert_save_model(self):
        if self.model:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".joblib",
                filetypes=[("Joblib Files", "*.joblib"), ("All files", "*.*")],
                title="Save Trained Model",
            )
            if file_path:
                model_data = {"model": self.model}
                joblib.dump(model_data, file_path)
                messagebox.showinfo("Info", "Trained model exported successfully.")
        else:
            messagebox.showwarning("Warning", "No trained model to export.")

    def convert_save_arduino_code(self):
        if self.model:
            print("Converting initializing.....")
            self.arduinoModel = port(self.model)
            print(self.arduinoModel)
            print("Converting is done")

            if self.arduinoModel:
                model_path = filedialog.asksaveasfilename(
                    defaultextension=".h",
                    title="Save TinyML Model",
                    filetypes=[("TinyML files", "*.h"), ("All files", "*.*")],
                )
                if model_path:
                    with open(model_path, "w") as file:
                        file.write(self.arduinoModel)
                    print("Model saved to:", model_path)
            else:
                print("No model to save. Train and convert a model first.")
        else:
            print("No model to convert. Train a model first.")

    # Define methods for data splitting operations:
    def train_test_split_data(self):
        if self.data is not None:
            # Implement the train-test data splitting logic here
            self.X = self.data.iloc[
                :, :-1
            ]  # Features (all columns except the last one)
            self.y = self.data.iloc[:, -1]  # Target variable (last column)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2
            )
            # print("X_train:", self.X_train)
            # print("X_test:", self.X_test)
            print("y_train:", self.y_train)
            print("y_test:", self.y_test)
            print("Data is split into X and y")
        else:
            messagebox.showerror(
                "Error", "No data available for splitting. Generate or load data first."
            )

    # Define methods for showing data splitting operations:
    def show_train_data(self):
        if self.X_train is not None:
            self.display_csv_data(self.X_train)
            self.display_csv_data(self.y_train)
        else:
            messagebox.showerror("Error", "No y train data splitted.First split data.")

    # Define methods for showing data splitting operations:
    def show_test_data(self):
        if self.y_test is not None:
            self.display_csv_data(self.X_test)
            self.display_csv_data(self.y_test)
        else:
            messagebox.showerror("Error", "No y train data splitted.First split data.")

    def plot_data(self):
        if self.X is not None:
            # Create a small window to select the plot type
            plot_type_window = tk.Toplevel(self.topLevel)
            plot_type_window.title("Select Plot Type")

            # Determine the number of columns in the data
            num_columns = len(self.data.columns)

            # Create checkboxes for selecting columns to be plotted
            checkbox_vars = [tk.IntVar() for _ in range(num_columns)]
            for i in range(self.data.columns):
                checkbox = tk.Checkbutton(
                    plot_type_window,
                    text=f"Column {i + 1}",
                    variable=checkbox_vars[i],
                    onvalue=1,
                    offvalue=0,
                )
                checkbox.pack(padx=10, pady=5)

            # Create a label for the dropdown menu
            label = tk.Label(plot_type_window, text="Select Plot Type:")
            label.pack(padx=10, pady=5)

            # Create a Combobox for selecting the plot type
            plot_type_var = tk.StringVar()
            plot_type_combobox = Combobox(plot_type_window, textvariable=plot_type_var)
            plot_type_combobox["values"] = ["Scatter 2D", "Scatter 3D", "Histogram"]
            plot_type_combobox.current(0)  # Set the default value
            plot_type_combobox.pack(padx=10, pady=5)

            def plot_graph():
                selected_columns = [
                    i + 1 for i, var in enumerate(checkbox_vars) if var.get() == 1
                ]
                if not selected_columns:
                    messagebox.showerror("Error", "Please select at least one column.")
                else:
                    plot_type = plot_type_var.get()
                    if plot_type == "Scatter 2D" and len(selected_columns) == 2:
                        self.plot_scatter_2d(selected_columns)
                    elif plot_type == "Scatter 3D" and len(selected_columns) == 3:
                        self.plot_scatter_3d(selected_columns)
                    elif plot_type == "Histogram" and len(selected_columns) == 1:
                        self.plot_histogram(selected_columns[0] - 1)
                    else:
                        messagebox.showerror("Error", "Invalid selection.")
                    plot_type_window.destroy()

            # Create a button to confirm the plot type and columns selection
            plot_button = tk.Button(plot_type_window, text="Plot", command=plot_graph)
            plot_button.pack(padx=10, pady=10)

        else:
            messagebox.showerror(
                "Error", "No data to plot. Generate or load data first."
            )

    def plot_scatter_2d(self, selected_columns):
        if self.X is not None:
            fig, ax = plt.subplots()
            ax.scatter(
                self.X[:, selected_columns[0] - 1],
                self.X[:, selected_columns[1] - 1],
                c=self.y,
            )
            ax.set_xlabel(f"Column {selected_columns[0]}")
            ax.set_ylabel(f"Column {selected_columns[1]}")
            ax.set_title("Scatter Plot - Model Data (2D)")

            plt.show()

    def plot_scatter_3d(self, selected_columns):
        if self.X is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                self.X[:, selected_columns[0] - 1],
                self.X[:, selected_columns[1] - 1],
                self.X[:, selected_columns[2] - 1],
                c=self.y,
            )
            ax.set_xlabel(f"Column {selected_columns[0]}")
            ax.set_ylabel(f"Column {selected_columns[1]}")
            ax.set_zlabel(f"Column {selected_columns[2]}")
            ax.set_title("Scatter Plot - Model Data (3D)")

            plt.show()

    def plot_histogram(self, selected_column):
        if self.X is not None:
            fig, ax = plt.subplots()
            ax.hist(
                self.X[:, selected_column],
                bins=10,
                alpha=0.5,
                label=f"Column {selected_column + 1}",
            )
            ax.set_xlabel(f"Values")
            ax.set_ylabel(f"Frequency")
            ax.set_title("Histogram - Model Data")
            ax.legend()
            plt.show()

    def plot_model(self):
        messagebox.showinfo("Info", "Plot Model function not implemented yet.")

    def run(self):
        self.topLevel.mainloop()


if __name__ == "__main__":
    ml_gui = MLGui()
    ml_gui.run()
