import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Style, Treeview
from sklearn.datasets import make_blobs
from everywhereml.sklearn.ensemble import RandomForestClassifier
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


class MLGui:
    def __init__(self):
        self.topLevel = tk.Tk()
        self.topLevel.geometry("800x600")
        self.topLevel.title("Machine Learning Trainer")

        style = Style()
        style.configure("TButton", font=("Helvetica", 12))

        #for holding data
        self.data = None

        #for splittting data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test =None

        #for breakinf down he data
        self.X = None
        self.y = None
        self.classifier = None

        #exporting arduino code data variable
        self.arduinoModel = None
        self.numValues = 1000;

        self.create_menubar()
        self.create_gui()

    def create_menubar(self):
        menubar = tk.Menu(self.topLevel)
        self.topLevel.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="IMport CSV file", command=self.open_csv)
        file_menu.add_command(label="Data Statistics", command=self.show_data_statistics)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.topLevel.quit)

        #visualisations menus
        visualizations_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualizations", menu=visualizations_menu)
        visualizations_menu.add_command(label="Plot Data", command=self.plot_data)
        visualizations_menu.add_command(label="Plot Model", command=self.plot_model)
        visualizations_menu.add_command(label="Plot Data vs Model", command=self.plot_data_vs_model)

        #data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="Generate Synthetic Data", command=self.make_data)
        data_menu.add_command(label="Generate Ultrasonic Data", command=self.generate_ultrasonic_csv)
        data_menu.add_command(label="Generate Linear Regression Data(2)", command=self.doubleLinearRegressData)
        data_menu.add_command(label="Generate Linear Regression Data(3)", command=self.tripleLinearRegressData)

        #preparation menu
        preparation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Preparation", menu=preparation_menu)
         preparation_menu.add_command(label="Encode String Data")
        preparation_menu.add_command(label="Train-Test Split", command=self.train_test_split_data)
        preparation_menu.add_command(label="Show Train Data", command=self.show_train_data)
        preparation_menu.add_command(label="Show Test data", command=self.show_test_data)



        #model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Train Model", command=self.train_model)
        model_menu.add_command(label="Select Model", command=self.select_model)
        model_menu.add_command(label="Set Model Parameters (Into SQL DB)", command=self.set_model_parameters)

        #microntorllers
        export_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export Model for Arduino", command=self.convert)
        export_menu.add_command(label="Export Model and labels", )
        export_menu.add_command(label="Export Model Only",)
        export_menu.add_command(label="Save Model", command=self.store_model)

    def create_gui(self):
        frame = tk.Frame(self.topLevel)
        frame.pack(padx=20, pady=20)

        title_label = tk.Label(
            frame, text="Machine Learning Trainer", font=("Helvetica", 16)
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

    
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

    def show_data_statistics(self):
        if self.data is not None:
            try:
                # Compute basic statistics on self.data
                data_statistics = self.data.describe()

                # Display the statistics in a messagebox
                statistics_message = data_statistics.to_string()
                messagebox.showinfo("Data Statistics", statistics_message)
            except Exception as e:
                messagebox.showerror("Error", f"Error calculating data statistics: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No data available for statistics. Generate or load data first.")


    def display_csv_data(self, dataframe):
        # Create a new window to display the CSV data
        csv_window = tk.Toplevel(self.topLevel)
        csv_window.title("CSV Data")
        csv_window.geometry("800x600")  # Set the dimensions as needed

        # Create a text widget to display the CSV data
        text_widget = tk.Text(csv_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Insert the DataFrame's data into the text widget
        text_widget.insert(tk.END, dataframe.to_string())

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
        self.data = pd.DataFrame({'X': self.X[:, 0], 'y': self.X[:, 1]})
        self.display_csv_data(self.data)

    def generate_ultrasonic_csv(self):
        data = []
        data.append(["Distance", "Command"])
        for i in range(self.numValues):
            distance = Decimal(random.uniform(10, 50)).quantize(Decimal("0.00"))  # Random distance between 10 and 50 cm
            if distance < 20:
                command = "bad"
            elif distance < 29:
                command = "good"
            else:
                command = "safe"
            data.append([distance, command])
            
         # Create a DataFrame from the generated data
        self.data= pd.DataFrame(data[1:], columns=data[0])

        # Save the DataFrame as a CSV file
        self.data.to_csv("Ultrasonic Data.csv", index=False)

        # Display the data using the display_csv_data function
        self.display_csv_data(self.data)

        # Display the data using the display_csv_data function
        messagebox.showinfo("Data Generation", "Ultrasonic data generation is complete.")
        print(f"CSV file '{'Ultrasonic Data.csv'}' generated successfully!")

    # Generate sample data
    def doubleLinearRegressData(self):
        np.random.seed(42)  # For reproducibility
        X = np.random.rand(100, 1) * 10
        y = 2 * X + 3 + np.random.randn(100, 1) * 2  # y = 2*X + 3 + noise
        self.data = pd.DataFrame({"X": X.flatten(), "Y": y.flatten()})
        self.data.to_csv("2 digit Linear.csv", index=False, float_format='%.2f')

        # Display the data using the display_csv_data function
        self.display_csv_data(self.data)
        messagebox.showinfo("Info", "Two-column linear regression data generation is complete.")


    def tripleLinearRegressData(self):
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        X = np.random.rand(100, 3) * 10
        y = (X[:, 0] + 2 * X[:, 1] - 3 * X[:, 2] + np.random.randn(100) * 2)  # y = X1 + 2*X2 - 3*X3 + noise
        self.data = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2], "y": y})
        self.data.to_csv("3 digit Linear.csv", index=False, float_format='%.2f')

        # Display the data using the display_csv_data function
        self.display_csv_data(self.data)

        messagebox.showinfo("Info", "Three-column linear regression data generation is complete.")

    def train_model(self):
        if self.X is not None and self.y is not None:
            print("Training is taking place .... ")
            self.classifier = RandomForestClassifier(n_estimators=10)
            self.classifier.fit(self.X, self.y)
            print("Training is done")
        else:
            print("No data to train on. Generate data first.")

    def convert(self):
        if self.classifier:
            print("Converting initializing.....")
            self.arduinoModel = self.classifier.to_arduino(
                instance_name="blobClassifier"
            )
            print(self.arduinoModel)
            print("Converting is done")
        else:
            print("No model to convert. Train a model first.")

    def store_model(self):
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
    
    # Define methods for data splitting operations:
    def train_test_split_data(self):
        if self.data is not None:
            # Implement the train-test data splitting logic here
            self.X = self.data.iloc[:, :-1]  # Features (all columns except the last one)
            self.y = self.data.iloc[:, -1]  # Target variable (last column)
            self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=0.2)
            #print("X_train:", self.X_train)
            #print("X_test:", self.X_test)
            #print("y_train:",self.y_train)
            #print("y_test:",self.y_test)
        else:
             messagebox.showerror("Error", "No data available for splitting. Generate or load data first.")
    
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
                selected_columns = [i + 1 for i, var in enumerate(checkbox_vars) if var.get() == 1]
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
            messagebox.showerror("Error", "No data to plot. Generate or load data first.")

    def plot_scatter_2d(self, selected_columns):
        if self.X is not None:
            fig, ax = plt.subplots()
            ax.scatter(self.X[:, selected_columns[0] - 1], self.X[:, selected_columns[1] - 1], c=self.y)
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
            ax.hist(self.X[:, selected_column], bins=10, alpha=0.5, label=f"Column {selected_column + 1}")
            ax.set_xlabel(f"Values")
            ax.set_ylabel(f"Frequency")
            ax.set_title("Histogram - Model Data")
            ax.legend()
            plt.show()
    
    def plot_model(self):
        messagebox.showinfo("Info", "Plot Model function not implemented yet.")

    def plot_data_vs_model(self):
        messagebox.showinfo("Info", "Plot Data vs Model function not implemented yet.")

    def select_model(self):
        messagebox.showinfo("Info", "Select Model function not implemented yet.")

    def set_model_parameters(self):
        messagebox.showinfo("Info", "Set Model Parameters (Into SQL DB) function not implemented yet.")


    def run(self):
        self.topLevel.mainloop()


if __name__ == "__main__":
    ml_gui = MLGui()
    ml_gui.run()
