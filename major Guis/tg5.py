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


class MLGui:
    def __init__(self):
        self.topLevel = tk.Tk()
        self.topLevel.geometry("800x600")
        self.topLevel.title("Machine Learning Trainer")

        style = Style()
        style.configure("TButton", font=("Helvetica", 12))

        #for holding data
        self.data = None

        #for breaking down data
        self.X = None
        self.y = None
        self.classifier = None
        self.arduinoModel = None

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
        data_menu.add_command(label="Generate Ultrasonic Data", command=self.generate_ultrasonic_data)
        data_menu.add_command(label="Generate Two Column Number Data", command=self.generate_two_column_data)
        data_menu.add_command(label="Generate Three Column Number Data", command=self.generate_three_column_data)

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

        # Create a frame to display data
        data_frame = tk.Frame(frame)
        data_frame.grid(row=1, column=0, columnspan=2)

        self.data_tree = Treeview(data_frame, columns=("X", "y"), show="headings")
        self.data_tree.heading("X", text="X")
        self.data_tree.heading("y", text="y")
        self.data_tree.pack(padx=20, pady=20)
    
    def open_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                # Load the CSV data into a DataFrame
                df = pd.read_csv(file_path)

                # Display the data in the Treeview widget
                self.display_data()

                # Display the data in a new window or widget
                self.display_csv_data(df)
            except Exception as e:
                messagebox.showerror("Error", f"Error loading CSV file: {str(e)}")

    def show_data_statistics(self):
        if self.X is not None and self.y is not None:
            # Compute basic statistics on self.X and self.y
            x_mean = self.X.mean()
            y_mean = self.y.mean()
            x_std = self.X.std()
            y_std = self.y.std()
            
            # Display the statistics in a messagebox
            statistics_message = f"X Mean: {x_mean}\nY Mean: {y_mean}\nX Std: {x_std}\nY Std: {y_std}"
            messagebox.showinfo("Data Statistics", statistics_message)

    
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
        print("Making data is done....")
        self.display_data()

    def display_data(self):
        # Clear any existing data
        for row in self.data_tree.get_children():
            self.data_tree.delete(row)
        
        #if self.data is defined but not x and y then put row in tree
        if self.data is not None:
            # If there are more than 2 columns, display each column as separate values
            for index, row in self.data.iterrows():
                self.data_tree.insert("", "end", values=row)

        else:
            if self.X is not None and self.y is not None:
                for x_val, y_val in zip(self.X, self.y):
                    self.data_tree.insert("", "end", values=(x_val, y_val))

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
    
    def plot_data(self):
        if self.X is not None:
            # Create a small window to select the plot type
            plot_type_window = tk.Toplevel(self.topLevel)
            plot_type_window.title("Select Plot Type")

            # Determine the number of columns in the data
            num_columns = 2  # Default to 2
            if self.X is not None:
                num_columns = self.X.shape[1]

            # Create checkboxes for selecting columns to be plotted
            checkbox_vars = [tk.IntVar() for _ in range(num_columns)]
            for i in range(num_columns):
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

    def generate_ultrasonic_data(self):
        messagebox.showinfo("Info", "Generate Ultrasonic Data function not implemented yet.")

    def generate_two_column_data(self):
        messagebox.showinfo("Info", "Generate Two Column Number Data function not implemented yet.")

    def generate_three_column_data(self):
        messagebox.showinfo("Info", "Generate Three Column Number Data function not implemented yet.")

    def select_model(self):
        messagebox.showinfo("Info", "Select Model function not implemented yet.")

    def set_model_parameters(self):
        messagebox.showinfo("Info", "Set Model Parameters (Into SQL DB) function not implemented yet.")


    def run(self):
        self.topLevel.mainloop()


if __name__ == "__main__":
    ml_gui = MLGui()
    ml_gui.run()
