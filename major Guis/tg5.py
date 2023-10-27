import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Style, Treeview
from sklearn.datasets import make_blobs
from everywhereml.sklearn.ensemble import RandomForestClassifier
import tkinter.messagebox as messagebox
import pandas as pd
class MLGui:
    def __init__(self):
        self.topLevel = tk.Tk()
        self.topLevel.geometry("800x600")
        self.topLevel.title("Machine Learning Trainer")

        style = Style()
        style.configure("TButton", font=("Helvetica", 12))

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
        file_menu.add_command(label="Open CSV", command=self.open_csv)
        file_menu.add_command(label="Data Statistics", command=self.show_data_statistics)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.topLevel.quit)

        action_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Actions", menu=action_menu)
        action_menu.add_command(label="Generate Synthetic Data", command=self.make_data)
        action_menu.add_command(label="Train Model", command=self.train_model)
        action_menu.add_command(label="Convert for Arduino", command=self.convert)
        action_menu.add_command(label="Save Model", command=self.store_model)

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
    def load_data(self):
        # Implement this function
        pass

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

    def run(self):
        self.topLevel.mainloop()


if __name__ == "__main__":
    ml_gui = MLGui()
    ml_gui.run()
