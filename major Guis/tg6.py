import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Style, Treeview
from sklearn.datasets import make_blobs
from everywhereml.sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        file_menu.add_command(label="Open Data", command=self.load_data)
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

        title_label = tk.Label(frame, text="Machine Learning Trainer", font=("Helvetica", 16))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Create a frame to display data
        data_frame = tk.Frame(frame)
        data_frame.grid(row=1, column=0, columnspan=2)

        self.data_tree = Treeview(data_frame, columns=("X", "y"), show="headings")
        self.data_tree.heading("X", text="X")
        self.data_tree.heading("y", text="y")
        self.data_tree.pack(padx=20, pady=20)

        # Create a frame for the scatter plot
        scatter_frame = tk.Frame(data_frame)
        scatter_frame.pack(padx=20, pady=20)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=scatter_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        # Implement this function
        pass

    def make_data(self):
        print("Making data is starting....")
        self.X, self.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
        print("Making data is done....")
        self.display_data()
        self.display_scatter_plot()

    def display_data(self):
        # Clear any existing data
        for row in self.data_tree.get_children():
            self.data_tree.delete(row)

        if self.X is not None and self.y is not None:
            for x_val, y_val in zip(self.X, self.y):
                self.data_tree.insert("", "end", values=(x_val, y_val))

    def display_scatter_plot(self):
        if self.X is not None and self.y is not None:
            self.ax.clear()
            self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
            self.ax.set_xlabel("X-axis")
            self.ax.set_ylabel("Y-axis")
            self.ax.set_title("Scatter Plot")
            self.canvas.draw()

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
            self.arduinoModel = self.classifier.to_arduino(instance_name="blobClassifier")
            print(self.arduinoModel)
            print("Converting is done")
        else:
            print("No model to convert. Train a model first.")

    def store_model(self):
        if self.arduinoModel:
            model_path = filedialog.asksaveasfilename(
                defaultextension=".h",
                title="Save TinyML Model",
                filetypes=[("TinyML files", "*.h"), ("All files", "*.*")]
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
