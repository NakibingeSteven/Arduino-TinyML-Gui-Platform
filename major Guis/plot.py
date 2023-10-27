def plot_model(self):
    if self.classifier:
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
        messagebox.showerror("Error", "No model to plot. Train a model first")

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