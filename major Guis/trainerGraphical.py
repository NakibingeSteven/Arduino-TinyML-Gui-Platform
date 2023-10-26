import tkinter as tk
from tkinter import filedialog
from sklearn.datasets import make_blobs
from everywhereml.sklearn.ensemble import RandomForestClassifier
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
# global variables
global clf
global X, y


class mlGui:
    def __init__(self):
        # the train values
        self.data = None
        self.X = None
        self.y = None
        self.classifier = None

        # the button
        self.make_btn = None
        self.classifier_btn = None
        self.train_btn = None
        self.converter_btn = None
        self.save_Button = None
        self.arduinoModel = None

    #load data
    def load_data(self):
        file_path = askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        # Load the data from CSV file
        self.data = pd.read_csv(file_path)

    # make button functions
    # make data function
    def makeData(self):
        print("Making data is starting....")
        self.X, self.y = make_blobs(
            n_samples=100, centers=3, n_features=2, random_state=0
        )
        print("Making data is done....")
        print(self.X)

    # random classifier function
    def randomClasser(self):
        self.classifier = RandomForestClassifier(n_estimators=10)
        print("Data is made")

    # the trainModel function
    def trainModel(self):
        print("Training is taking place .... ")
        self.classifier.fit(self.X, self.y)
        print("Training is done")

    # convert to arduino button
    def convert(self):
        print("Converting initialising.....")
        self.arduinoModel = self.classifier.to_arduino(instance_name="blobClassifier")
        print(self.arduinoModel)
        print("Converting  is done")

    # store model
    def storeModel(self):
        model_path = filedialog.asksaveasfilename(
            defaultextension=".h",
            title="Save tinyML Model",
            filetypes=[("TinyMl files", "*.h"), ("All files", "")],
        )
        if model_path:
            file = open(model_path, "w", newline="")
            file.write(self.arduinoModel)
            file.close()

    def createGui(self):
        # the container
        self.topLevel = tk.Tk()
        self.topLevel.geometry("400x300")
        self.topLevel.title("TrainerGraphical Gui")
        # make data
        self.make_btn = tk.Button(
            self.topLevel, text="Make Data Blob", command=self.makeData
        )
        self.make_btn.pack()

        # classifier
        self.classifier_btn = tk.Button(
            self.topLevel, text="Classifier", command=self.randomClasser
        )
        self.classifier_btn.pack()

        # train model
        self.train_btn = tk.Button(
            self.topLevel, text="Train Data", command=self.trainModel
        )
        self.train_btn.pack() 

        # convert model
        self.convert_btn = tk.Button(
            self.topLevel, text="Convert for Arduino", command=self.convert
        )
        self.convert_btn.pack()

        # save button
        self.save_btn = tk.Button(
            self.topLevel, text="Save Button", command=self.storeModel
        )
        self.save_btn.pack()

        # pack in container
        self.topLevel.mainloop()


tinyGui = mlGui()
tinyGui.createGui()
