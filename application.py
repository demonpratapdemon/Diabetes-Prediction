from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

scaler = pickle.load(open("models/standards_scaler_model.pkl", "rb"))
lr_model = pickle.load(open("models/LogisticRegressionModel.pkl", "rb"))


@app.route("/predictData", methods=["GET", "POST"])
def predict_datapoint():
    result = ""
    if request.method == "POST":
        Pregnancies = float(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))

        scaled_data = scaler.transform(
            [
                [
                    Pregnancies,
                    Glucose,
                    BloodPressure,
                    SkinThickness,
                    Insulin,
                    BMI,
                    DiabetesPedigreeFunction,
                    Age,
                ]
            ]
        )

        predict = lr_model.predict(scaled_data)
        if predict[0] == 1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"
        return render_template("home.html", results=result)
    else:
        return render_template("home.html")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
