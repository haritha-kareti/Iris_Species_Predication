from flask import Flask, render_template, request
import numpy as np
import pickle
import json

app = Flask(__name__)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
with open("KNN_Model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("NB_Model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# --------------------------------------------------
# Load Performance JSON Files
# --------------------------------------------------
def load_json(file):
    with open(file, "r") as f:
        return json.load(f)

train_knn = load_json("train_data_KNN.json")
test_knn = load_json("test_data_KNN.json")

train_nb = load_json("train_data_NB.json")
test_nb = load_json("test_data_NB.json")


# --------------------------------------------------
# Home Route
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction Route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    model_choice = request.form["model"]

    features = [
        float(request.form["feature1"]),
        float(request.form["feature2"]),
        float(request.form["feature3"]),
        float(request.form["feature4"])
    ]

    input_data = np.array([features])

    # Model Selection
    if model_choice == "knn":
        model = knn_model
        train_data = train_knn
        test_data = test_knn

    elif model_choice == "nb":
        model = nb_model
        train_data = train_nb
        test_data = test_nb

    else:
        return "Invalid Model Selected"

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        model_name=train_data["model_name"],

        # TRAIN DATA
        train_accuracy=train_data["accuracy"],
        train_confusion=train_data["confusion_matrix"],
        train_report=train_data["classification_report"],

        # TEST DATA
        test_accuracy=test_data["accuracy"],
        test_confusion=test_data["confusion_matrix"],
        test_report=test_data["classification_report"]
    )


# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
