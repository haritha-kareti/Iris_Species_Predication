from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report

app = Flask(__name__)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
with open("KNN_Model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("NB_Model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# --------------------------------------------------
# Load TRAIN data
# --------------------------------------------------
with open("train_data.json", "r") as f:
    train_data = json.load(f)

X_train = np.array(train_data["X_train"])
y_train = np.array(train_data["y_train"])

with open("train_data1.json", "r") as f:
    train_data1 = json.load(f)

X_train = np.array(train_data1["X_train"])
y_train = np.array(train_data1["y_train"])

# --------------------------------------------------
# Load TEST data
# --------------------------------------------------
with open("test_data.json", "r") as f:
    test_data = json.load(f)

X_test = np.array(test_data["X_test"])
y_test = np.array(test_data["y_test"])

with open("test_data1.json", "r") as f:
    test_data1 = json.load(f)

X_test = np.array(test_data1["X_test"])
y_test = np.array(test_data1["y_test"])

# --------------------------------------------------
# Class Labels
# --------------------------------------------------
labels = ["Setosa", "Versicolor", "Virginica"]

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Input from UI
    input_features = np.array([[
        float(data["sepal_length"]),
        float(data["sepal_width"]),
        float(data["petal_length"]),
        float(data["petal_width"])
    ]])

    # --------------------------------------------------
    # Model Selection from UI
    # --------------------------------------------------
    model_name = data.get("model", "knn")

    if model_name == "nb":
        model = nb_model
        model_used = "Naive Bayes"
    else:
        model = knn_model
        model_used = "KNN"

    # --------------------------------------------------
    # Predict single input
    # --------------------------------------------------
    pred_class = model.predict(input_features)[0]
    prediction = labels[int(pred_class)]

    # --------------------------------------------------
    # Evaluate model on TEST data
    # --------------------------------------------------
    y_pred_test = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_test).tolist()
    cr = classification_report(
        y_test,
        y_pred_test,
        target_names=labels
    )

    return jsonify({
        "prediction": prediction,
        "model_used": model_used,
        "confusion_matrix": cm,
        "classification_report": cr
    })


# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
