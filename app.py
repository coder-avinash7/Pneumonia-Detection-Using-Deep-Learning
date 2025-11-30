import os
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19


# ------------------------------
# Load Model
# ------------------------------
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
x = base_model.output

flat = Flatten()(x)
dense1 = Dense(4608, activation='relu')(flat)
dropout = Dropout(0.2)(dense1)
dense2 = Dense(1152, activation='relu')(dropout)
output = Dense(2, activation='softmax')(dense2)

model_03 = Model(base_model.inputs, output)
model_03.load_weights("model_weights/vgg19_model_01.weights.h5")

print("Model loaded successfully.")


# ------------------------------
# Flask App
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------------------
# Helper Functions
# ------------------------------
def get_className(classNo):
    return "Normal" if classNo == 0 else "Pneumonia"


def get_result(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image)
    image = image.resize((128, 128))

    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    result = model_03.predict(image)
    result_index = np.argmax(result, axis=1)

    return int(result_index[0])


# ------------------------------
# Routes
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        pred = get_result(file_path)
        result = get_className(pred)

        return render_template(
            "index.html",
            filename=filename,
            prediction=result
        )

    return render_template("index.html")


# ------------------------------
# Run Flask App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
