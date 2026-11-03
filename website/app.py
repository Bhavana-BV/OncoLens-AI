from flask import Flask, render_template, request, jsonify
from pathlib import Path
from uuid import uuid4
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

app = Flask(__name__)

# Automatically locate the OncoLens_GitHub repository root
ROOT = Path(__file__).resolve().parent.parent

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# All trained models are stored in the repository's models folder
MODEL_PATHS = {
    "MobileNetV2": ROOT / "models" / "mobilenetv2_sharpen.keras",
    "EfficientNetB0": ROOT / "models" / "efficientnetb0_sharpen.keras",
    "ResNet50": ROOT / "models" / "resnet50_best.keras"
}

GRADCAM_CONFIG = {
    "MobileNetV2": {
        "backbone": "mobilenetv2_1.00_224",
        "target_layer": "Conv_1"
    },
    "EfficientNetB0": {
        "backbone": "efficientnetb0",
        "target_layer": "top_conv"
    },
    "ResNet50": {
        "backbone": "resnet50",
        "target_layer": "conv5_block3_3_conv"
    }
}

GRADCAM_DIR = Path(app.static_folder) / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

loaded_models = {}

# Temporary storage for uploaded images.
# Suitable for local demonstration; not persistent.
image_sessions = {}


# ============================================================
# VALIDATION
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_image_quality(image):

    width, height = image.size

    if min(width, height) < 160:
        return False, "Image resolution is too small."

    gray = np.array(image.convert("L"))

    brightness = float(np.mean(gray))

    if brightness < 10 or brightness > 245:
        return False, "Image lighting is unsuitable."

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score < 5:
        return False, "Image appears too blurry."

    return True, "Image quality accepted."


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_sharpened_image(image):

    image = image.convert("RGB")
    array = np.array(image)

    resized = cv2.resize(
        array,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    blurred = cv2.GaussianBlur(
        resized,
        (0, 0),
        sigmaX=1
    )

    sharpened = cv2.addWeighted(
        resized,
        1.5,
        blurred,
        -0.5,
        0
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)

    return np.expand_dims(sharpened, axis=0)


def preprocess_resnet_image(image):

    image = image.convert("RGB")
    array = np.array(image)

    resized = tf.image.resize(
        array,
        (224, 224),
        antialias=True
    )

    resized = tf.cast(
        resized,
        tf.float32
    )

    return tf.expand_dims(resized, axis=0)


def preprocess_gradcam_input(image, model_name):

    if model_name in ["MobileNetV2", "EfficientNetB0"]:
        return preprocess_sharpened_image(image)

    if model_name == "ResNet50":
        return preprocess_resnet_image(image)

    raise ValueError("Unknown model.")


# ============================================================
# MODEL LOADING
# ============================================================

def load_selected_model(model_name):

    if model_name in loaded_models:
        return loaded_models[model_name]

    if model_name not in MODEL_PATHS:
        raise ValueError("Unknown model.")

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print(f"\nLoading {model_name}...")

    start = time.time()

    if model_name == "ResNet50":

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                "preprocess_input": preprocess_input
            }
        )

    else:

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    loaded_models[model_name] = model

    print(
        f"{model_name} loaded in "
        f"{time.time() - start:.2f} seconds."
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image, model_name):

    model = load_selected_model(model_name)

    if model_name == "ResNet50":

        input_tensor = preprocess_resnet_image(image)

        output = model.predict(
            input_tensor,
            verbose=0
        )

        cancer_probability = float(
            np.asarray(output).reshape(-1)[0]
        )

    else:

        input_tensor = preprocess_sharpened_image(image)

        output = model.predict(
            input_tensor,
            verbose=0
        )

        cancer_probability = float(output[0][1])

    cancer_probability = float(
        np.clip(cancer_probability, 0, 1)
    )

    non_cancer_probability = 1 - cancer_probability

    prediction = (
        "Cancer"
        if cancer_probability >= 0.5
        else "Non-Cancer"
    )

    confidence = max(
        cancer_probability,
        non_cancer_probability
    )

    return {
        "model": model_name,
        "prediction": prediction,
        "cancer_probability": round(
            cancer_probability * 100, 2
        ),
        "non_cancer_probability": round(
            non_cancer_probability * 100, 2
        ),
        "confidence": round(
            confidence * 100, 2
        ),
        "threshold": 0.5
    }


# ============================================================
# COMPARISON
# ============================================================

def compare_predictions(results):

    cancer_votes = sum(
        1 for result in results
        if result["prediction"] == "Cancer"
    )

    non_cancer_votes = len(results) - cancer_votes

    probabilities = [
        result["cancer_probability"]
        for result in results
    ]

    majority_prediction = (
        "Cancer"
        if cancer_votes > non_cancer_votes
        else "Non-Cancer"
    )

    unanimous = (
        cancer_votes == len(results)
        or non_cancer_votes == len(results)
    )

    return {
        "majority_prediction": majority_prediction,
        "agreement": (
            "Unanimous agreement"
            if unanimous
            else "Models disagree"
        ),
        "average_cancer_probability": round(
            float(np.mean(probabilities)), 2
        ),
        "probability_range": round(
            float(max(probabilities) - min(probabilities)), 2
        ),
        "cancer_votes": cancer_votes,
        "non_cancer_votes": non_cancer_votes
    }


# ============================================================
# GRAD-CAM HELPERS
# ============================================================

def find_layer_recursive(model, target_name):

    for layer in model.layers:

        if layer.name == target_name:
            return layer

        if isinstance(layer, tf.keras.Model):

            found = find_layer_recursive(
                layer,
                target_name
            )

            if found is not None:
                return found

    return None


def generate_gradcam(
    model,
    backbone,
    target_layer,
    input_tensor,
    model_name
):

    backbone_node = backbone._inbound_nodes[-1]

    backbone_inputs = tf.nest.flatten(
        backbone_node.input_tensors
    )

    if len(backbone_inputs) != 1:
        raise ValueError("Expected one backbone input.")

    backbone_call_input = backbone_inputs[0]

    preprocessing_model = Model(
        inputs=model.inputs,
        outputs=backbone_call_input
    )

    feature_model = Model(
        inputs=backbone.inputs,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    backbone_index = model.layers.index(backbone)

    head_layers = model.layers[backbone_index + 1:]

    with tf.GradientTape() as tape:

        processed_input = preprocessing_model(
            input_tensor,
            training=False
        )

        conv_outputs, backbone_features = feature_model(
            processed_input,
            training=False
        )

        predictions = backbone_features

        for layer in head_layers:

            predictions = layer(
                predictions,
                training=False
            )

        if model_name == "ResNet50":
            target_score = predictions[:, 0]
        else:
            target_score = predictions[:, 1]

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError("Gradients could not be calculated.")

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    heatmap = tf.nn.relu(heatmap)

    maximum = tf.reduce_max(
        heatmap,
        axis=(1, 2),
        keepdims=True
    )

    heatmap = tf.math.divide_no_nan(
        heatmap,
        maximum
    )

    return heatmap[0].numpy()


def create_gradcam_overlay(image, heatmap):

    original = image.convert("RGB").resize((224, 224))

    original_array = np.array(original)

    resized_heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(
        255 * resized_heatmap
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_array,
        0.6,
        colored_heatmap,
        0.4,
        0
    )

    return Image.fromarray(overlay)


def generate_model_gradcam(image, model_name, session_id):

    model = load_selected_model(model_name)

    config = GRADCAM_CONFIG[model_name]

    backbone = find_layer_recursive(
        model,
        config["backbone"]
    )

    if backbone is None:
        raise ValueError("Backbone not found.")

    target_layer = find_layer_recursive(
        backbone,
        config["target_layer"]
    )

    if target_layer is None:
        raise ValueError("Grad-CAM layer not found.")

    input_tensor = preprocess_gradcam_input(
        image,
        model_name
    )

    heatmap = generate_gradcam(
        model,
        backbone,
        target_layer,
        input_tensor,
        model_name
    )

    overlay = create_gradcam_overlay(
        image,
        heatmap
    )

    filename = f"{session_id}_{model_name}.jpg"

    output_path = GRADCAM_DIR / filename

    overlay.save(
        output_path,
        quality=95
    )

    return {
        "model": model_name,
        "image_url": f"/static/gradcam/{filename}",
        "gradcam_note": "Experimental feature-attribution visualization."
    }


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/research")
def research():
    return render_template("research.html")


@app.route("/demo")
def demo():
    return render_template("demo.html")


# ============================================================
# THREE-MODEL PREDICTION API
# GRAD-CAM IS NOT RUN HERE
# ============================================================

@app.route("/api/compare", methods=["POST"])
def api_compare():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format."}), 400

    try:

        image = Image.open(file.stream).convert("RGB")

        valid, message = validate_image_quality(image)

        if not valid:
            return jsonify({"error": message}), 400

        session_id = uuid4().hex

        # Store the uploaded image for separate Grad-CAM requests.
        image_sessions[session_id] = image.copy()

        results = []

        model_names = [
            "MobileNetV2",
            "EfficientNetB0",
            "ResNet50"
        ]

        for model_name in model_names:

            print(f"\nPredicting with {model_name}...")

            start = time.time()

            result = predict_image(
                image,
                model_name
            )

            results.append(result)

            print(
                f"{model_name} prediction completed in "
                f"{time.time() - start:.2f} seconds."
            )

        comparison = compare_predictions(results)

        print("\nAll predictions completed.")

        return jsonify({
            "success": True,
            "session_id": session_id,
            "filename": file.filename,
            "results": results,
            "comparison": comparison,
            "notice": (
                "OncoLens is an academic research prototype "
                "and is not a diagnostic tool."
            )
        })

    except Exception as error:

        print("Comparison error:", repr(error))

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# SEPARATE GRAD-CAM API
# ONE MODEL PER REQUEST
# ============================================================

@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():

    data = request.get_json(silent=True) or {}

    session_id = data.get("session_id")
    model_name = data.get("model")

    if not session_id or session_id not in image_sessions:
        return jsonify({"error": "Image session expired. Upload again."}), 400

    if model_name not in GRADCAM_CONFIG:
        return jsonify({"error": "Invalid model."}), 400

    try:

        print(f"\nGenerating Grad-CAM for {model_name}...")

        start = time.time()

        result = generate_model_gradcam(
            image_sessions[session_id],
            model_name,
            session_id
        )

        print(
            f"{model_name} Grad-CAM completed in "
            f"{time.time() - start:.2f} seconds."
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:

        print(
            f"Grad-CAM error for {model_name}:",
            repr(error)
        )

        return jsonify({
            "success": False,
            "error": str(error),
            "model": model_name
        }), 500


# ============================================================
# SINGLE MODEL PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    model_name = request.form.get("model")

    if model_name not in MODEL_PATHS:
        return jsonify({"error": "Invalid model selection."}), 400

    try:

        image = Image.open(file.stream).convert("RGB")

        valid, message = validate_image_quality(image)

        if not valid:
            return jsonify({"error": message}), 400

        result = predict_image(image, model_name)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:

        print("Prediction error:", repr(error))

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )