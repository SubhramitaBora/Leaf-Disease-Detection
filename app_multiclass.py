import json
import os
import re
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess_input,
)
from tensorflow.keras.models import load_model

from chatbot_service import create_gemini_model, generate_reply, generate_reply_stream

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_KEY = "resnet50"

MODEL_CONFIGS = {
    "resnet50": {
        "display_name": "ResNet50",
        "dataset_name": os.environ.get(
            "LEAF_MULTICLASS_RESNET_DATASET_NAME",
            "MultiClass_Dataset_Split_New",
        ),
        "model_path": os.environ.get(
            "LEAF_MULTICLASS_RESNET_MODEL_PATH",
            os.path.join(BASE_DIR, "new_multiclass_resnet50_best.keras"),
        ),
        "class_names_path": os.environ.get(
            "LEAF_MULTICLASS_RESNET_CLASS_NAMES_PATH",
            os.path.join(BASE_DIR, "new_multiclass_class_names.json"),
        ),
        "image_size": (224, 224),
        "preprocess_input": resnet50_preprocess_input,
    },
}


def load_class_names(class_names_path):
    with open(class_names_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_whitespace(value):
    return re.sub(r"\s+", " ", value).strip()


def prettify_label(value):
    label = normalize_whitespace(value.replace("_", " "))
    if label.islower():
        return label.title()
    return label


def split_prediction_label(class_name):
    if "___" in class_name:
        plant_name, disease_name = class_name.split("___", 1)
    else:
        plant_name, disease_name = class_name, "Unknown"

    return prettify_label(plant_name), prettify_label(disease_name)


def build_model_registry():
    registry = {}

    for model_key, config in MODEL_CONFIGS.items():
        class_names = load_class_names(config["class_names_path"])

        print(f"[INFO] Loading multiclass {config['display_name']} model from: {config['model_path']}")
        model = load_model(config["model_path"])
        print(
            f"[INFO] Loaded {len(class_names)} labels for {config['display_name']} "
            f"from: {config['class_names_path']}"
        )

        print(f"[INFO] Warming up {config['display_name']} with a dummy prediction...")
        dummy_input = np.zeros(
            (1, config["image_size"][0], config["image_size"][1], 3),
            dtype=np.float32,
        )
        dummy_input = config["preprocess_input"](dummy_input)
        model.predict(dummy_input, verbose=0)
        print(f"[INFO] Warm-up complete for {config['display_name']}.")

        registry[model_key] = {
            **config,
            "class_names": class_names,
            "model": model,
        }

    return registry


def build_available_models_response():
    return {
        model_key: {
            "display_name": config["display_name"],
            "dataset_name": config["dataset_name"],
            "class_count": len(config["class_names"]),
            "image_size": list(config["image_size"]),
        }
        for model_key, config in MODEL_REGISTRY.items()
    }


STATIC_FOLDER = os.path.join(BASE_DIR, "Frontend_Multi", "dist")
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="/")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_REGISTRY = build_model_registry()

gemini_model = create_gemini_model()
if gemini_model:
    model_name = os.environ.get("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
    print(f"[INFO] Gemini model ready: {model_name}")
else:
    if os.environ.get("GEMINI_API_KEY"):
        print("[WARN] Gemini key found, but model could not be initialized.")
    else:
        print("[WARN] Gemini not configured. Set GEMINI_API_KEY to enable /chat.")


def get_model_key():
    requested_model = (
        request.form.get("model")
        or request.args.get("model")
        or DEFAULT_MODEL_KEY
    ).strip().lower()

    if requested_model not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{requested_model}'. Available models: {available}.")

    return requested_model


def prepare_image(image_bytes, config):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, config["image_size"])
    image = np.expand_dims(image.astype(np.float32), axis=0)
    image = config["preprocess_input"](image)
    return image


def format_top_predictions(probabilities, class_names, limit=3):
    ranked_indices = np.argsort(probabilities)[::-1][:limit]
    top_predictions = []

    for index in ranked_indices:
        class_name = class_names[index]
        plant_name, disease_name = split_prediction_label(class_name)
        confidence_score = float(probabilities[index])
        top_predictions.append(
            {
                "class_name": class_name,
                "plant": plant_name,
                "disease": disease_name,
                "confidence": f"{confidence_score * 100:.2f}%",
                "confidence_score": confidence_score,
            }
        )

    return top_predictions


def build_prediction_response(probabilities, model_key, config):
    predicted_index = int(np.argmax(probabilities))
    predicted_class = config["class_names"][predicted_index]
    plant_name, disease_name = split_prediction_label(predicted_class)
    confidence_score = float(probabilities[predicted_index])

    return {
        "plant": plant_name,
        "disease": disease_name,
        "class_name": predicted_class,
        "confidence": f"{confidence_score * 100:.2f}%",
        "confidence_score": confidence_score,
        "symptoms": [],
        "cure": [],
        "prevention": [],
        "top_predictions": format_top_predictions(probabilities, config["class_names"]),
        "model_key": model_key,
        "model_name": config["display_name"],
        "model_type": f"multiclass_{model_key}",
        "dataset_name": config["dataset_name"],
        "class_count": len(config["class_names"]),
    }


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.post("/predict")
def predict():
    request_start = time.time()
    print("[INFO] /predict request received for multiclass backend.")

    if "image" not in request.files:
        print("[ERROR] No image file provided in request.")
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        print("[ERROR] Empty image file received.")
        return jsonify({"error": "No image file selected."}), 400

    try:
        model_key = get_model_key()
        config = MODEL_REGISTRY[model_key]
        image_bytes = image_file.read()

        print(f"[INFO] Processing file: {image_file.filename} with {config['display_name']}")
        processed_image = prepare_image(image_bytes, config)
        print(f"[INFO] Image preprocessed in {time.time() - request_start:.2f}s")

        predict_start = time.time()
        prediction = config["model"].predict(processed_image, verbose=0)
        print(f"[INFO] Model inference finished in {time.time() - predict_start:.2f}s")

        response = build_prediction_response(prediction[0], model_key, config)
        print(
            f"[INFO] Prediction complete in {time.time() - request_start:.2f}s -> "
            f"{response['model_name']} | {response['class_name']} ({response['confidence']})"
        )
        return jsonify(response)
    except ValueError as error:
        print(f"[ERROR] Validation failed: {error}")
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        print(f"[ERROR] Prediction failed: {error}")
        return jsonify({"error": f"Prediction failed: {error}"}), 500


@app.post("/chat")
def chat():
    if gemini_model is None:
        return jsonify({"error": "Gemini API is not configured."}), 400

    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "").strip()
    prediction = payload.get("prediction") or {}

    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        def generate():
            for chunk in generate_reply_stream(gemini_model, message, prediction):
                yield chunk
        return Response(stream_with_context(generate()), mimetype="text/plain")
    except Exception as error:
        return jsonify({"error": f"Gemini request failed: {error}"}), 500


if __name__ == "__main__":
    host = os.environ.get("LEAF_MULTICLASS_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("LEAF_MULTICLASS_BACKEND_PORT", "5001"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
