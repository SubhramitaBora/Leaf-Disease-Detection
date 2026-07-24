import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

from chatbot_service import create_gemini_model, generate_reply
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.environ.get(
    "LEAF_BINARY_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet50_leaf_disease_final.h5"),
)
IMAGE_SIZE = (224, 224)
CLASS_NAMES = {0: "Diseased leaf detected", 1: "Healthy"}

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

print(f"[INFO] Loading binary ResNet model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

print("[INFO] Warming up model with a dummy prediction...")
dummy_input = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
dummy_input = preprocess_input(dummy_input)
model.predict(dummy_input, verbose=0)
print("[INFO] Warm-up complete. Backend is ready for requests.")

gemini_model = create_gemini_model()
if gemini_model:
    model_name = os.environ.get("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
    print(f"[INFO] Gemini model ready: {model_name}")
else:
    if os.environ.get("GEMINI_API_KEY"):
        print("[WARN] Gemini key found, but model could not be initialized.")
    else:
        print("[WARN] Gemini not configured. Set GEMINI_API_KEY to enable /chat.")


def prepare_image(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.expand_dims(image.astype(np.float32), axis=0)
    image = preprocess_input(image)
    return image


def build_prediction_response(score):
    predicted_index = 1 if score > 0.5 else 0
    confidence = score if predicted_index == 1 else 1 - score

    return {
        "plant": "Unknown crop",
        "disease": CLASS_NAMES[predicted_index],
        "confidence": f"{confidence * 100:.2f}%",
        "symptoms": [],
        "cure": [],
        "prevention": [],
        "raw_score": float(score),
        "model_type": "binary_resnet50",
    }


@app.get("/")
def home():
    return jsonify(
        {
            "message": "Leaf disease detection backend is running.",
            "predict_endpoint": "/predict",
            "model_path": MODEL_PATH,
            "model_type": "binary_resnet50",
        }
    )


@app.post("/predict")
def predict():
    request_start = time.time()
    print("[INFO] /predict request received.")

    if "image" not in request.files:
        print("[ERROR] No image file provided in request.")
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        print("[ERROR] Empty image file received.")
        return jsonify({"error": "No image file selected."}), 400

    try:
        print(f"[INFO] Processing file: {image_file.filename}")
        processed_image = prepare_image(image_file)
        print(f"[INFO] Image preprocessed in {time.time() - request_start:.2f}s")

        predict_start = time.time()
        prediction = model.predict(processed_image, verbose=0)
        print(f"[INFO] Model inference finished in {time.time() - predict_start:.2f}s")

        score = float(prediction[0][0])
        response = build_prediction_response(score)
        print(
            f"[INFO] Prediction complete in {time.time() - request_start:.2f}s -> "
            f"{response['disease']} ({response['confidence']})"
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
        reply = generate_reply(gemini_model, message, prediction)
        return jsonify({"reply": reply})
    except Exception as error:
        return jsonify({"error": f"Gemini request failed: {error}"}), 500


if __name__ == "__main__":
    host = os.environ.get("LEAF_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("LEAF_BACKEND_PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
