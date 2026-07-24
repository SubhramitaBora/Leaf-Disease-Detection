import os

try:
    import google.generativeai as genai
except Exception:
    genai = None

DEFAULT_MODEL_NAME = "models/gemini-2.5-flash"


def create_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL_NAME", DEFAULT_MODEL_NAME)

    if not genai or not api_key:
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def build_prompt(message, prediction):
    return (
        "You are an assistant for plant leaf disease detection. "
        "If a prediction is provided, use it to guide a short, practical reply. "
        "If the prediction is binary, avoid naming specific diseases. "
        "Keep the response concise and helpful. "
        "Use plain text only, no markdown, no **bold**, no italics. "
        "Use clear line breaks and simple sentences.\n\n"
        f"Prediction: {prediction}\n"
        f"User: {message}\n"
        "Assistant:"
    )


def generate_reply(model, message, prediction):
    if model is None:
        raise RuntimeError("Gemini API is not configured.")

    prompt = build_prompt(message, prediction)
    response = model.generate_content(prompt)
    reply = response.text.strip() if response and response.text else ""
    if not reply:
        raise RuntimeError("Empty response from Gemini.")
    return (
        reply.replace("**", "")
        .replace("__", "")
        .replace("*", "")
        .replace("`", "")
    )


def generate_reply_stream(model, message, prediction):
    if model is None:
        raise RuntimeError("Gemini API is not configured.")

    prompt = build_prompt(message, prediction)
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            cleaned = (
                chunk.text.replace("**", "")
                .replace("__", "")
                .replace("*", "")
                .replace("`", "")
            )
            yield cleaned

