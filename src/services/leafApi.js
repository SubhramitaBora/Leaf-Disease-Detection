const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

async function parseApiResponse(response, fallbackMessage) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || fallbackMessage);
  }
  return data;
}

export async function predictLeafDisease(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  return parseApiResponse(response, "Prediction failed.");
}

export async function askPlantAssistant(message, prediction) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      prediction,
    }),
  });

  const data = await parseApiResponse(response, "Chat request failed.");
  return data.reply;
}
