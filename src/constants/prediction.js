export const EMPTY_PREDICTION = {
  plant: "Unknown crop",
  disease: "No prediction yet",
  confidence: "0.00%",
  symptoms: [],
  cure: [],
  prevention: [],
  model_type: "binary_resnet50",
};

export const INITIAL_MESSAGES = [
  {
    role: "assistant",
    content:
      "Upload a leaf image and I can help explain the prediction and suggest next steps.",
  },
];
