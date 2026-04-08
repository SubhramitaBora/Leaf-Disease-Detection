import { useEffect, useState, startTransition } from "react";
import { EMPTY_PREDICTION, INITIAL_MESSAGES } from "../constants/prediction";
import { askPlantAssistant, predictLeafDisease } from "../services/leafApi";

function useLeafDiagnosis() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState(EMPTY_PREDICTION);
  const [requestError, setRequestError] = useState("");
  const [chatError, setChatError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState(INITIAL_MESSAGES);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }

    const nextPreviewUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(nextPreviewUrl);

    return () => URL.revokeObjectURL(nextPreviewUrl);
  }, [selectedFile]);

  function handleFileSelect(file) {
    setSelectedFile(file);
    setPrediction(EMPTY_PREDICTION);
    setRequestError("");
  }

  async function runPrediction() {
    if (!selectedFile) {
      setRequestError("Please select a leaf image before detection.");
      return;
    }

    setIsPredicting(true);
    setRequestError("");
    setChatError("");

    try {
      const result = await predictLeafDisease(selectedFile);
      startTransition(() => {
        setPrediction({ ...EMPTY_PREDICTION, ...result });
        setMessages((current) => [
          ...current,
          {
            role: "assistant",
            content: `Prediction complete: ${result.disease} with ${result.confidence} confidence.`,
          },
        ]);
      });
    } catch (error) {
      setRequestError(error.message || "Prediction failed.");
    } finally {
      setIsPredicting(false);
    }
  }

  async function sendMessage(messageOverride) {
    const nextMessage = (messageOverride ?? chatInput).trim();
    if (!nextMessage) {
      return;
    }

    setIsChatting(true);
    setChatError("");
    setChatInput("");
    setMessages((current) => [...current, { role: "user", content: nextMessage }]);

    try {
      const reply = await askPlantAssistant(
        nextMessage,
        prediction.disease === EMPTY_PREDICTION.disease ? undefined : prediction,
      );
      setMessages((current) => [...current, { role: "assistant", content: reply }]);
    } catch (error) {
      setChatError(error.message || "Chat request failed.");
    } finally {
      setIsChatting(false);
    }
  }

  return {
    chatError,
    chatInput,
    isChatting,
    isPredicting,
    messages,
    prediction,
    previewUrl,
    requestError,
    selectedFile,
    sendMessage,
    setChatInput,
    handleFileSelect,
    runPrediction,
  };
}

export default useLeafDiagnosis;
