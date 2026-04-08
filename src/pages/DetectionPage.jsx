import AssistantPanel from "../components/chat/AssistantPanel";
import PreviewCard from "../components/detection/PreviewCard";
import ResultPanel from "../components/detection/ResultPanel";
import UploadCard from "../components/detection/UploadCard";
import useLeafDiagnosis from "../hooks/useLeafDiagnosis";

function DetectionPage() {
  const {
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
  } = useLeafDiagnosis();

  return (
    <main className="page page--detect">
      <section className="detect-hero">
        <h1>Protect Your Plants with Instant Disease Detection</h1>
        <p>Upload a plant image and analyze the disease</p>
      </section>

      <section className="detect-grid">
        <UploadCard
          fileName={selectedFile?.name}
          isPredicting={isPredicting}
          requestError={requestError}
          onFileSelect={handleFileSelect}
          onPredict={runPrediction}
        />
        <PreviewCard previewUrl={previewUrl} />
      </section>

      <ResultPanel prediction={prediction} />

      <AssistantPanel
        chatInput={chatInput}
        chatError={chatError}
        isChatting={isChatting}
        messages={messages}
        onChatInputChange={setChatInput}
        onSendMessage={sendMessage}
      />
    </main>
  );
}

export default DetectionPage;
