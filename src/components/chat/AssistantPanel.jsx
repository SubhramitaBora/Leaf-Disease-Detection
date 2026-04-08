const starterQuestions = [
  "give me step by step to improve the diseased leaf here",
  "what are the prevention steps for future leaves",
  "explain what this confidence score means",
];

function AssistantPanel({
  chatInput,
  chatError,
  isChatting,
  messages,
  onChatInputChange,
  onSendMessage,
}) {
  return (
    <section className="card assistant-panel">
      <p className="assistant-panel__eyebrow">Plant Assistant</p>
      <h2>AI Plant Chatbot</h2>
      <p className="card-subtitle">
        Ask about diseases, symptoms, treatment plans, or prevention steps.
      </p>

      <div className="assistant-quick-actions">
        {starterQuestions.map((question) => (
          <button
            key={question}
            type="button"
            className="quick-chip"
            disabled={isChatting}
            onClick={() => onSendMessage(question)}
          >
            {question}
          </button>
        ))}
      </div>

      <div className="chat-window">
        {messages.map((message, index) => (
          <article
            key={`${message.role}-${index}`}
            className={
              message.role === "user" ? "chat-bubble chat-bubble--user" : "chat-bubble"
            }
          >
            <p>{message.content}</p>
          </article>
        ))}
      </div>

      <form
        className="chat-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSendMessage();
        }}
      >
        <input
          type="text"
          value={chatInput}
          onChange={(event) => onChatInputChange(event.target.value)}
          placeholder="Ask about leaf disease..."
        />
        <button type="submit" className="button button--primary" disabled={isChatting}>
          {isChatting ? "Sending..." : "Send"}
        </button>
      </form>

      {chatError ? <p className="error-text">{chatError}</p> : null}
    </section>
  );
}

export default AssistantPanel;
