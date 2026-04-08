function UploadCard({
  fileName,
  isPredicting,
  requestError,
  onFileSelect,
  onPredict,
}) {
  return (
    <section className="card detector-card">
      <h2>File Upload &amp; Image Preview</h2>
      <p className="card-subtitle">No plugins. Just simple image upload.</p>

      <div className="upload-dropzone">
        <input
          className="upload-dropzone__input"
          type="file"
          accept="image/*"
          onChange={(event) => onFileSelect(event.target.files?.[0] || null)}
        />
        <div className="upload-dropzone__content">
          <div className="upload-dropzone__icon">Upload</div>
          <p>Select a file or drag here</p>
          <span className="button button--small">Select a file</span>
        </div>
      </div>

      <p className="upload-status">
        {fileName ? `Selected file: ${fileName}` : "No image selected yet."}
      </p>

      <button
        type="button"
        className="button button--primary button--wide"
        disabled={isPredicting}
        onClick={onPredict}
      >
        {isPredicting ? "Detecting..." : "Detect Disease"}
      </button>

      {requestError ? <p className="error-text">{requestError}</p> : null}
    </section>
  );
}

export default UploadCard;
