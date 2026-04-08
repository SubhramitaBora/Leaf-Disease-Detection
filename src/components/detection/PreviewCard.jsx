function PreviewCard({ previewUrl }) {
  return (
    <section className="card detector-card">
      <h2>Image Preview</h2>

      <div className="preview-frame">
        {previewUrl ? (
          <img className="preview-frame__image" src={previewUrl} alt="Selected leaf" />
        ) : (
          <p>Your selected leaf image will appear here.</p>
        )}
      </div>
    </section>
  );
}

export default PreviewCard;
