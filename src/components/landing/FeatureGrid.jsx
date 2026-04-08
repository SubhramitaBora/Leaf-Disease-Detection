const features = [
  {
    title: "Upload and preview",
    description:
      "Select a leaf image from your device and confirm the preview before running analysis.",
  },
  {
    title: "Instant prediction",
    description:
      "Your Flask model returns a healthy vs diseased result with a confidence score in seconds.",
  },
  {
    title: "Plant assistant",
    description:
      "Ask follow-up questions about care, symptoms, and prevention after the prediction is ready.",
  },
];

function FeatureGrid() {
  return (
    <section className="feature-section" id="learn-more">
      <div className="feature-section__intro">
        <span className="section-kicker">Why PlantCare</span>
        <h2>A simple workflow for quick plant health checks</h2>
      </div>

      <div className="feature-grid">
        {features.map((feature) => (
          <article key={feature.title} className="feature-card">
            <h3>{feature.title}</h3>
            <p>{feature.description}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default FeatureGrid;
