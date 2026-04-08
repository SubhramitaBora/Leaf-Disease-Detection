import { EMPTY_PREDICTION } from "../../constants/prediction";

function ResultPanel({ prediction }) {
  if (prediction.disease === EMPTY_PREDICTION.disease) {
    return null;
  }

  return (
    <section className="card result-card-panel">
      <h2>Prediction Result</h2>

      <div className="result-pill-grid">
        <div className="result-pill">
          <strong>Plant:</strong> {prediction.plant}
        </div>
        <div className="result-pill">
          <strong>Disease:</strong> {prediction.disease}
        </div>
        <div className="result-pill">
          <strong>Confidence:</strong> {prediction.confidence}
        </div>
      </div>
    </section>
  );
}

export default ResultPanel;
