import { Link } from "react-router-dom";

function HeroSection() {
  return (
    <section className="hero-section">
      <div className="hero-section__content">
        <span className="hero-badge">Leaf Disease Detection</span>
        <h1>
          Protect Your Plants with
          <span> Deep Learning Technology</span>
        </h1>
        <p>Detect leaf diseases instantly.</p>

        <div className="hero-actions">
          <Link className="button button--primary" to="/detect">
            Start Detection
          </Link>
          <a className="button button--secondary" href="#learn-more">
            Learn More
          </a>
        </div>
      </div>
    </section>
  );
}

export default HeroSection;
