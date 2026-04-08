import FeatureGrid from "../components/landing/FeatureGrid";
import HeroSection from "../components/landing/HeroSection";

function LandingPage() {
  return (
    <main className="page page--landing">
      <HeroSection />
      <FeatureGrid />
    </main>
  );
}

export default LandingPage;
