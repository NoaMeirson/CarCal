import { Link } from 'react-router-dom';
import GlassyCard from '../GlassyCard/GlassyCard';
import './Hero.css';

function Hero() {
  return (
    <section className="hero-section">
      <div className="hero-bg">
        <div className="hero-gradient-1"></div>
        <div className="hero-gradient-2"></div>
        <div className="hero-gradient-3"></div>
        <div className="hero-grid"></div>
      </div>
      <GlassyCard className="hero-card">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="hero-badge-dot"></span>
            AI-Powered Analysis
          </div>
          <h1 className="hero-title">
            Detect Vehicle Damage <br />
            <span className="hero-title-gradient">In Seconds</span>
          </h1>
          <p className="hero-subtitle">
            Upload a photo of your car and get instant, AI-driven damage detection and repair cost estimates. 
            Fast, accurate, and easy to use.
          </p>
          <div className="hero-actions">
            <Link to="/analyze" className="hero-cta-primary">
              Start Analysis
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </Link>
            <button className="hero-cta-secondary">
              Watch Demo
            </button>
          </div>
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-value">50K+</span>
              <span className="hero-stat-label">Vehicles Analyzed</span>
            </div>
            <div className="hero-stat-divider"></div>
            <div className="hero-stat">
              <span className="hero-stat-value">98.5%</span>
              <span className="hero-stat-label">Accuracy</span>
            </div>
            <div className="hero-stat-divider"></div>
            <div className="hero-stat">
              <span className="hero-stat-value">&lt;3s</span>
              <span className="hero-stat-label">Processing Time</span>
            </div>
          </div>
        </div>
        <div className="hero-visual">
          <div className="hero-image-container">
            <img 
              src="https://images.unsplash.com/photo-1503736334956-4c8f8e92946d?auto=format&fit=crop&w=600&q=80" 
              alt="Car analysis" 
              className="hero-image"
            />
            <div className="hero-image-overlay">
              <div className="hero-overlay-card">
                <div className="hero-overlay-icon">🔍</div>
                <div className="hero-overlay-text">
                  <span className="hero-overlay-label">Analyzing...</span>
                  <span className="hero-overlay-value">Front Bumper</span>
                </div>
              </div>
            </div>
          </div>
          <div className="hero-floating-cards">
            <div className="hero-float-card float-1">
              <span className="float-icon">⚡</span>
              <span className="float-text">Instant Results</span>
            </div>
            <div className="hero-float-card float-2">
              <span className="float-icon">🎯</span>
              <span className="float-text">98.5% Accuracy</span>
            </div>
          </div>
        </div>
      </GlassyCard>
    </section>
  );
}

export default Hero;
