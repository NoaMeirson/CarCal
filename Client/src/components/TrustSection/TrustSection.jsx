import GlassyCard from '../GlassyCard/GlassyCard';
import './TrustSection.css';

function TrustSection() {
  const benefits = [
    {
      icon: '⚡',
      title: 'Lightning Fast',
      description: 'Get results in under 3 seconds with our optimized AI models'
    },
    {
      icon: '🎯',
      title: 'Highly Accurate',
      description: '98.5% accuracy rate powered by deep learning models'
    },
    {
      icon: '💰',
      title: 'Cost Estimation',
      description: 'Get instant repair cost estimates to plan your budget'
    },
    {
      icon: '🔒',
      title: 'Secure & Private',
      description: 'Your data is encrypted and never shared with third parties'
    }
  ];

  const steps = [
    { number: '01', title: 'Upload', description: 'Upload a clear photo of your vehicle' },
    { number: '02', title: 'Select Mode', description: 'Choose analysis type (full/damage/parts)' },
    { number: '03', title: 'Analyze', description: 'Our AI processes the image instantly' },
    { number: '04', title: 'Results', description: 'View detailed damage report & recommendations' }
  ];

  return (
    <div className="trust-section-container">
      <GlassyCard className="benefits-card">
        <h2 className="section-title">
          <span className="title-icon">✨</span>
          Why Choose CarCal
        </h2>
        <div className="benefits-grid">
          {benefits.map((benefit, idx) => (
            <div 
              key={benefit.title} 
              className="benefit-item"
              style={{ '--delay': `${idx * 0.1}s` }}
            >
              <div className="benefit-icon">{benefit.icon}</div>
              <div className="benefit-content">
                <h3 className="benefit-title">{benefit.title}</h3>
                <p className="benefit-desc">{benefit.description}</p>
              </div>
            </div>
          ))}
        </div>
      </GlassyCard>

      <GlassyCard className="how-it-works-card">
        <h2 className="section-title">
          <span className="title-icon">🔄</span>
          How It Works
        </h2>
        <div className="steps-grid">
          {steps.map((step, idx) => (
            <div 
              key={step.number} 
              className="step-item"
              style={{ '--delay': `${idx * 0.15}s` }}
            >
              <div className="step-number">{step.number}</div>
              <div className="step-content">
                <h3 className="step-title">{step.title}</h3>
                <p className="step-desc">{step.description}</p>
              </div>
              {idx < steps.length - 1 && <div className="step-arrow">→</div>}
            </div>
          ))}
        </div>
      </GlassyCard>
    </div>
  );
}

export default TrustSection;