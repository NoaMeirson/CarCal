import { useEffect, useState } from 'react';
import './Stepper.css';

function Stepper({ steps, current }) {
  const [animatedStep, setAnimatedStep] = useState(current);
  
  useEffect(() => {
    setAnimatedStep(current);
  }, [current]);

  return (
    <div className="stepper-container">
      <div className="stepper">
        {steps.map((step, idx) => (
          <div 
            key={step} 
            className={`step ${idx === animatedStep ? 'active' : ''} ${idx < animatedStep ? 'completed' : ''}`}
            style={{ '--step-index': idx }}
          >
            <div className="step-connector">
              <div className="step-connector-fill"></div>
            </div>
            <div className="step-circle">
              {idx < animatedStep ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              ) : (
                idx + 1
              )}
            </div>
            <div className="step-content">
              <div className="step-label">{step}</div>
              {idx === animatedStep && (
                <div className="step-indicator"></div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Stepper;
