import { useEffect, useState } from 'react';
import './Loader.css';

function Loader({ text = 'Analyzing...' }) {
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          setPhase(p => (p + 1) % 3);
          return 0;
        }
        return p + Math.random() * 15;
      });
    }, 400);
    return () => clearInterval(interval);
  }, []);

  const phases = [
    'Uploading image...',
    'Running AI detection...',
    'Generating report...'
  ];

  return (
    <div className="loader-container">
      <div className="loader-wrapper">
        <div className="loader-outer">
          <div className="loader-inner">
            <div className="loader-core"></div>
          </div>
        </div>
        <div className="loader-ring"></div>
        <div className="loader-particles">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="particle" style={{ '--i': i }}></div>
          ))}
        </div>
      </div>
      <div className="loader-content">
        <span className="loader-text">{text}</span>
        <span className="loader-phase">{phases[phase]}</span>
        <div className="loader-progress">
          <div className="loader-progress-fill" style={{ width: `${Math.min(progress, 100)}%` }}></div>
        </div>
      </div>
    </div>
  );
}

export default Loader;
