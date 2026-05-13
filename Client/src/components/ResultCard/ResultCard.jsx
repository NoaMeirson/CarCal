import Badge from '../Badge/Badge';
import './ResultCard.css';

function ResultCard({ result }) {
  if (!result) return null;

  const damageCount = result.detections?.filter(d => d.type === 'damage').length || 0;
  const partCount = result.detections?.filter(d => d.type === 'part').length || 0;

  return (
    <div className="result-card-container">
      <div className="result-header-card">
        <div className="result-status-badge">
          <div className="status-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          </div>
          <div className="status-content">
            <span className="status-label">Analysis Complete</span>
            <span className="status-id">ID: {result.requestId}</span>
          </div>
        </div>
        <div className="result-actions">
          <button className="result-action-btn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download
          </button>
          <button className="result-action-btn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/>
              <polyline points="16 6 12 2 8 6"/>
              <line x1="12" y1="2" x2="12" y2="15"/>
            </svg>
            Share
          </button>
        </div>
      </div>

      <div className="result-stats-grid">
        <div className="result-stat-card">
          <div className="stat-icon damage">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="15" y1="9" x2="9" y2="15"/>
              <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{damageCount}</span>
            <span className="stat-label">Damages Found</span>
          </div>
        </div>
        <div className="result-stat-card">
          <div className="stat-icon parts">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 17h14v-5H5v5z"/>
              <path d="M5 17V7l-2 3v4"/>
              <path d="M19 17V7l2 3v4"/>
            </svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{partCount}</span>
            <span className="stat-label">Parts Detected</span>
          </div>
        </div>
        <div className="result-stat-card">
          <div className="stat-icon cost">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="1" x2="12" y2="23"/>
              <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{result.cost || '$1,200 (est.)'}</span>
            <span className="stat-label">Est. Repair Cost</span>
          </div>
        </div>
        <div className="result-stat-card">
          <div className="stat-icon confidence">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div className="stat-info">
            <span className="stat-value">{Math.round((result.confidence || 0.93) * 100)}%</span>
            <span className="stat-label">AI Confidence</span>
          </div>
        </div>
      </div>

      <div className="result-detections">
        <h3 className="detections-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          Detection Details
        </h3>
        <div className="detections-list">
          {result.detections?.map((detection, idx) => (
            <div 
              key={detection.id} 
              className={`detection-row ${detection.type}`}
              style={{ '--delay': `${idx * 0.1}s` }}
            >
              <div className="detection-icon">
                {detection.type === 'damage' ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 15s1-1 4-1 5 1 8 1 4-1 4-1V3s-1 1-4 1-5-1-8-1-4 1-4 1z"/>
                    <line x1="4" y1="22" x2="4" y2="15"/>
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 6v6l4 2"/>
                  </svg>
                )}
              </div>
              <div className="detection-info">
                <span className="detection-label">{detection.label}</span>
                <Badge type={detection.type === 'damage' ? 'danger' : 'info'}>
                  {detection.type}
                </Badge>
              </div>
              <div className="detection-confidence">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${detection.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="confidence-value">
                  {Math.round(detection.confidence * 100)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="result-recommendations">
        <h3 className="recommendations-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9.663 17h4.673M12 3v1m6.364 1.635l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
          </svg>
          Recommended Next Steps
        </h3>
        <div className="recommendations-list">
          <div className="recommendation-item">
            <span className="recommendation-icon">🔧</span>
            <span>Contact a certified repair shop for professional assessment</span>
          </div>
          <div className="recommendation-item">
            <span className="recommendation-icon">📋</span>
            <span>Share this report with your insurance provider</span>
          </div>
          <div className="recommendation-item">
            <span className="recommendation-icon">💾</span>
            <span>Save this analysis for your vehicle maintenance records</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultCard;