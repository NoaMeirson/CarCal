import './ModeCard.css';

function ModeCard({ icon, label, description, active, onClick }) {
  return (
    <div 
      className={`mode-card-new ${active ? 'active' : ''}`}
      onClick={onClick}
    >
      <div className="mode-card-icon">{icon}</div>
      <div className="mode-card-content">
        <div className="mode-card-label">{label}</div>
        <div className="mode-card-desc">{description}</div>
      </div>
      {active && (
        <div className="mode-card-check">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </div>
      )}
      <div className="mode-card-glow"></div>
    </div>
  );
}

export default ModeCard;