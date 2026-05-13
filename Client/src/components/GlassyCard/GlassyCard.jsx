import './GlassyCard.css';

function GlassyCard({ children, className = '', style = {}, ...props }) {
  return (
    <div className={`glassy-card ${className}`} style={style} {...props}>
      {children}
    </div>
  );
}

export default GlassyCard;
