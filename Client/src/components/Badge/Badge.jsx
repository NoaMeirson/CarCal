import './Badge.css';

function Badge({ children, type = 'info' }) {
  return <span className={`badge badge-${type}`}>{children}</span>;
}

export default Badge;
