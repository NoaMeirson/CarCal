function Button({ children, onClick, disabled = false, loading = false, variant = 'primary', className = '' }) {
  return (
    <button 
      className={`btn btn-${variant} ${className}`}
      onClick={onClick}
      disabled={disabled || loading}
    >
      {loading ? 'Loading...' : children}
    </button>
  )
}

export default Button