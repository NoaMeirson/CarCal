// API Service layer for CarCal backend integration
// This service is prepared for future integration with the backend APIs

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

/**
 * Analyze an image using the CarCal client API
 * @param {File} file - Image file to analyze
 * @param {string} mode - Analysis mode (full, damage, parts)
 * @returns {Promise<Object>} Analysis result
 */
export async function analyzeImage(file, mode = 'full') {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await fetch(`${API_BASE}/analyze?mode=${mode}`, {
    method: 'POST',
    body: formData
  })
  
  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`)
  }
  
  return response.json()
}

/**
 * Check health status of the client API
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`)
  
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`)
  }
  
  return response.json()
}

/**
 * Get analysis history from the backend
 * @param {number} limit - Number of results to fetch
 * @returns {Promise<Array>} Analysis history
 */
export async function getHistory(limit = 50) {
  const response = await fetch(`${API_BASE}/history?limit=${limit}`)
  
  if (!response.ok) {
    throw new Error(`Failed to fetch history: ${response.statusText}`)
  }
  
  return response.json()
}

export default {
  analyzeImage,
  checkHealth,
  getHistory
}