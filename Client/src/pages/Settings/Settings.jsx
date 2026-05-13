import { useState } from 'react'
import Card from '../../components/Card/Card'
import Button from '../../components/Button/Button'
import './Settings.css'

function Settings() {
  const [settings, setSettings] = useState({
    apiEndpoint: 'http://localhost:8000',
    engineEndpoint: 'http://localhost:8001',
    autoRefresh: true,
    refreshInterval: 30,
    theme: 'light'
  })

  const [saved, setSaved] = useState(false)

  const handleChange = (field, value) => {
    setSettings(prev => ({ ...prev, [field]: value }))
    setSaved(false)
  }

  const handleSave = () => {
    // In production, save to localStorage or backend
    localStorage.setItem('carcal-settings', JSON.stringify(settings))
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  return (
    <div className="settings">
      <div className="page-header">
        <h2>Settings</h2>
        <p className="subtitle">Configure your CarCal client preferences</p>
      </div>

      <div className="settings-grid">
        <Card title="API Configuration" className="settings-card">
          <div className="form-group">
            <label>Client API Endpoint</label>
            <input
              type="text"
              value={settings.apiEndpoint}
              onChange={(e) => handleChange('apiEndpoint', e.target.value)}
              className="form-input"
            />
          </div>
          <div className="form-group">
            <label>Engine API Endpoint</label>
            <input
              type="text"
              value={settings.engineEndpoint}
              onChange={(e) => handleChange('engineEndpoint', e.target.value)}
              className="form-input"
            />
          </div>
        </Card>

        <Card title="Preferences" className="settings-card">
          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={settings.autoRefresh}
                onChange={(e) => handleChange('autoRefresh', e.target.checked)}
              />
              <span>Auto-refresh dashboard</span>
            </label>
          </div>
          <div className="form-group">
            <label>Refresh Interval (seconds)</label>
            <input
              type="number"
              value={settings.refreshInterval}
              onChange={(e) => handleChange('refreshInterval', parseInt(e.target.value))}
              className="form-input"
              min="10"
              max="300"
            />
          </div>
        </Card>

        <Card title="About" className="settings-card">
          <div className="about-info">
            <p><strong>CarCal</strong> - Vehicle Analysis System</p>
            <p>Version 1.0.0</p>
            <p className="about-desc">
              AI-powered vehicle damage and parts detection using YOLO models.
            </p>
          </div>
        </Card>
      </div>

      <div className="settings-actions">
        <Button onClick={handleSave}>
          {saved ? '✓ Saved!' : 'Save Settings'}
        </Button>
      </div>
    </div>
  )
}

export default Settings