import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import GlassyCard from '../../components/GlassyCard/GlassyCard'
import Hero from '../../components/Hero/Hero'
import './Dashboard.css'

function Dashboard() {
  const [stats, setStats] = useState({
    totalAnalyses: 156,
    todayAnalyses: 12,
    successRate: 98.5,
    avgProcessingTime: '2.3s'
  })

  const [recentActivity, setRecentActivity] = useState([
    { id: 1, type: 'analysis', vehicle: 'Toyota Camry', status: 'completed', time: '5 min ago' },
    { id: 2, type: 'analysis', vehicle: 'Honda Civic', status: 'completed', time: '12 min ago' },
    { id: 3, type: 'analysis', vehicle: 'Ford F-150', status: 'processing', time: 'now' },
    { id: 4, type: 'analysis', vehicle: 'Tesla Model 3', status: 'completed', time: '25 min ago' }
  ])

  const [services, setServices] = useState([
    { name: 'Client API', status: 'healthy', url: 'http://localhost:8000' },
    { name: 'Engine API', status: 'healthy', url: 'http://localhost:8001' },
    { name: 'API Gateway', status: 'healthy', url: 'http://localhost:8002' }
  ])

  return (
    <div className="dashboard">
      <Hero />
      
      <div className="stats-grid">
        <GlassyCard className="stat-card">
          <div className="stat-icon-new">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20V10"/>
              <path d="M18 20V4"/>
              <path d="M6 20v-4"/>
            </svg>
          </div>
          <div className="stat-content">
            <span className="stat-value">{stats.totalAnalyses.toLocaleString()}</span>
            <span className="stat-label">Total Analyses</span>
          </div>
        </GlassyCard>
        <GlassyCard className="stat-card">
          <div className="stat-icon-new">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
              <line x1="16" y1="2" x2="16" y2="6"/>
              <line x1="8" y1="2" x2="8" y2="6"/>
              <line x1="3" y1="10" x2="21" y2="10"/>
            </svg>
          </div>
          <div className="stat-content">
            <span className="stat-value">{stats.todayAnalyses}</span>
            <span className="stat-label">Today's Analyses</span>
          </div>
        </GlassyCard>
        <GlassyCard className="stat-card">
          <div className="stat-icon-new success">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          </div>
          <div className="stat-content">
            <span className="stat-value">{stats.successRate}%</span>
            <span className="stat-label">Success Rate</span>
          </div>
        </GlassyCard>
        <GlassyCard className="stat-card">
          <div className="stat-icon-new warning">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <div className="stat-content">
            <span className="stat-value">{stats.avgProcessingTime}</span>
            <span className="stat-label">Avg Processing Time</span>
          </div>
        </GlassyCard>
      </div>

      <div className="dashboard-grid">
        <GlassyCard className="activity-card">
          <h3 className="card-title-new">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
            Recent Activity
          </h3>
          <div className="activity-list">
            {recentActivity.map(item => (
              <div key={item.id} className="activity-item-new">
                <div className="activity-info">
                  <span className="activity-vehicle">{item.vehicle}</span>
                  <span className="activity-time">{item.time}</span>
                </div>
                <span className={`activity-status-new ${item.status}`}>
                  {item.status === 'completed' ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  ) : (
                    <span className="spinner"></span>
                  )}
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </GlassyCard>

        <GlassyCard className="services-card">
          <h3 className="card-title-new">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
              <line x1="8" y1="21" x2="16" y2="21"/>
              <line x1="12" y1="17" x2="12" y2="21"/>
            </svg>
            Service Status
          </h3>
          <div className="services-list">
            {services.map(service => (
              <div key={service.name} className="service-item-new">
                <div className="service-info">
                  <span className="service-name">{service.name}</span>
                  <span className="service-url">{service.url}</span>
                </div>
                <span className={`service-status-new ${service.status}`}>
                  <span className="status-dot"></span>
                  {service.status}
                </span>
              </div>
            ))}
          </div>
        </GlassyCard>
      </div>
    </div>
  )
}

export default Dashboard