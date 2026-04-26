import { useState } from 'react'
import Card from '../../components/Card/Card'
import './History.css'

function History() {
  const [analyses] = useState([
    { id: 'req-001', vehicle: 'Toyota Camry', date: '2026-04-25', mode: 'full', status: 'completed', detections: 5 },
    { id: 'req-002', vehicle: 'Honda Civic', date: '2026-04-25', mode: 'damage', status: 'completed', detections: 2 },
    { id: 'req-003', vehicle: 'Ford F-150', date: '2026-04-24', mode: 'full', status: 'completed', detections: 8 },
    { id: 'req-004', vehicle: 'Tesla Model 3', date: '2026-04-24', mode: 'parts', status: 'completed', detections: 6 },
    { id: 'req-005', vehicle: 'BMW 3 Series', date: '2026-04-23', mode: 'full', status: 'completed', detections: 4 }
  ])

  const [filter, setFilter] = useState('')

  const filteredAnalyses = analyses.filter(a => 
    a.vehicle.toLowerCase().includes(filter.toLowerCase()) ||
    a.id.toLowerCase().includes(filter.toLowerCase())
  )

  return (
    <div className="history">
      <div className="page-header">
        <h2>Analysis History</h2>
        <p className="subtitle">View past vehicle analysis results</p>
      </div>

      <Card className="history-card">
        <div className="filters">
          <input
            type="text"
            placeholder="Search by vehicle or request ID..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="search-input"
          />
        </div>

        <div className="table-container">
          <table className="history-table">
            <thead>
              <tr>
                <th>Request ID</th>
                <th>Vehicle</th>
                <th>Date</th>
                <th>Mode</th>
                <th>Detections</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredAnalyses.map(analysis => (
                <tr key={analysis.id}>
                  <td className="id-cell">{analysis.id}</td>
                  <td>{analysis.vehicle}</td>
                  <td>{analysis.date}</td>
                  <td>
                    <span className="mode-badge">{analysis.mode}</span>
                  </td>
                  <td>{analysis.detections}</td>
                  <td>
                    <span className={`status-badge ${analysis.status}`}>
                      {analysis.status}
                    </span>
                  </td>
                  <td>
                    <button className="action-btn">View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

export default History