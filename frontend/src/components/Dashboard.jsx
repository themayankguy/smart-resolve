import React from 'react';
import { 
  Ticket, AlertTriangle, CheckCircle, Clock, 
  Flame, Database, ArrowUpRight, ShieldAlert, Cpu
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, Legend
} from 'recharts';

const CATEGORY_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4', '#ef4444', '#64748b'];
const STATUS_COLORS = {
  'Open': '#3b82f6',
  'In Progress': '#f59e0b',
  'Resolved': '#10b981',
  'Closed': '#64748b'
};

export default function Dashboard({ metrics, onSelectTicket, onNavigateTab }) {
  if (!metrics) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-400">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mr-3"></div>
        Loading operational metrics...
      </div>
    );
  }

  const statCards = [
    { title: 'Total Tickets', value: metrics.total_tickets, icon: Ticket, color: 'text-blue-400', bg: 'bg-blue-500/10' },
    { title: 'Open Tickets', value: metrics.open_tickets, icon: Clock, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
    { title: 'In Progress', value: metrics.in_progress_tickets, icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-500/10' },
    { title: 'Resolved', value: metrics.resolved_tickets, icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    { title: 'High & Critical', value: metrics.high_critical_tickets, icon: Flame, color: 'text-rose-400', bg: 'bg-rose-500/10' },
    { title: 'Anomalies Detected', value: metrics.anomalous_tickets, icon: ShieldAlert, color: 'text-red-400', bg: 'bg-red-500/10' },
    { title: 'Training Samples', value: metrics.dataset_count, icon: Database, color: 'text-purple-400', bg: 'bg-purple-500/10' }
  ];

  const categoryData = metrics.category_distribution || [];
  const statusData = (metrics.status_distribution || []).map(s => ({
    name: s.status,
    value: s.count
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Top Banner / SIH Overview */}
      <div className="glass-panel" style={{ padding: '1.5rem', background: 'linear-gradient(135deg, rgba(30,58,138,0.25), rgba(15,23,42,0.85))' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
              <span className="badge badge-medium">SIH25195 Domain Solution</span>
              <span className="badge badge-normal">AI Core Active</span>
            </div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc' }}>
              Smart Helpdesk Operations & AI Analytics
            </h1>
            <p style={{ color: '#94a3b8', fontSize: '0.875rem', marginTop: '0.25rem' }}>
              Real-time semantic routing, PyTorch Autoencoder anomaly detection, and explainable SLA prioritization.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button 
              onClick={() => onNavigateTab('submit')} 
              className="btn-primary"
              style={{ fontSize: '0.875rem' }}
            >
              <Ticket size={16} /> Log New Ticket
            </button>
            <button 
              onClick={() => onNavigateTab('models')} 
              className="btn-secondary"
              style={{ fontSize: '0.875rem' }}
            >
              <Cpu size={16} /> Model Health
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
        gap: '1rem' 
      }}>
        {statCards.map((card, idx) => {
          const Icon = card.icon;
          return (
            <div key={idx} className="glass-panel" style={{ padding: '1.125rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' }}>
                  {card.title}
                </span>
                <div style={{ padding: '0.375rem', borderRadius: '0.375rem', background: 'rgba(255,255,255,0.05)' }}>
                  <Icon size={16} className={card.color} />
                </div>
              </div>
              <div style={{ fontSize: '1.625rem', fontWeight: 800, color: '#f8fafc' }}>
                {card.value}
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.25rem' }}>
        {/* Category Breakdown */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h2 style={{ fontSize: '1rem', color: '#f1f5f9', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>Ticket Volume by IT Domain Category</span>
          </h2>
          {categoryData.length === 0 ? (
            <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
              No ticket data available yet.
            </div>
          ) : (
            <div style={{ width: '100%', height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={categoryData} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
                  <XAxis 
                    dataKey="category" 
                    stroke="#64748b" 
                    fontSize={10} 
                    angle={-25} 
                    textAnchor="end" 
                    interval={0} 
                  />
                  <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                  <Tooltip 
                    contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                    itemStyle={{ color: '#60a5fa' }}
                  />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Status Distribution */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h2 style={{ fontSize: '1rem', color: '#f1f5f9', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>Ticket Status Lifecycle Distribution</span>
          </h2>
          {statusData.length === 0 ? (
            <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
              No ticket status data.
            </div>
          ) : (
            <div style={{ width: '100%', height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {statusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={STATUS_COLORS[entry.name] || '#3b82f6'} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  />
                  <Legend 
                    verticalAlign="bottom" 
                    wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} 
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {/* Recent Activity Table */}
      <div className="glass-panel" style={{ padding: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.125rem', color: '#f1f5f9' }}>Recent Incident Tickets</h2>
            <p style={{ color: '#94a3b8', fontSize: '0.8125rem' }}>Latest complaints processed by the automated ML pipeline</p>
          </div>
          <button 
            onClick={() => onNavigateTab('tickets')} 
            className="btn-ghost"
            style={{ fontSize: '0.8125rem' }}
          >
            View All ({metrics.total_tickets}) <ArrowUpRight size={14} />
          </button>
        </div>

        {(!metrics.recent_tickets || metrics.recent_tickets.length === 0) ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
            No tickets have been submitted yet. Use the "Submit Complaint" tab to create your first ticket.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Ticket ID</th>
                  <th>Complaint</th>
                  <th>Category</th>
                  <th>Department</th>
                  <th>Priority</th>
                  <th>Anomaly</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {metrics.recent_tickets.map((t) => (
                  <tr key={t.ticket_id} style={{ cursor: 'pointer' }} onClick={() => onSelectTicket(t)}>
                    <td style={{ fontFamily: 'monospace', fontWeight: 600, color: '#60a5fa' }}>
                      {t.ticket_id}
                    </td>
                    <td style={{ maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {t.complaint_text}
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8125rem', color: '#cbd5e1' }}>{t.category}</span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>{t.department}</span>
                    </td>
                    <td>
                      <span className={`badge badge-${t.priority.toLowerCase()}`}>
                        {t.priority}
                      </span>
                    </td>
                    <td>
                      {t.is_anomaly ? (
                        <span className="badge badge-anomaly" title={`Reconstruction error: ${t.reconstruction_error}`}>
                          Anomaly
                        </span>
                      ) : (
                        <span className="badge badge-normal">
                          Normal
                        </span>
                      )}
                    </td>
                    <td>
                      <span className={`badge badge-status-${t.status.toLowerCase().replace(' ', '')}`}>
                        {t.status}
                      </span>
                    </td>
                    <td>
                      <button 
                        onClick={(e) => { e.stopPropagation(); onSelectTicket(t); }} 
                        className="btn-ghost"
                        style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
