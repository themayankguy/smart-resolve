import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, PlusCircle, ListFilter, BrainCircuit, Sparkles, 
  Search, RefreshCw, Upload, Check, AlertCircle, ShieldAlert,
  ArrowRight, X, Cpu, CheckCircle2, ChevronRight, MessageSquare
} from 'lucide-react';
import Dashboard from './components/Dashboard.jsx';

const API_BASE = '/api';

const ROLES = {
  employee: {
    id: 'employee',
    label: '👤 Employee / User',
    portalTitle: 'Employee Self-Service Portal',
    tabs: [
      { id: 'submit', label: 'Report Issue', icon: PlusCircle },
      { id: 'tickets', label: 'My Tickets', icon: ListFilter }
    ],
    defaultTab: 'submit'
  },
  resolver: {
    id: 'resolver',
    label: '🛡️ IT Resolver / Support',
    portalTitle: 'IT Incident Resolver Console',
    tabs: [
      { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { id: 'tickets', label: 'Ticket Queue', icon: ListFilter },
      { id: 'submit', label: 'Log Ticket', icon: PlusCircle }
    ],
    defaultTab: 'dashboard'
  },
  ml_admin: {
    id: 'ml_admin',
    label: '⚙️ ML Admin / Data Scientist',
    portalTitle: 'AI Core & Administrator Console',
    tabs: [
      { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { id: 'tickets', label: 'Ticket Queue', icon: ListFilter },
      { id: 'models', label: 'Model Evaluation', icon: Cpu },
      { id: 'fewshot', label: 'Few-Shot Studio', icon: Sparkles },
      { id: 'submit', label: 'Log Ticket', icon: PlusCircle }
    ],
    defaultTab: 'dashboard'
  }
};

export default function App() {
  const [currentRole, setCurrentRole] = useState('resolver');
  const [activeTab, setActiveTab] = useState('dashboard');
  const [metrics, setMetrics] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  const handleRoleChange = (newRole) => {
    setCurrentRole(newRole);
    const permittedTabs = ROLES[newRole].tabs.map(t => t.id);
    if (!permittedTabs.includes(activeTab)) {
      setActiveTab(ROLES[newRole].defaultTab);
    }
  };

  // Tickets state
  const [tickets, setTickets] = useState([]);
  const [totalTickets, setTotalTickets] = useState(0);
  const [ticketPage, setTicketPage] = useState(0);
  const [filterCategory, setFilterCategory] = useState('All');
  const [filterPriority, setFilterPriority] = useState('All');
  const [filterStatus, setFilterStatus] = useState('All');
  const [filterAnomaly, setFilterAnomaly] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [isUpdatingTicket, setIsUpdatingTicket] = useState(false);

  // Submit Complaint state
  const [complaintText, setComplaintText] = useState('');
  const [impact, setImpact] = useState('Individual');
  const [urgency, setUrgency] = useState('Normal');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionResult, setSubmissionResult] = useState(null);
  const [liveAnalysis, setLiveAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Categories & Few-Shot state
  const [categories, setCategories] = useState([]);
  const [newCatName, setNewCatName] = useState('');
  const [newCatDept, setNewCatDept] = useState('');
  const [newCatDesc, setNewCatDesc] = useState('');
  const [selectedCatForExample, setSelectedCatForExample] = useState('');
  const [newExampleText, setNewExampleText] = useState('');
  const [fewShotTestText, setFewShotTestText] = useState('');
  const [fewShotTestResult, setFewShotTestResult] = useState(null);

  // Model Evaluation state
  const [evalData, setEvalData] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);

  // Fetch dashboard metrics
  const fetchMetrics = async () => {
    try {
      setLoadingMetrics(true);
      const res = await fetch(`${API_BASE}/dashboard`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error('Failed to load metrics:', err);
    } finally {
      setLoadingMetrics(false);
    }
  };

  // Fetch tickets list with filters
  const fetchTickets = async () => {
    try {
      const params = new URLSearchParams();
      if (filterCategory !== 'All') params.append('category', filterCategory);
      if (filterPriority !== 'All') params.append('priority', filterPriority);
      if (filterStatus !== 'All') params.append('status', filterStatus);
      if (filterAnomaly === 'anomaly') params.append('is_anomaly', 'true');
      if (filterAnomaly === 'normal') params.append('is_anomaly', 'false');
      if (searchQuery.trim()) params.append('search', searchQuery.trim());
      params.append('limit', '25');
      params.append('offset', String(ticketPage * 25));

      const res = await fetch(`${API_BASE}/tickets?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setTickets(data.tickets);
        setTotalTickets(data.total);
      }
    } catch (err) {
      console.error('Failed to fetch tickets:', err);
    }
  };

  // Fetch categories
  const fetchCategories = async () => {
    try {
      const res = await fetch(`${API_BASE}/categories`);
      if (res.ok) {
        const data = await res.json();
        setCategories(data);
        if (data.length > 0 && !selectedCatForExample) {
          setSelectedCatForExample(data[0].name);
        }
      }
    } catch (err) {
      console.error('Failed to fetch categories:', err);
    }
  };

  // Fetch model evaluation and status
  const fetchEvaluation = async () => {
    try {
      const [evalRes, statusRes] = await Promise.all([
        fetch(`${API_BASE}/models/evaluation`),
        fetch(`${API_BASE}/models/status`)
      ]);
      if (evalRes.ok) {
        const d = await evalRes.json();
        setEvalData(d);
      }
      if (statusRes.ok) {
        const s = await statusRes.json();
        setModelStatus(s);
      }
    } catch (err) {
      console.error('Failed to fetch model evaluation:', err);
    }
  };

  useEffect(() => {
    fetchMetrics();
    fetchCategories();
    fetchEvaluation();
  }, []);

  useEffect(() => {
    if (activeTab === 'tickets') {
      fetchTickets();
    }
  }, [activeTab, filterCategory, filterPriority, filterStatus, filterAnomaly, searchQuery, ticketPage]);

  // Handle live AI test analyze
  const handleLiveAnalyze = async () => {
    if (!complaintText.trim() || complaintText.length < 5) return;
    try {
      setIsAnalyzing(true);
      const res = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complaint_text: complaintText,
          impact,
          urgency
        })
      });
      if (res.ok) {
        const data = await res.json();
        setLiveAnalysis(data);
      }
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle complaint submission
  const handleSubmitComplaint = async (e) => {
    e.preventDefault();
    if (!complaintText.trim() || complaintText.length < 5) return;

    try {
      setIsSubmitting(true);
      setSubmissionResult(null);
      const res = await fetch(`${API_BASE}/tickets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complaint_text: complaintText,
          impact,
          urgency
        })
      });

      if (res.ok) {
        const data = await res.json();
        setSubmissionResult(data);
        setComplaintText('');
        fetchMetrics();
      } else {
        const err = await res.json();
        alert(`Error: ${err.detail || 'Could not submit complaint'}`);
      }
    } catch (err) {
      alert(`Submission failed: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Update Ticket Details
  const handleUpdateTicket = async (ticketId, updates) => {
    try {
      setIsUpdatingTicket(true);
      const res = await fetch(`${API_BASE}/tickets/${ticketId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      if (res.ok) {
        const updated = await res.json();
        setSelectedTicket(updated);
        fetchTickets();
        fetchMetrics();
      }
    } catch (err) {
      alert(`Update failed: ${err.message}`);
    } finally {
      setIsUpdatingTicket(false);
    }
  };

  // Retrain Models
  const handleRetrain = async () => {
    try {
      setIsTraining(true);
      const res = await fetch(`${API_BASE}/models/train`, { method: 'POST' });
      if (res.ok) {
        await fetchEvaluation();
        await fetchMetrics();
        alert('Models successfully retrained and evaluated on fresh train/val/test splits!');
      } else {
        const err = await res.json();
        alert(`Training error: ${err.detail}`);
      }
    } catch (err) {
      alert(`Training request failed: ${err.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  // Upload CSV
  const handleUploadCSV = async (e) => {
    e.preventDefault();
    if (!uploadFile) return;

    const formData = new FormData();
    formData.append('file', uploadFile);

    try {
      setUploadStatus({ loading: true, msg: 'Validating and uploading CSV...' });
      const res = await fetch(`${API_BASE}/dataset/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus({ success: true, msg: data.message });
        setUploadFile(null);
        fetchMetrics();
        fetchEvaluation();
      } else {
        setUploadStatus({ error: true, msg: data.detail || 'Upload failed' });
      }
    } catch (err) {
      setUploadStatus({ error: true, msg: err.message });
    }
  };

  // Add Category
  const handleAddCategory = async (e) => {
    e.preventDefault();
    if (!newCatName.trim() || !newCatDept.trim()) return;

    try {
      const res = await fetch(`${API_BASE}/categories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newCatName.trim(),
          department: newCatDept.trim(),
          description: newCatDesc.trim()
        })
      });
      if (res.ok) {
        setNewCatName('');
        setNewCatDept('');
        setNewCatDesc('');
        fetchCategories();
        alert('Category added successfully!');
      } else {
        const err = await res.json();
        alert(`Error: ${err.detail}`);
      }
    } catch (err) {
      alert(`Failed to add category: ${err.message}`);
    }
  };

  // Add Support Example
  const handleAddExample = async (e) => {
    e.preventDefault();
    if (!selectedCatForExample || !newExampleText.trim()) return;

    try {
      const res = await fetch(`${API_BASE}/categories/${encodeURIComponent(selectedCatForExample)}/examples`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ example_text: newExampleText.trim() })
      });
      if (res.ok) {
        setNewExampleText('');
        fetchCategories();
        fetchEvaluation();
        alert(`Support example added and prototypical vector updated!`);
      }
    } catch (err) {
      alert(`Failed: ${err.message}`);
    }
  };

  // Test Few-Shot Sandbox
  const handleTestFewShot = async () => {
    if (!fewShotTestText.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ complaint_text: fewShotTestText })
      });
      if (res.ok) {
        const data = await res.json();
        setFewShotTestResult(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* App Header */}
      <header className="glass-header" style={{ position: 'sticky', top: 0, zIndex: 40 }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0.875rem 1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: 'linear-gradient(135deg, #2563eb, #7c3aed)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 15px rgba(37,99,235,0.5)' }}>
              <BrainCircuit size={22} color="#ffffff" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#f8fafc' }}>
                  SmartResolve
                </span>
                <span className="badge badge-medium" style={{ fontSize: '0.6875rem', padding: '0.125rem 0.5rem' }}>
                  SIH25195
                </span>
              </div>
              <p style={{ fontSize: '0.6875rem', color: '#94a3b8' }}>
                AI-Based Smart Helpdesk Ticketing Solution for IT Services
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {/* Role Switcher Pill */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(15,23,42,0.85)', padding: '0.3125rem 0.625rem', borderRadius: '0.625rem', border: '1px solid rgba(255,255,255,0.1)' }}>
              <span style={{ fontSize: '0.6875rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em' }}>Portal:</span>
              <select 
                value={currentRole} 
                onChange={(e) => handleRoleChange(e.target.value)}
                style={{ background: 'transparent', color: '#60a5fa', border: 'none', fontSize: '0.8125rem', fontWeight: 700, cursor: 'pointer', outline: 'none' }}
              >
                <option value="employee" style={{ background: '#0f172a', color: '#f8fafc' }}>👤 Employee / User</option>
                <option value="resolver" style={{ background: '#0f172a', color: '#f8fafc' }}>🛡️ IT Resolver / Support</option>
                <option value="ml_admin" style={{ background: '#0f172a', color: '#f8fafc' }}>⚙️ ML Admin Console</option>
              </select>
            </div>

            {/* Navigation Tabs (filtered by active role) */}
            <nav style={{ display: 'flex', gap: '0.375rem', background: 'rgba(15,23,42,0.6)', padding: '0.3125rem', borderRadius: '0.625rem', border: '1px solid rgba(255,255,255,0.06)' }}>
              {ROLES[currentRole].tabs.map(tab => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.375rem',
                      padding: '0.5rem 0.875rem',
                      fontSize: '0.8125rem',
                      fontWeight: 600,
                      borderRadius: '0.5rem',
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      background: isActive ? '#2563eb' : 'transparent',
                      color: isActive ? '#ffffff' : '#94a3b8',
                      boxShadow: isActive ? '0 2px 8px rgba(37,99,235,0.4)' : 'none'
                    }}
                  >
                    <Icon size={15} />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ maxWidth: '1400px', width: '100%', margin: '0 auto', padding: '1.75rem 1.5rem', flex: 1 }}>
        {/* TAB 1: DASHBOARD */}
        {activeTab === 'dashboard' && (
          <Dashboard 
            metrics={metrics} 
            onSelectTicket={setSelectedTicket}
            onNavigateTab={setActiveTab}
          />
        )}

        {/* TAB 2: SUBMIT COMPLAINT */}
        {activeTab === 'submit' && (
          <div style={{ maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="glass-panel" style={{ padding: '1.75rem' }}>
              <div style={{ marginBottom: '1.5rem' }}>
                <span className="badge badge-medium" style={{ marginBottom: '0.5rem' }}>Automated IT Triage</span>
                <h2 style={{ fontSize: '1.5rem', color: '#f8fafc' }}>Log an IT Complaint</h2>
                <p style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                  Input your technical issue. Sentence-BERT encodes semantics, PyTorch Autoencoder evaluates anomaly likelihood, and priority is calculated transparently.
                </p>
              </div>

              <form onSubmit={handleSubmitComplaint} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '0.375rem' }}>
                    Describe the Issue in Detail *
                  </label>
                  <textarea
                    rows={4}
                    value={complaintText}
                    onChange={(e) => setComplaintText(e.target.value)}
                    placeholder="e.g., Wi-Fi keeps dropping in conference room 3B, or ERP SAP invoice generation crashes with run-time error 1004..."
                    className="form-textarea"
                    required
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748b', marginTop: '0.375rem' }}>
                    <span>Minimum 5 characters. Natural technical phrasing supported.</span>
                    <span>{complaintText.length} characters</span>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '0.375rem' }}>
                      Reported Impact Scope
                    </label>
                    <select 
                      value={impact} 
                      onChange={(e) => setImpact(e.target.value)}
                      className="form-select"
                    >
                      <option value="Individual">Individual Workstation</option>
                      <option value="Department">Department / Team</option>
                      <option value="Enterprise">Enterprise / Organization-Wide</option>
                    </select>
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '0.375rem' }}>
                      User Urgency
                    </label>
                    <select 
                      value={urgency} 
                      onChange={(e) => setUrgency(e.target.value)}
                      className="form-select"
                    >
                      <option value="Low">Low - Normal Operations</option>
                      <option value="Normal">Normal - Standard Turnaround</option>
                      <option value="Urgent">Urgent - Work Halted / Time Sensitive</option>
                    </select>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.5rem' }}>
                  <button
                    type="submit"
                    disabled={isSubmitting || complaintText.trim().length < 5}
                    className="btn-primary"
                    style={{ flex: 1 }}
                  >
                    {isSubmitting ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                        Generating Ticket & Running AI Pipeline...
                      </>
                    ) : (
                      <>
                        <PlusCircle size={16} /> Submit & Generate Ticket
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={handleLiveAnalyze}
                    disabled={isAnalyzing || complaintText.trim().length < 5}
                    className="btn-secondary"
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Test AI Preview'}
                  </button>
                </div>
              </form>
            </div>

            {/* Live AI Analysis Preview if requested */}
            {liveAnalysis && (
              <div className="glass-panel" style={{ padding: '1.5rem', borderColor: 'rgba(59,130,246,0.4)', background: 'rgba(30,58,138,0.1)' }}>
                <h3 style={{ fontSize: '1rem', color: '#60a5fa', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <BrainCircuit size={18} /> Live AI Diagnosis Preview (Pre-submission)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
                  <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.75rem', borderRadius: '0.5rem' }}>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Predicted Domain</div>
                    <div style={{ fontWeight: 700, color: '#f8fafc' }}>{liveAnalysis.category}</div>
                    <div style={{ fontSize: '0.75rem', color: '#38bdf8' }}>Confidence: {(liveAnalysis.confidence * 100).toFixed(1)}%</div>
                  </div>
                  <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.75rem', borderRadius: '0.5rem' }}>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Autoencoder Anomaly</div>
                    <div style={{ fontWeight: 700, color: liveAnalysis.is_anomaly ? '#f87171' : '#34d399' }}>
                      {liveAnalysis.is_anomaly ? 'Unusual Complaint' : 'Normal Pattern'}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                      Error: {liveAnalysis.reconstruction_error} (Threshold: {liveAnalysis.anomaly_threshold})
                    </div>
                  </div>
                  <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.75rem', borderRadius: '0.5rem' }}>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Recommended Priority</div>
                    <div>
                      <span className={`badge badge-${liveAnalysis.priority.toLowerCase()}`}>
                        {liveAnalysis.priority}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#cbd5e1', marginTop: '0.25rem' }}>
                      Routing: {liveAnalysis.department}
                    </div>
                  </div>
                </div>
                <div style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>
                  <strong>Prioritization Rationale:</strong>
                  <ul style={{ paddingLeft: '1.25rem', marginTop: '0.25rem' }}>
                    {liveAnalysis.priority_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {/* Generated Ticket Success Card */}
            {submissionResult && (
              <div className="glass-panel" style={{ padding: '1.75rem', borderColor: 'rgba(16,185,129,0.4)', background: 'rgba(6,78,59,0.1)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                  <div style={{ background: 'rgba(16,185,129,0.2)', padding: '0.5rem', borderRadius: '50%' }}>
                    <CheckCircle2 size={24} color="#34d399" />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.25rem', color: '#34d399' }}>Ticket Generated Successfully!</h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.125rem' }}>
                      <span style={{ fontSize: '0.875rem', color: '#cbd5e1' }}>Ticket ID:</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 800, color: '#f8fafc', fontSize: '1rem', background: 'rgba(15,23,42,0.8)', padding: '0.125rem 0.5rem', borderRadius: '0.25rem' }}>
                        {submissionResult.ticket.ticket_id}
                      </span>
                    </div>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '1.25rem', background: 'rgba(15,23,42,0.7)', padding: '1rem', borderRadius: '0.75rem' }}>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Assigned Category</span>
                    <div style={{ fontWeight: 700, color: '#f8fafc' }}>{submissionResult.ticket.category}</div>
                    <span style={{ fontSize: '0.75rem', color: '#60a5fa' }}>Confidence: {(submissionResult.ticket.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Target Resolver Dept</span>
                    <div style={{ fontWeight: 700, color: '#f8fafc' }}>{submissionResult.ticket.department}</div>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Priority Assigned</span>
                    <div>
                      <span className={`badge badge-${submissionResult.ticket.priority.toLowerCase()}`}>
                        {submissionResult.ticket.priority}
                      </span>
                    </div>
                  </div>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Anomaly Status</span>
                    <div>
                      {submissionResult.ticket.is_anomaly ? (
                        <span className="badge badge-anomaly">Anomaly Detected</span>
                      ) : (
                        <span className="badge badge-normal">Normal Pattern</span>
                      )}
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
                  <button 
                    onClick={() => { setSelectedTicket(submissionResult.ticket); }} 
                    className="btn-secondary"
                    style={{ fontSize: '0.875rem' }}
                  >
                    Manage Ticket
                  </button>
                  <button 
                    onClick={() => setSubmissionResult(null)} 
                    className="btn-ghost"
                    style={{ fontSize: '0.875rem' }}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: TICKET MANAGER */}
        {activeTab === 'tickets' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* Filter Bar */}
            <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center', flex: 1 }}>
                {/* Search */}
                <div style={{ position: 'relative', minWidth: '220px' }}>
                  <Search size={16} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
                  <input
                    type="text"
                    placeholder="Search ID, text, department..."
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); setTicketPage(0); }}
                    className="form-input"
                    style={{ paddingLeft: '2.25rem' }}
                  />
                </div>

                {/* Category filter */}
                <select 
                  value={filterCategory} 
                  onChange={(e) => { setFilterCategory(e.target.value); setTicketPage(0); }}
                  className="form-select"
                  style={{ width: 'auto' }}
                >
                  <option value="All">All Categories</option>
                  {categories.map(c => (
                    <option key={c.name} value={c.name}>{c.name}</option>
                  ))}
                </select>

                {/* Priority filter */}
                <select 
                  value={filterPriority} 
                  onChange={(e) => { setFilterPriority(e.target.value); setTicketPage(0); }}
                  className="form-select"
                  style={{ width: 'auto' }}
                >
                  <option value="All">All Priorities</option>
                  <option value="Low">Low</option>
                  <option value="Medium">Medium</option>
                  <option value="High">High</option>
                  <option value="Critical">Critical</option>
                </select>

                {/* Status filter */}
                <select 
                  value={filterStatus} 
                  onChange={(e) => { setFilterStatus(e.target.value); setTicketPage(0); }}
                  className="form-select"
                  style={{ width: 'auto' }}
                >
                  <option value="All">All Statuses</option>
                  <option value="Open">Open</option>
                  <option value="In Progress">In Progress</option>
                  <option value="Resolved">Resolved</option>
                  <option value="Closed">Closed</option>
                </select>

                {/* Anomaly filter */}
                <select 
                  value={filterAnomaly} 
                  onChange={(e) => { setFilterAnomaly(e.target.value); setTicketPage(0); }}
                  className="form-select"
                  style={{ width: 'auto' }}
                >
                  <option value="All">All Reconstruction Patterns</option>
                  <option value="anomaly">Anomalies Only</option>
                  <option value="normal">Normal Only</option>
                </select>
              </div>

              <button 
                onClick={fetchTickets} 
                className="btn-secondary"
                style={{ padding: '0.5rem 0.75rem', fontSize: '0.8125rem' }}
              >
                <RefreshCw size={14} /> Refresh
              </button>
            </div>

            {/* Tickets Table */}
            <div className="glass-panel" style={{ overflow: 'hidden' }}>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Ticket ID</th>
                      <th>Complaint Description</th>
                      <th>Category</th>
                      <th>Priority</th>
                      <th>Department</th>
                      <th>Anomaly</th>
                      <th>Status</th>
                      <th>Created</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tickets.length === 0 ? (
                      <tr>
                        <td colSpan={9} style={{ textAlign: 'center', padding: '3rem', color: '#64748b' }}>
                          No tickets found matching the specified filters.
                        </td>
                      </tr>
                    ) : (
                      tickets.map((t) => (
                        <tr key={t.ticket_id} style={{ cursor: 'pointer' }} onClick={() => setSelectedTicket(t)}>
                          <td style={{ fontFamily: 'monospace', fontWeight: 600, color: '#60a5fa' }}>
                            {t.ticket_id}
                          </td>
                          <td style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {t.complaint_text}
                          </td>
                          <td>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                              <span style={{ fontSize: '0.8125rem', color: '#e2e8f0' }}>{t.category}</span>
                              {t.confidence < 0.40 && (
                                <span className="badge badge-high" style={{ fontSize: '0.625rem', padding: '0.125rem 0.375rem', width: 'fit-content' }}>
                                  ⚠️ Human Review
                                </span>
                              )}
                            </div>
                          </td>
                          <td>
                            <span className={`badge badge-${t.priority.toLowerCase()}`}>
                              {t.priority}
                            </span>
                          </td>
                          <td>
                            <span style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>{t.department}</span>
                          </td>
                          <td>
                            {t.is_anomaly ? (
                              <span className="badge badge-anomaly" title={`Recon error: ${t.reconstruction_error}`}>
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
                          <td style={{ fontSize: '0.75rem', color: '#64748b', whiteSpace: 'nowrap' }}>
                            {new Date(t.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' })}
                          </td>
                          <td>
                            <button
                              onClick={(e) => { e.stopPropagation(); setSelectedTicket(t); }}
                              className="btn-ghost"
                              style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                            >
                              Edit
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div style={{ padding: '0.875rem 1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <span style={{ fontSize: '0.8125rem', color: '#94a3b8' }}>
                  Showing {tickets.length} of {totalTickets} tickets
                </span>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    disabled={ticketPage === 0}
                    onClick={() => setTicketPage(p => Math.max(0, p - 1))}
                    className="btn-secondary"
                    style={{ padding: '0.375rem 0.75rem', fontSize: '0.75rem' }}
                  >
                    Previous
                  </button>
                  <button
                    disabled={(ticketPage + 1) * 25 >= totalTickets}
                    onClick={() => setTicketPage(p => p + 1)}
                    className="btn-secondary"
                    style={{ padding: '0.375rem 0.75rem', fontSize: '0.75rem' }}
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: MODEL EVALUATION & DATASET */}
        {activeTab === 'models' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {/* Header with Retrain Action */}
            <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <span className="badge badge-normal" style={{ marginBottom: '0.375rem' }}>Evaluation Sandbox</span>
                <h2 style={{ fontSize: '1.5rem', color: '#f8fafc' }}>Machine Learning Pipeline & Evaluation</h2>
                <p style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                  Strict train/val/test splits (70/15/15). Metrics calculated only on held-out test splits without data leakage.
                </p>
              </div>
              <button
                onClick={handleRetrain}
                disabled={isTraining}
                className="btn-primary"
              >
                {isTraining ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                    Retraining & Calibrating...
                  </>
                ) : (
                  <>
                    <RefreshCw size={16} /> Retrain All Models
                  </>
                )}
              </button>
            </div>

            {/* Model Architecture & Calibration Overview Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
              {/* Module A */}
              <div className="glass-panel" style={{ padding: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#60a5fa', fontWeight: 600 }}>MODULE A</span>
                    <h3 style={{ fontSize: '1.125rem', color: '#f8fafc', marginTop: '0.125rem' }}>Transformer Classifier</h3>
                    <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginTop: '0.25rem' }}>
                      sentence-transformers/all-MiniLM-L6-v2 + Logistic Regression
                    </p>
                  </div>
                  <span className="badge badge-normal">Ready</span>
                </div>
                {evalData?.metrics?.classifier && (
                  <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                    <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.5rem 0.75rem', borderRadius: '0.375rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Test Accuracy</span>
                      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                        {(evalData.metrics.classifier.accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.5rem 0.75rem', borderRadius: '0.375rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Macro F1-Score</span>
                      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                        {(evalData.metrics.classifier.macro_f1 * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Module B */}
              <div className="glass-panel" style={{ padding: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#a78bfa', fontWeight: 600 }}>MODULE B</span>
                    <h3 style={{ fontSize: '1.125rem', color: '#f8fafc', marginTop: '0.125rem' }}>PyTorch Autoencoder</h3>
                    <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginTop: '0.25rem' }}>
                      384-dim embedding reconstruction for out-of-distribution anomaly detection
                    </p>
                  </div>
                  <span className="badge badge-medium">
                    {modelStatus?.is_ae_calibrated ? 'Calibrated' : 'Uncalibrated'}
                  </span>
                </div>
                {evalData?.metrics?.autoencoder && (
                  <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                    <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.5rem 0.75rem', borderRadius: '0.375rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Val Threshold</span>
                      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                        {evalData.metrics.autoencoder.calibrated_threshold}
                      </div>
                    </div>
                    <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.5rem 0.75rem', borderRadius: '0.375rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Test ROC-AUC</span>
                      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                        {typeof evalData.metrics.autoencoder.roc_auc === 'number' 
                          ? (evalData.metrics.autoencoder.roc_auc * 100).toFixed(1) + '%'
                          : 'N/A'}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Module C */}
              <div className="glass-panel" style={{ padding: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span style={{ fontSize: '0.75rem', color: '#34d399', fontWeight: 600 }}>MODULE C</span>
                    <h3 style={{ fontSize: '1.125rem', color: '#f8fafc', marginTop: '0.125rem' }}>Prototypical Few-Shot</h3>
                    <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginTop: '0.25rem' }}>
                      Cosine similarity against average support centroid vectors
                    </p>
                  </div>
                  <span className="badge badge-normal">Active</span>
                </div>
                {evalData?.metrics?.few_shot && (
                  <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                    <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.5rem 0.75rem', borderRadius: '0.375rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Few-Shot Acc</span>
                      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                        {(evalData.metrics.few_shot.accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div style={{ background: 'rgba(15,23,42,0.6)', padding: '0.5rem 0.75rem', borderRadius: '0.375rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Active Prototypes</span>
                      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                        {evalData.metrics.few_shot.num_prototypes}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Per-Class Evaluation Breakdown Table */}
            {evalData?.metrics?.classifier?.per_class && (
              <div className="glass-panel" style={{ padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1.125rem', color: '#f8fafc', marginBottom: '0.25rem' }}>
                  Test Split Per-Category Classification Performance
                </h3>
                <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginBottom: '1rem' }}>
                  Detailed breakdown on the held-out test split (test samples: {evalData.metrics.classifier.test_samples})
                </p>
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Category</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Test Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(evalData.metrics.classifier.per_class).map(([cls, scores]) => (
                        <tr key={cls}>
                          <td style={{ fontWeight: 600, color: '#f1f5f9' }}>{cls}</td>
                          <td>{(scores.precision * 100).toFixed(1)}%</td>
                          <td>{(scores.recall * 100).toFixed(1)}%</td>
                          <td>
                            <span style={{ fontWeight: 700, color: scores.f1 >= 0.7 ? '#34d399' : '#fbbf24' }}>
                              {(scores.f1 * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td style={{ color: '#94a3b8' }}>{scores.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* CSV Dataset Uploader */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.125rem', color: '#f8fafc', marginBottom: '0.25rem' }}>
                Import Complaint Dataset (CSV)
              </h3>
              <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginBottom: '1rem' }}>
                Upload new training data. Expected columns: <code>complaint_text</code>, <code>category</code>, and optionally <code>is_normal</code>, <code>priority</code>, <code>impact</code>.
              </p>

              <form onSubmit={handleUploadCSV} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setUploadFile(e.target.files[0])}
                  className="form-input"
                  style={{ width: 'auto', flex: 1, minWidth: '240px' }}
                />
                <button
                  type="submit"
                  disabled={!uploadFile || uploadStatus?.loading}
                  className="btn-primary"
                >
                  <Upload size={16} /> Upload & Ingest
                </button>
              </form>

              {uploadStatus && (
                <div style={{ marginTop: '0.75rem', fontSize: '0.8125rem', color: uploadStatus.error ? '#f87171' : uploadStatus.success ? '#34d399' : '#94a3b8' }}>
                  {uploadStatus.msg}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 5: FEW-SHOT CATEGORY STUDIO */}
        {activeTab === 'fewshot' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <span className="badge badge-normal" style={{ marginBottom: '0.5rem' }}>Prototypical Embedding Space</span>
              <h2 style={{ fontSize: '1.5rem', color: '#f8fafc' }}>Few-Shot Category Studio</h2>
              <p style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                Dynamically create new IT categories and supply 1 to 5 labelled support examples. Prototype vectors are recomputed instantly in embedding space using sentence-transformers.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
              {/* Add New Category */}
              <div className="glass-panel" style={{ padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', color: '#f8fafc', marginBottom: '1rem' }}>Create New IT Category</h3>
                <form onSubmit={handleAddCategory} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Category Name *</label>
                    <input 
                      type="text" 
                      placeholder="e.g., Cloud & Virtualization" 
                      value={newCatName} 
                      onChange={(e) => setNewCatName(e.target.value)}
                      className="form-input" 
                      required 
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Target Resolver Department *</label>
                    <input 
                      type="text" 
                      placeholder="e.g., Cloud Operations Group" 
                      value={newCatDept} 
                      onChange={(e) => setNewCatDept(e.target.value)}
                      className="form-input" 
                      required 
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Scope / Description</label>
                    <input 
                      type="text" 
                      placeholder="e.g., AWS, Azure, VMware ESXi issues" 
                      value={newCatDesc} 
                      onChange={(e) => setNewCatDesc(e.target.value)}
                      className="form-input" 
                    />
                  </div>
                  <button type="submit" className="btn-primary" style={{ marginTop: '0.5rem' }}>
                    <PlusCircle size={16} /> Register Category
                  </button>
                </form>
              </div>

              {/* Add Support Example */}
              <div className="glass-panel" style={{ padding: '1.25rem' }}>
                <h3 style={{ fontSize: '1rem', color: '#f8fafc', marginBottom: '1rem' }}>Add Labelled Support Example</h3>
                <form onSubmit={handleAddExample} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Target Category *</label>
                    <select 
                      value={selectedCatForExample} 
                      onChange={(e) => setSelectedCatForExample(e.target.value)}
                      className="form-select"
                    >
                      {categories.map(c => (
                        <option key={c.name} value={c.name}>
                          {c.name} ({c.support_count || 0} examples)
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Complaint Example Text *</label>
                    <textarea 
                      rows={3}
                      placeholder="e.g., AWS EC2 instance kernel panic after rebooting during scheduled maintenance..."
                      value={newExampleText} 
                      onChange={(e) => setNewExampleText(e.target.value)}
                      className="form-textarea" 
                      required 
                    />
                  </div>
                  <button type="submit" className="btn-secondary" style={{ marginTop: '0.5rem' }}>
                    <Sparkles size={16} /> Update Prototype Vector
                  </button>
                </form>
              </div>
            </div>

            {/* Few-Shot Sandbox Tester */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.125rem', color: '#f8fafc', marginBottom: '0.25rem' }}>
                Few-Shot Classification Testing Sandbox
              </h3>
              <p style={{ fontSize: '0.8125rem', color: '#94a3b8', marginBottom: '1rem' }}>
                Test any arbitrary complaint to observe real-time cosine similarity scores across all active category prototypes.
              </p>

              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <input
                  type="text"
                  placeholder="Enter a test complaint (e.g. AWS S3 storage quota exceeded)..."
                  value={fewShotTestText}
                  onChange={(e) => setFewShotTestText(e.target.value)}
                  className="form-input"
                  style={{ flex: 1 }}
                />
                <button
                  onClick={handleTestFewShot}
                  disabled={!fewShotTestText.trim()}
                  className="btn-primary"
                >
                  Evaluate Similarities
                </button>
              </div>

              {fewShotTestResult && (
                <div style={{ marginTop: '1.25rem', background: 'rgba(15,23,42,0.7)', padding: '1.25rem', borderRadius: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Predicted Few-Shot Category</span>
                      <h4 style={{ fontSize: '1.25rem', color: '#60a5fa' }}>{fewShotTestResult.few_shot_category}</h4>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Cosine Match Score</span>
                      <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#34d399' }}>
                        {(fewShotTestResult.few_shot_similarity * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {fewShotTestResult.classification_details?.probability_distribution && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 600 }}>Domain Similarity Distribution:</span>
                      {Object.entries(fewShotTestResult.classification_details.probability_distribution).map(([cat, score]) => (
                        <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8125rem' }}>
                          <span style={{ width: '200px', color: '#cbd5e1', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                            {cat}
                          </span>
                          <div style={{ flex: 1, background: 'rgba(255,255,255,0.06)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                            <div 
                              style={{ 
                                width: `${Math.min(100, score * 100)}%`, 
                                height: '100%', 
                                background: cat === fewShotTestResult.category ? '#3b82f6' : '#64748b' 
                              }} 
                            />
                          </div>
                          <span style={{ width: '45px', textAlign: 'right', color: '#94a3b8', fontFamily: 'monospace' }}>
                            {(score * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Ticket Details & Management Modal */}
      {selectedTicket && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(6px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50, padding: '1rem' }}>
          <div className="glass-panel" style={{ width: '100%', maxWidth: '750px', maxHeight: '90vh', overflowY: 'auto', padding: '1.75rem', position: 'relative' }}>
            {/* Close Button */}
            <button
              onClick={() => setSelectedTicket(null)}
              style={{ position: 'absolute', right: '1.25rem', top: '1.25rem', background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
            >
              <X size={20} />
            </button>

            {/* Modal Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
              <span style={{ fontFamily: 'monospace', fontWeight: 800, fontSize: '1.25rem', color: '#60a5fa' }}>
                {selectedTicket.ticket_id}
              </span>
              <span className={`badge badge-${selectedTicket.priority.toLowerCase()}`}>
                {selectedTicket.priority}
              </span>
              <span className={`badge badge-status-${selectedTicket.status.toLowerCase().replace(' ', '')}`}>
                {selectedTicket.status}
              </span>
              {selectedTicket.is_anomaly && (
                <span className="badge badge-anomaly">Anomaly</span>
              )}
            </div>

            {/* Complaint Text */}
            <div style={{ background: 'rgba(15,23,42,0.6)', padding: '1rem', borderRadius: '0.5rem', marginBottom: '1.25rem' }}>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>Original Complaint:</div>
              <p style={{ color: '#f8fafc', fontSize: '0.9375rem', lineHeight: 1.5 }}>
                {selectedTicket.complaint_text}
              </p>
            </div>

            {/* AI Diagnosis Insights */}
            {currentRole === 'employee' ? (
              <div style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: '0.5rem', padding: '1rem', marginBottom: '1.25rem', background: 'rgba(30,41,59,0.3)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', fontSize: '0.8125rem' }}>
                  <div>
                    <span style={{ color: '#94a3b8' }}>Classified Category:</span>
                    <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{selectedTicket.category}</div>
                  </div>
                  <div>
                    <span style={{ color: '#94a3b8' }}>Assigned Resolver Group:</span>
                    <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{selectedTicket.department}</div>
                  </div>
                  <div>
                    <span style={{ color: '#94a3b8' }}>Priority Level:</span>
                    <div>
                      <span className={`badge badge-${selectedTicket.priority.toLowerCase()}`}>
                        {selectedTicket.priority}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: '0.5rem', padding: '1rem', marginBottom: '1.25rem', background: 'rgba(30,41,59,0.3)' }}>
                <h4 style={{ fontSize: '0.875rem', color: '#38bdf8', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <BrainCircuit size={16} /> AI Pipeline Diagnostic Record
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', fontSize: '0.8125rem' }}>
                  <div>
                    <span style={{ color: '#94a3b8' }}>Classified Category:</span>
                    <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{selectedTicket.category}</div>
                    <span style={{ fontSize: '0.75rem', color: '#60a5fa' }}>Confidence: {(selectedTicket.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span style={{ color: '#94a3b8' }}>Reconstruction Error:</span>
                    <div style={{ fontWeight: 600, color: selectedTicket.is_anomaly ? '#f87171' : '#34d399' }}>
                      {selectedTicket.reconstruction_error}
                    </div>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Threshold: {selectedTicket.anomaly_threshold}</span>
                  </div>
                  <div>
                    <span style={{ color: '#94a3b8' }}>Assigned Group:</span>
                    <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{selectedTicket.department}</div>
                  </div>
                </div>

                {selectedTicket.priority_reasons && selectedTicket.priority_reasons.length > 0 && (
                  <div style={{ marginTop: '0.75rem', fontSize: '0.8125rem' }}>
                    <span style={{ color: '#94a3b8' }}>Prioritization Rationale:</span>
                    <ul style={{ paddingLeft: '1.25rem', color: '#cbd5e1', marginTop: '0.25rem' }}>
                      {selectedTicket.priority_reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Controls based on role */}
            {currentRole === 'employee' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '1.25rem' }}>
                <div style={{ background: 'rgba(37,99,235,0.08)', border: '1px solid rgba(59,130,246,0.25)', padding: '0.875rem', borderRadius: '0.5rem', fontSize: '0.8125rem', color: '#93c5fd' }}>
                  <strong>Ticket Status:</strong> Your issue is being handled by <strong>{selectedTicket.department}</strong>.
                </div>
                {selectedTicket.admin_notes ? (
                  <div style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', padding: '0.875rem', borderRadius: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: '#34d399', fontWeight: 700, textTransform: 'uppercase' }}>Resolution Notes:</span>
                    <p style={{ color: '#f8fafc', fontSize: '0.875rem', marginTop: '0.25rem' }}>{selectedTicket.admin_notes}</p>
                  </div>
                ) : (
                  <div style={{ fontSize: '0.8125rem', color: '#94a3b8', fontStyle: 'italic' }}>
                    No technician notes posted yet. The ticket is currently <strong>{selectedTicket.status}</strong>.
                  </div>
                )}
              </div>
            ) : (
              <>
                {/* Editable Management Controls for Resolvers / Admins */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.8125rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                      Change Status
                    </label>
                    <select
                      value={selectedTicket.status}
                      onChange={(e) => handleUpdateTicket(selectedTicket.ticket_id, { status: e.target.value })}
                      disabled={isUpdatingTicket}
                      className="form-select"
                    >
                      <option value="Open">Open</option>
                      <option value="In Progress">In Progress</option>
                      <option value="Resolved">Resolved</option>
                      <option value="Closed">Closed</option>
                    </select>
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: '0.8125rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                      Override Priority
                    </label>
                    <select
                      value={selectedTicket.priority}
                      onChange={(e) => handleUpdateTicket(selectedTicket.ticket_id, { priority: e.target.value })}
                      disabled={isUpdatingTicket}
                      className="form-select"
                    >
                      <option value="Low">Low</option>
                      <option value="Medium">Medium</option>
                      <option value="High">High</option>
                      <option value="Critical">Critical</option>
                    </select>
                  </div>
                </div>

                {/* Admin Resolution Notes */}
                <div style={{ marginBottom: '1.25rem' }}>
                  <label style={{ display: 'block', fontSize: '0.8125rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                    Resolution / Administrator Notes
                  </label>
                  <textarea
                    rows={3}
                    defaultValue={selectedTicket.admin_notes || ''}
                    onBlur={(e) => handleUpdateTicket(selectedTicket.ticket_id, { admin_notes: e.target.value })}
                    placeholder="Enter technician notes, diagnostic updates, or resolution confirmation..."
                    className="form-textarea"
                  />
                </div>
              </>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button
                onClick={() => setSelectedTicket(null)}
                className="btn-primary"
              >
                Close Window
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
