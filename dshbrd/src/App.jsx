import { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { useTransformerData } from './hooks/useTransformerData';
import { downloadReport }     from './utils/reportDownload';
import { fmt, urgCls }        from './utils/statusHelpers';

const Overview         = lazy(() => import('./pages/Overview'));
const Prescription     = lazy(() => import('./pages/Prescription'));
const Contribution     = lazy(() => import('./pages/Contribution'));
const SensorIntel      = lazy(() => import('./pages/SensorIntel'));
const PowerHardware    = lazy(() => import('./pages/PowerHardware'));
const SystemPage       = lazy(() => import('./pages/SystemPage'));
const HistoricalTrends = lazy(() => import('./pages/HistoricalTrends'));

const TABS = [
  { id: 'overview',     label: 'Overview' },
  { id: 'prescription', label: 'Prescription' },
  { id: 'contribution', label: 'Sensor Contribution' },
  { id: 'sensorintel',  label: 'Sensor Intelligence' },
  { id: 'hardware',     label: 'Hardware' },
  { id: 'history',      label: '📈 Historical Trends' },
  { id: 'system',       label: 'System Status' },
];

export default function App() {
  const { data, connected, stale, lastPacketMs, history, prevValues, fetchHistory } = useTransformerData();
  const [tab, setTab]     = useState('overview');
  const analysisCount     = useRef(0);

  // ── theme: persist to localStorage ──
  const [theme, setTheme] = useState(() => localStorage.getItem('thm-theme') || 'dark');
  const toggleTheme = () => setTheme(t => {
    const next = t === 'dark' ? 'light' : 'dark';
    localStorage.setItem('thm-theme', next);
    return next;
  });
  useEffect(() => { document.documentElement.setAttribute('data-theme', theme); }, [theme]);

  // ── tab title update ──
  useEffect(() => {
    if (!data) { document.title = 'THM v2'; return; }
    const urgency = data.urgency_level || 'NORMAL';
    const icon    = urgency === 'CRITICAL' ? '🔴' : urgency === 'URGENT' ? '🟠' : urgency === 'WARNING' ? '🟡' : '🟢';
    document.title = `${icon} ${urgency} | THM v2`;
  }, [data]);

  useEffect(() => { if (data) analysisCount.current += 1; }, [data]);

  const urgency = data?.urgency_level || '';
  const uCls    = urgCls(urgency);

  return (
    <>
      <div className="topbar">
        <div className="brand-mark">⚡ THM <span>v2</span></div>
        <div className="nav-tabs">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`nav-tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          ))}
        </div>
        <div className="topbar-right">
          {stale && (
            <div className="stale-badge">⚠ STALE DATA</div>
          )}
          <div className={`badge-live ${connected && !stale ? '' : 'off'}`}>
            {connected ? `Live · ${fmt(data?.analysis_timestamp)}` : 'No data'}
          </div>
          <button className="btn-report" onClick={() => downloadReport(data)}>⬇ Report</button>
          <button className="btn-theme" onClick={toggleTheme}>{theme === 'dark' ? '🌙' : '☀️'}</button>
        </div>
      </div>

      {/* stale data full banner */}
      {stale && (
        <div className="stale-banner">
          ⚠ No data received from ESP32 in the last 3 minutes — connection may be lost
        </div>
      )}

      <main className="main" style={{ top: stale ? 84 : 52 }}>
        <Suspense fallback={null}>
          {tab === 'overview'     && <Overview      data={data} history={history} prevValues={prevValues} />}
          {tab === 'prescription' && <Prescription  data={data} />}
          {tab === 'contribution' && <Contribution  data={data} />}
          {tab === 'sensorintel'  && <SensorIntel   data={data} />}
          {tab === 'hardware'     && <PowerHardware data={data} />}
          {tab === 'history'      && <HistoricalTrends fetchHistory={fetchHistory} />}
          {tab === 'system'       && <SystemPage    data={data} connected={connected} stale={stale} lastPacketMs={lastPacketMs} analysisCount={analysisCount.current} />}
        </Suspense>
      </main>
    </>
  );
}