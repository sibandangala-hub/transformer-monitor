import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import { urgCls, stateCls, pretty, healthText } from '../utils/statusHelpers';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler, zoomPlugin);

const baseDataset = (label, borderColor, bgColor) => ({
  label, borderColor, backgroundColor: bgColor,
  borderWidth: 2, tension: 0.35, fill: true, pointRadius: 2, pointHoverRadius: 4,
});

const chartOpts = (extraPlugins = {}) => ({
  responsive: true, maintainAspectRatio: false, animation: false,
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { size: 10 } } },
    zoom: {
      zoom:  { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
      pan:   { enabled: true, mode: 'x' },
    },
    ...extraPlugins,
  },
  scales: {
    x: { ticks: { color: '#64748b', maxRotation: 0, autoSkip: true, maxTicksLimit: 4 }, grid: { color: '#1a2640' } },
    y: { ticks: { color: '#64748b' }, grid: { color: '#1a2640' } },
  },
});

// threshold annotation line on recon error chart
const reconOpts = (adaptive, failure) => ({
  ...chartOpts(),
  plugins: {
    ...chartOpts().plugins,
    annotation: undefined, // handled via dataset instead
  },
});

function trendArrow(curr, prev, key) {
  if (prev === null || prev === undefined) return null;
  const c = Number(curr ?? 0);
  const p = Number(prev[key] ?? 0);
  const diff = c - p;
  if (Math.abs(diff) < 0.001) return <span style={{ color: '#64748b' }}>→</span>;
  return diff > 0
    ? <span style={{ color: '#ef4444', fontSize: 13 }}>↑ +{diff.toFixed(3)}</span>
    : <span style={{ color: '#22c55e', fontSize: 13 }}>↓ {diff.toFixed(3)}</span>;
}

export default function Overview({ data, history, prevValues }) {
  if (!data) return (
    <section className="page active">
      <div className="page-head"><h2>Overview</h2><p>Waiting for live data…</p></div>
    </section>
  );

  const lv     = data.latest_values || {};
  const curr   = Number(lv.current      ?? 0);
  const oilT   = Number(lv.oil_temp     ?? 0);
  const winT   = Number(lv.winding_temp ?? 0);
  const vib    = Number(lv.vibration    ?? 0);
  const oil    = Number(lv.oil_level    ?? 0);
  const health = Number(data.health     ?? 0);
  const rul    = Number(data.rul_hours  ?? 0);
  const mps    = Number(data.maintenance_priority_score ?? 0);
  const uCls   = urgCls(data.urgency_level);
  const re     = Number(data.reconstruction_error ?? -1);
  const se     = Number(data.smoothed_error       ?? -1);
  const at     = Number(data.adaptive_threshold   ?? data.threshold ?? 0);
  const ft     = Number(data.failure_threshold    ?? 0);

  const mkDataset = (key, label, border, bg) => ({
    labels: history.labels,
    datasets: [{ ...baseDataset(label, border, bg), data: history.series[key] }],
  });

  // recon error chart with threshold lines as extra datasets
  const reconData = {
    labels: history.labels,
    datasets: [
      { ...baseDataset('Recon Error', '#38bdf8', 'rgba(56,189,248,.08)'), data: history.series.current.map(() => re < 0 ? null : re) },
      { label: 'Adaptive Threshold', borderColor: '#facc15', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4,3], pointRadius: 0, tension: 0, fill: false, data: history.labels.map(() => at > 0 ? at : null) },
      { label: 'Failure Threshold',  borderColor: '#ef4444', backgroundColor: 'transparent', borderWidth: 1, borderDash: [4,3], pointRadius: 0, tension: 0, fill: false, data: history.labels.map(() => ft > 0 ? ft : null) },
    ],
  };

  const sensors = [
    { title: 'Current',      val: `${curr.toFixed(2)} A`,  state: data.current_state,      lv_key: 'current' },
    { title: 'Oil Temp',     val: `${oilT.toFixed(1)} °C`, state: data.oil_temp_state,     lv_key: 'oil_temp' },
    { title: 'Winding Temp', val: `${winT.toFixed(1)} °C`, state: data.winding_temp_state, lv_key: 'winding_temp' },
    { title: 'Vibration',    val: vib.toFixed(4),          state: data.vibration_state,    lv_key: 'vibration' },
    { title: 'Oil Level',    val: `${oil.toFixed(1)} %`,   state: data.oil_level_state,    lv_key: 'oil_level' },
  ];

  return (
    <section className="page active">
      <div className="page-head"><h2>Overview</h2><p>Live transformer status · scroll/pinch charts to zoom · drag to pan</p></div>
      <div className="ov-wrap">

        {/* KPI row */}
        <div className="ov-kpi">
          <div className="card">
            <div className="card-title">System Status</div>
            <div className={`big ${uCls}`}>{data.is_anomaly ? pretty(data.urgency_level) : 'NORMAL'}</div>
            <div className="sub">{data.is_anomaly ? 'Anomaly detected — see prescription' : 'No anomaly detected'}</div>
          </div>
          <div className="card">
            <div className="card-title">Health Index</div>
            <div className="big" style={{ color: health >= 80 ? '#22c55e' : health >= 50 ? '#facc15' : '#ef4444' }}>{health.toFixed(1)}%</div>
            <div className="barwrap"><div className="bar" style={{ width: `${Math.max(0, Math.min(100, health))}%` }} /></div>
            <div className="sub" style={{ color: health >= 80 ? '#22c55e' : health >= 50 ? '#facc15' : '#ef4444' }}>{healthText(health)}</div>
          </div>
          <div className="card">
            <div className="card-title">RUL Estimate</div>
            <div className="big">{rul.toFixed(1)} h</div>
            <div className="sub">{pretty(data.rul_state)}</div>
          </div>
          <div className="card">
            <div className="card-title">Priority Score</div>
            <div className="big">{mps.toFixed(1)}</div>
            <div className={`sub ${uCls}`}>{pretty(data.urgency_level)}</div>
          </div>
          <div className="card">
            <div className="card-title">Main Cause</div>
            <div className="big">{(data.main_cause || '--').replace(/_/g, ' ').toUpperCase()}</div>
            <div className="sub">Dominant: {pretty(data.dominant_state)}</div>
          </div>
          <div className="card">
            <div className="card-title">Operating Region</div>
            <div className="big" style={{ fontSize: 13 }}>{pretty(data.operating_region)}</div>
            <div className="sub">{data.prescription_context || '--'}</div>
          </div>
        </div>

        {/* Sensor row with trend arrows */}
        <div className="ov-sensors">
          {sensors.map(({ title, val, state, lv_key }) => (
            <div className="card" key={title}>
              <div className="card-title">{title}</div>
              <div className="metric">{val}</div>
              <div className={`mini ${stateCls(state)}`} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>State: {pretty(state)}</span>
                <span>{trendArrow(lv[lv_key], prevValues, lv_key)}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Charts row — zoom/pan enabled */}
        <div className="ov-charts">
          {[
            { key: 'current',      label: 'Current (A)',       border: '#38bdf8', bg: 'rgba(56,189,248,.10)' },
            { key: 'oil_temp',     label: 'Oil Temp (°C)',     border: '#f97316', bg: 'rgba(249,115,22,.10)' },
            { key: 'winding_temp', label: 'Winding Temp (°C)', border: '#f43f5e', bg: 'rgba(244,63,94,.10)' },
            { key: 'vibration',    label: 'Vibration',         border: '#a855f7', bg: 'rgba(168,85,247,.10)' },
            { key: 'oil_level',    label: 'Oil Level (%)',     border: '#22c55e', bg: 'rgba(34,197,94,.10)' },
          ].map(({ key, label, border, bg }) => (
            <div className="card chart-card" key={key}>
              <div className="chart-inner">
                <Line data={mkDataset(key, label, border, bg)} options={chartOpts()} />
              </div>
            </div>
          ))}
        </div>

        {/* Diagnostics row — recon error with threshold lines */}
        <div className="ov-diag">
          <div className="card chart-card" style={{ gridColumn: 'span 2' }}>
            <div className="card-title">Reconstruction Error vs Thresholds <span style={{ color: '#64748b', fontSize: 9 }}>(scroll to zoom)</span></div>
            <div style={{ height: 80 }}>
              <Line data={reconData} options={chartOpts()} />
            </div>
          </div>
          <div className="card">
            <div className="card-title">Trend / Persistence</div>
            <div className="big mono">{Number(data.degradation_rate ?? 0).toFixed(6)}</div>
            <div className="sub mono">Persistence: {(Number(data.persistence_factor ?? 0) * 100).toFixed(1)}%</div>
            <div className="mini">Mode: {pretty(data.threshold_mode)}</div>
            <div className="mini">Adaptive history: {data.adaptive_history_count ?? 0} pts</div>
          </div>
        </div>

      </div>
    </section>
  );
}
