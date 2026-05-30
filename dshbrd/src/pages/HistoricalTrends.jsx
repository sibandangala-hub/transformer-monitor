import { useState, useEffect, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler, zoomPlugin);

const RANGES = [
  { label: '30 min', ms: 30 * 60 * 1000 },
  { label: '1 h',    ms: 60 * 60 * 1000 },
  { label: '6 h',    ms: 6  * 60 * 60 * 1000 },
];

const chartOpts = {
  responsive: true, maintainAspectRatio: false, animation: false,
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { size: 10 } } },
    zoom: {
      zoom:  { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
      pan:   { enabled: true, mode: 'x' },
    },
  },
  scales: {
    x: { ticks: { color: '#64748b', maxRotation: 0, autoSkip: true, maxTicksLimit: 6 }, grid: { color: '#1a2640' } },
    y: { ticks: { color: '#64748b' }, grid: { color: '#1a2640' } },
  },
};

const ds = (label, data, border, bg, extra = {}) => ({
  label, data, borderColor: border, backgroundColor: bg,
  borderWidth: 2, tension: 0.35, fill: !!bg, pointRadius: 1, pointHoverRadius: 3, ...extra,
});

function fmt(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function csvDownload(pts) {
  if (!pts.length) return;
  const header = 'timestamp,datetime,current,oil_temp,winding_temp,vibration,oil_level,health,recon_error,threshold\n';
  const rows = pts.map(p =>
    `${p.ts},${new Date(p.ts * 1000).toISOString()},${p.current},${p.oil_temp},${p.winding_temp},${p.vibration},${p.oil_level},${p.health},${p.recon_error},${p.threshold}`
  ).join('\n');
  const blob = new Blob([header + rows], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `thm_history_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function HistoricalTrends({ fetchHistory }) {
  const [rangeIdx, setRangeIdx] = useState(0);
  const [pts, setPts]           = useState([]);
  const [loading, setLoading]   = useState(false);

  const load = useCallback(async (idx) => {
    setLoading(true);
    const data = await fetchHistory(RANGES[idx].ms);
    setPts(data);
    setLoading(false);
  }, [fetchHistory]);

  useEffect(() => { load(rangeIdx); }, [rangeIdx, load]);

  const labels   = pts.map(p => fmt(p.ts));
  const noData   = pts.length === 0;

  // temperature delta: oil_temp - winding_temp
  const deltaData = pts.map(p => Number((p.oil_temp - p.winding_temp).toFixed(3)));

  // recon error vs thresholds
  const reconData = {
    labels,
    datasets: [
      ds('Recon Error',         pts.map(p => p.recon_error), '#38bdf8', 'rgba(56,189,248,.08)'),
      ds('Adaptive Threshold',  pts.map(p => p.threshold),   '#facc15', 'transparent', { borderDash: [4,3], pointRadius: 0, fill: false }),
    ],
  };

  return (
    <section className="page active">
      <div className="page-head">
        <h2>Historical Trends</h2>
        <p>Sensor history stored in Firebase RTDB · scroll/pinch to zoom · drag to pan</p>
      </div>

      {/* toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 4 }}>
          {RANGES.map((r, i) => (
            <button
              key={r.label}
              className={`nav-tab ${rangeIdx === i ? 'active' : ''}`}
              onClick={() => setRangeIdx(i)}
            >{r.label}</button>
          ))}
        </div>
        <span className="mini" style={{ marginLeft: 8 }}>
          {loading ? 'Loading…' : `${pts.length} points`}
        </span>
        <button
          className="btn-report"
          style={{ marginLeft: 'auto' }}
          onClick={() => csvDownload(pts)}
          disabled={noData}
        >⬇ CSV</button>
      </div>

      {noData && !loading && (
        <div className="card" style={{ textAlign: 'center', padding: 24 }}>
          <div className="sub">No history data for this range yet. Data accumulates as the system runs.</div>
        </div>
      )}

      {!noData && (
        <div className="hist-grid">

          {/* Oil Temp vs Winding Temp */}
          <div className="card chart-card hist-tall">
            <div className="card-title">Oil Temp vs Winding Temp (°C)</div>
            <div className="chart-inner">
              <Line data={{
                labels,
                datasets: [
                  ds('Oil Temp (°C)',     pts.map(p => p.oil_temp),     '#f97316', 'rgba(249,115,22,.08)'),
                  ds('Winding Temp (°C)', pts.map(p => p.winding_temp), '#f43f5e', 'rgba(244,63,94,.08)'),
                ],
              }} options={chartOpts} />
            </div>
          </div>

          {/* Temperature Delta */}
          <div className="card chart-card hist-tall">
            <div className="card-title">
              Temp Delta: Oil − Winding (°C)
              <span style={{ marginLeft: 8, color: '#64748b', fontSize: 9 }}>positive = oil hotter than winding</span>
            </div>
            <div className="chart-inner">
              <Line data={{
                labels,
                datasets: [
                  ds('Δ Temp', deltaData, '#a855f7', 'rgba(168,85,247,.08)'),
                  // zero reference line
                  ds('Zero', labels.map(() => 0), '#64748b', 'transparent', { borderDash: [3,3], pointRadius: 0, fill: false, borderWidth: 1 }),
                ],
              }} options={chartOpts} />
            </div>
          </div>

          {/* Current */}
          <div className="card chart-card hist-tall">
            <div className="card-title">Current (A)</div>
            <div className="chart-inner">
              <Line data={{ labels, datasets: [ds('Current (A)', pts.map(p => p.current), '#38bdf8', 'rgba(56,189,248,.08)')] }} options={chartOpts} />
            </div>
          </div>

          {/* Vibration */}
          <div className="card chart-card hist-tall">
            <div className="card-title">Vibration</div>
            <div className="chart-inner">
              <Line data={{ labels, datasets: [ds('Vibration', pts.map(p => p.vibration), '#a855f7', 'rgba(168,85,247,.08)')] }} options={chartOpts} />
            </div>
          </div>

          {/* Oil Level */}
          <div className="card chart-card hist-tall">
            <div className="card-title">Oil Level (%)</div>
            <div className="chart-inner">
              <Line data={{ labels, datasets: [ds('Oil Level (%)', pts.map(p => p.oil_level), '#22c55e', 'rgba(34,197,94,.08)')] }} options={chartOpts} />
            </div>
          </div>

          {/* Health */}
          <div className="card chart-card hist-tall">
            <div className="card-title">Health Index (%)</div>
            <div className="chart-inner">
              <Line data={{ labels, datasets: [ds('Health (%)', pts.map(p => p.health), '#22c55e', 'rgba(34,197,94,.08)')] }} options={chartOpts} />
            </div>
          </div>

          {/* Recon error vs threshold — full width */}
          <div className="card chart-card hist-wide">
            <div className="card-title">
              Reconstruction Error vs Adaptive Threshold
              <span style={{ marginLeft: 8, color: '#64748b', fontSize: 9 }}>dashed = threshold line</span>
            </div>
            <div className="chart-inner">
              <Line data={reconData} options={chartOpts} />
            </div>
          </div>

        </div>
      )}
    </section>
  );
}
