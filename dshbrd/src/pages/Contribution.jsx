import { pretty } from '../utils/statusHelpers';

const SENSORS = [
  { key: 'current',      label: 'Current',              cls: 'pf-current' },
  { key: 'oil_temp',     label: 'Oil Temperature',       cls: 'pf-oil-temp' },
  { key: 'winding_temp', label: 'Winding Temperature',   cls: 'pf-winding-temp' },
  { key: 'vibration',    label: 'Vibration',             cls: 'pf-vibration' },
  { key: 'oil_level',    label: 'Oil Level',             cls: 'pf-oil-level' },
];

export default function Contribution({ data }) {
  if (!data) return <section className="page active"><div className="page-head"><h2>Sensor Contribution</h2><p>Waiting…</p></div></section>;

  const sc  = data.sensor_contributions || {};
  const re  = Number(data.reconstruction_error ?? -1);
  const se  = Number(data.smoothed_error       ?? -1);
  const at  = Number(data.adaptive_threshold   ?? data.threshold ?? 0);
  const ft  = Number(data.failure_threshold    ?? 0);
  const ood = data.ood_direction_details || {};

  return (
    <section className="page active">
      <div className="page-head"><h2>Sensor Contribution to Anomaly</h2><p>Which sensor is driving the abnormal reconstruction error</p></div>
      <div className="contrib-wrap">
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div className="card-title">Contribution Breakdown</div>
          {SENSORS.map(({ key, label, cls }) => {
            const v = Number(sc[key] ?? 0);
            return (
              <div className="crow" key={key}>
                <div className="chead"><span>{label}</span><span>{v.toFixed(1)}%</span></div>
                <div className="pwrap"><div className={`pfill ${cls}`} style={{ width: `${Math.max(0, Math.min(100, v))}%` }} /></div>
              </div>
            );
          })}
          <div className="mini" style={{ marginTop: 10 }}>Contribution shows how much each sensor deviates from the learned normal pattern. The dominant contributor drives the prescription.</div>
        </div>
        <div className="contrib-summary">
          <div className="card">
            <div className="card-title">Main Cause</div>
            <div className="big">{(data.main_cause || '--').replace(/_/g, ' ').toUpperCase()}</div>
            <div className="sub">Dominant contributor</div>
          </div>
          <div className="card">
            <div className="card-title">Reconstruction Error</div>
            <div className="big mono">{re < 0 ? 'N/A' : re.toFixed(6)}</div>
            <div className="sub mono">Smoothed: {se < 0 ? 'N/A' : se.toFixed(6)}</div>
          </div>
          <div className="card">
            <div className="card-title">Active Threshold</div>
            <div className="big mono">{at.toFixed(6)}</div>
            <div className="sub mono">Failure: {ft.toFixed(6)}</div>
          </div>
          <div className="card">
            <div className="card-title">Operating Region</div>
            <div className="big" style={{ fontSize: 12 }}>{pretty(data.operating_region)}</div>
            <div className="sub">{data.prescription_context || '--'}</div>
          </div>
          <div className="card">
            <div className="card-title">OOD Score</div>
            <div className="big mono">{data.ood_score != null ? Number(data.ood_score).toFixed(4) : 'N/A'}</div>
            <div className="sub">Per sensor: {Object.entries(ood).map(([k, v]) => `${k.replace(/_/g, ' ')}=${v}`).join(', ') || 'N/A'}</div>
          </div>
          <div className="card">
            <div className="card-title">Confidence</div>
            <div className="big">{pretty(data.confidence_level)} ({Number(data.confidence_score ?? 0).toFixed(1)}%)</div>
            <div className="sub">{data.uncertainty_reason || '--'}</div>
          </div>
        </div>
      </div>
    </section>
  );
}
