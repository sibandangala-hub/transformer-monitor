import { urgCls, pretty } from '../utils/statusHelpers';

export default function Prescription({ data }) {
  if (!data) return <section className="page active"><div className="page-head"><h2>Maintenance Prescription</h2><p>Waiting…</p></div></section>;

  const mps    = Number(data.maintenance_priority_score ?? 0);
  const rulMin = Number(data.rul_min ?? 0);
  const rulMax = Number(data.rul_max ?? 0);
  const uCls   = urgCls(data.urgency_level);
  const acts   = Array.isArray(data.prescription_actions) ? data.prescription_actions : [];

  return (
    <section className="page active">
      <div className="page-head"><h2>Maintenance Prescription</h2><p>Recommended actions derived from transformer-specific anomaly analysis</p></div>
      <div className="rx-split" style={{ flex: 1, overflow: 'hidden' }}>
        <div className="card" style={{ overflowY: 'auto' }}>
          <div className="card-title">Prescription</div>
          <div className="rx-title">{data.prescription_title || 'Waiting for prescription…'}</div>
          <div className="pill">{data.prescription_category_label || '--'}</div>
          <ul className="actions">{acts.length ? acts.map((a, i) => <li key={i}>{a}</li>) : <li>--</li>}</ul>
          <div className="reason">{data.prescription_reason || '--'}</div>
        </div>
        <div className="decision">
          <div className="card">
            <div className="card-title">Urgency</div>
            <div className={`big ${uCls}`}>{pretty(data.urgency_level)}</div>
            <div className="sub">Maintenance urgency level</div>
          </div>
          <div className="card">
            <div className="card-title">Auto Action</div>
            <div className="big" style={{ fontSize: 12 }}>{pretty(data.auto_action)}</div>
            <div className="sub">Recommended automatic action</div>
          </div>
          <div className="card">
            <div className="card-title">Main Cause</div>
            <div className="big">{(data.main_cause || '--').replace(/_/g, ' ').toUpperCase()}</div>
            <div className="sub">Dominant: {pretty(data.dominant_state)}</div>
          </div>
          <div className="card">
            <div className="card-title">Priority Score</div>
            <div className="big">{mps.toFixed(1)}</div>
            <div className="sub">Out of 100</div>
          </div>
          <div className="card">
            <div className="card-title">Operating Region</div>
            <div className="big" style={{ fontSize: 12 }}>{pretty(data.operating_region)}</div>
            <div className="sub">{data.prescription_context || '--'}</div>
          </div>
          <div className="card">
            <div className="card-title">RUL Range</div>
            <div className="big">{rulMin.toFixed(0)} – {rulMax.toFixed(0)} h</div>
            <div className="sub">Confidence: {pretty(data.confidence_level)} ({Number(data.confidence_score ?? 0).toFixed(1)}%)</div>
          </div>
        </div>
      </div>
    </section>
  );
}
