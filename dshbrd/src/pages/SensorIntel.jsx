import { stateCls, stateDesc, pretty, fmt } from '../utils/statusHelpers';

const SENSORS = [
  { key: 'current',      label: 'Current State' },
  { key: 'oil_temp',     label: 'Oil Temp State' },
  { key: 'winding_temp', label: 'Winding Temp State' },
  { key: 'vibration',    label: 'Vibration State' },
  { key: 'oil_level',    label: 'Oil Level State' },
];

export default function SensorIntel({ data }) {
  if (!data) return <section className="page active"><div className="page-head"><h2>Sensor Intelligence</h2><p>Waiting…</p></div></section>;

  const lv   = data.latest_values || {};
  const curr = Number(lv.current      ?? 0);
  const oilT = Number(lv.oil_temp     ?? 0);
  const winT = Number(lv.winding_temp ?? 0);
  const vib  = Number(lv.vibration    ?? 0);
  const oil  = Number(lv.oil_level    ?? 0);
  const ts   = fmt(data.analysis_timestamp);

  const stateMap = {
    current: data.current_state, oil_temp: data.oil_temp_state,
    winding_temp: data.winding_temp_state, vibration: data.vibration_state,
    oil_level: data.oil_level_state,
  };

  return (
    <section className="page active">
      <div className="page-head"><h2>Sensor Intelligence</h2><p>Per-sensor state classification against learned normal operating bands and confidence metrics</p></div>
      <div className="states-wrap">
        <div className="states-row">
          {SENSORS.map(({ key, label }) => (
            <div className="card" key={key}>
              <div className="card-title">{label}</div>
              <div className={`big ${stateCls(stateMap[key])}`}>{pretty(stateMap[key])}</div>
              <div className="sub">{stateDesc(label.replace(' State', ''), stateMap[key])}</div>
            </div>
          ))}
        </div>
        <div className="action-row">
          <div className="card">
            <div className="card-title">Confidence</div>
            <div className="big">{pretty(data.confidence_level)} ({Number(data.confidence_score ?? 0).toFixed(1)}%)</div>
            <div className="sub">{data.uncertainty_reason || '--'}</div>
          </div>
          <div className="card">
            <div className="card-title">Auto Action</div>
            <div className="big" style={{ fontSize: 14 }}>{pretty(data.auto_action)}</div>
            <div className="sub">Last update: {ts}</div>
          </div>
          <div className="card">
            <div className="card-title">Latest Values Summary</div>
            <div className="sub">Current: {curr.toFixed(2)} A | Oil T: {oilT.toFixed(1)}°C | Wind T: {winT.toFixed(1)}°C | Vib: {vib.toFixed(3)} | Oil Lvl: {oil.toFixed(1)}%</div>
            <div className="mini">
              Current: {pretty(stateMap.current)} | Oil Temp: {pretty(stateMap.oil_temp)} | Winding Temp: {pretty(stateMap.winding_temp)} | Vibration: {pretty(stateMap.vibration)} | Oil Level: {pretty(stateMap.oil_level)}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
