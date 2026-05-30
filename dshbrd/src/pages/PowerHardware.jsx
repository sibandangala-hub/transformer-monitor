export default function PowerHardware({ data }) {
  if (!data) return (
    <section className="page active">
      <div className="page-head"><h2>Hardware Diagnostics</h2><p>Waiting…</p></div>
    </section>
  );

  const ledSt  = data.led_status || 'GREEN';
  const faults = [
    { id: 'temp_fault',     label: 'TEMP FAULT' },
    { id: 'oil_temp_fault', label: 'OIL TEMP FAULT' },
    { id: 'voltage_fault',  label: 'VOLTAGE FAULT' },
  ];
  const isOk = v => !v || v === false || v === 0 || v === '0' || v === 'false';

  return (
    <section className="page active">
      <div className="page-head">
        <h2>Hardware Diagnostics</h2>
        <p>ESP32 LED indicator status and hardware fault flags — independent of ML model</p>
      </div>
      <div className="ph-wrap">

        {/* LEDs */}
        <div className="card">
          <div className="card-title">ESP32 LED Indicator Status</div>
          <div className="mini" style={{ marginBottom: 10 }}>
            Physical LED states derived from ML model output. One LED active at a time.
          </div>
          <div className="led-row">
            {[
              { color: 'GREEN',  cls: 'green',  label: 'GREEN',  desc: 'NORMAL' },
              { color: 'YELLOW', cls: 'yellow', label: 'YELLOW', desc: 'WARNING' },
              { color: 'RED',    cls: 'red',    label: 'RED',    desc: 'URGENT/CRITICAL' },
            ].map(({ color, cls, label, desc }) => (
              <div className="led-card" key={color}>
                <div className={`led-dot ${ledSt === color ? cls : 'off'}`} />
                <div className="led-label">{label}</div>
                <div className="led-status">{ledSt === color ? 'ACTIVE' : desc}</div>
              </div>
            ))}
          </div>

          {/* current LED status summary */}
          <div style={{ marginTop: 16, padding: '10px 14px', borderRadius: 10, background: 'var(--bg)', border: '1px solid var(--border)' }}>
            <div className="card-title">Active LED</div>
            <div className={`big ${ledSt === 'GREEN' ? 'normal' : ledSt === 'YELLOW' ? 'warning' : 'urgent'}`}>
              {ledSt}
            </div>
            <div className="sub">
              {ledSt === 'GREEN'  && 'Transformer operating normally'}
              {ledSt === 'YELLOW' && 'Warning — monitor closely'}
              {ledSt === 'RED'    && 'Urgent/Critical — immediate attention required'}
            </div>
          </div>
        </div>

        {/* Fault flags */}
        <div className="card">
          <div className="card-title">Hardware Fault Flags</div>
          <div className="mini" style={{ marginBottom: 14 }}>
            Raw fault signals from ESP32 sensor hardware. Independent of the ML model — direct hardware-level detection.
          </div>
          <div className="fault-row">
            {faults.map(({ id, label }) => {
              const ok = isOk(data[id]);
              return (
                <div key={id} className={`fault-badge ${ok ? 'fault-ok' : 'fault-err'}`}>
                  {label}<br />{ok ? 'OK' : 'FAULT'}
                </div>
              );
            })}
          </div>
          <div className="sub" style={{ marginTop: 14, fontSize: 11 }}>
            These fault flags are not used as ML model inputs. The model infers fault conditions
            from learned patterns in the 5 core sensor signals. Fault flags provide independent
            hardware-level verification.
          </div>

          {/* fault summary */}
          <div style={{ marginTop: 14 }}>
            {['temp_fault', 'oil_temp_fault', 'voltage_fault'].some(f => !isOk(data[f])) ? (
              <div style={{ padding: '8px 12px', borderRadius: 8, background: '#2a0a0a', border: '1px solid #7f1d1d', color: '#f87171', fontSize: 12, fontWeight: 700 }}>
                ⚠ Hardware fault detected — cross-check with ML prescription
              </div>
            ) : (
              <div style={{ padding: '8px 12px', borderRadius: 8, background: '#0a1f10', border: '1px solid #166534', color: '#4ade80', fontSize: 12, fontWeight: 700 }}>
                ✓ All hardware fault flags clear
              </div>
            )}
          </div>
        </div>

      </div>
    </section>
  );
}
