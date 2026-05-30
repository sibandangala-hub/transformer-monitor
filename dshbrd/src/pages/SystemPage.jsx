import { fmt, pretty } from '../utils/statusHelpers';

function SysCard({ name, status, badge, detail, meta1, meta2 }) {
  return (
    <div className="sys-card">
      <div className="sys-indicator">
        <div className={`sys-dot ${status}`} />
        <div>
          <div className="sys-name">{name}</div>
          <span className={`sys-badge ${status}`}>{badge || status.toUpperCase()}</span>
        </div>
      </div>
      {detail && <div className="sys-detail">{detail}</div>}
      {meta1  && <div className="sys-meta">{meta1}</div>}
      {meta2  && <div className="sys-meta">{meta2}</div>}
    </div>
  );
}

// How long without a packet before the ESP32 is considered offline
const ESP32_OFFLINE_MS = 10 * 1000; // 1 minute

export default function SystemPage({ data, connected, stale, lastPacketMs, analysisCount }) {
  const ts = data ? fmt(data.analysis_timestamp) : '--';

  const hasData     = data && data.latest_values && Object.keys(data.latest_values).length > 0;
  const packetAge   = lastPacketMs ? Date.now() - lastPacketMs : Infinity;
  const esp32OK     = hasData && connected && !stale && packetAge < ESP32_OFFLINE_MS;
  const esp32Stale  = hasData && connected && stale;   // had data, but gone quiet
  const hasFaults   = data && (data.temp_fault || data.oil_temp_fault || data.voltage_fault);
  const modelReady  = data && data.reconstruction_error != null;
  const modelOK     = data && data.threshold_mode != null;
  const adaptiveRdy = data && data.adaptive_threshold_ready;
  const allOK       = esp32OK && modelReady && connected && modelOK;

  const esp32Status = esp32OK ? 'online' : (esp32Stale ? 'unknown' : 'offline');
  const esp32Detail = esp32OK
    ? (hasFaults ? 'Transmitting — hardware fault flags detected' : 'Transmitting normally — all sensors active')
    : esp32Stale
      ? `No packet received for ${Math.round(packetAge / 1000)}s — device may be offline or rebooting`
      : 'No data from sensor hardware — check power and Wi-Fi';

  return (
    <section className="page active">
      <div className="page-head"><h2>System Status</h2><p>Live connectivity and health of all system components</p></div>
      <div className="sys-wrap">
        <SysCard
          name="ESP32 Sensor Node"
          status={esp32Status}
          detail={esp32Detail}
          meta1={`Last packet: ${ts}`}
          meta2={data ? `Temp fault: ${data.temp_fault?'YES':'NO'} | Oil temp fault: ${data.oil_temp_fault?'YES':'NO'} | Voltage fault: ${data.voltage_fault?'YES':'NO'}` : 'Sensors: --'}
        />
        <SysCard
          name="Render Backend (API)"
          status={modelReady ? 'online' : 'offline'}
          detail={modelReady ? 'Backend responding — inference active' : 'Backend not returning inference results'}
          meta1="Endpoint: transformer-monitor-lvtd.onrender.com"
          meta2={`Last response: ${ts}`}
        />
        <SysCard
          name="Firebase Realtime DB"
          status={connected ? 'online' : 'offline'}
          detail={connected ? 'Connected — receiving live updates' : 'Connection lost — check Firebase config'}
          meta1="Path: /transformer_monitor/latest"
          meta2={`Last write: ${ts}`}
        />
        <SysCard
          name="LSTM Autoencoder Model"
          status={modelOK ? 'online' : 'offline'}
          detail={modelOK ? `Inference active — mode: ${pretty(data?.threshold_mode)}` : 'Model not loaded'}
          meta1={`Threshold: ${data?.adaptive_threshold ? Number(data.adaptive_threshold).toFixed(6) : '--'} | Failure: ${data?.failure_threshold ? Number(data.failure_threshold).toFixed(6) : '--'}`}
        />
        <SysCard
          name="Adaptive Threshold Engine"
          status={adaptiveRdy ? 'online' : (data?.adaptive_threshold_enabled ? 'unknown' : 'offline')}
          detail={adaptiveRdy ? 'Adaptive threshold active and calibrated' : (data?.adaptive_threshold_enabled ? `Building history — ${data?.adaptive_history_count || 0}/20 points` : 'Adaptive threshold disabled')}
          meta1={`History: ${data?.adaptive_history_count || 0} pts | Ready: ${adaptiveRdy ? 'YES' : 'NO'}`}
        />
        <SysCard
          name="Overall System Health"
          status={allOK ? 'online' : 'offline'}
          detail={allOK ? 'All systems operational' : 'One or more components offline or degraded'}
          meta1={`Last full update: ${ts}`}
          meta2={`Analysis count this session: ${analysisCount}`}
        />
      </div>
    </section>
  );
}