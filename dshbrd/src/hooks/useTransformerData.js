import { useEffect, useState, useRef } from 'react';
import { ref, onValue, push, query, orderByChild, limitToLast } from 'firebase/database';
import { db } from '../firebase';

const LATEST_PATH  = 'transformer_monitor/latest';
const HISTORY_PATH = 'transformer_monitor/history';
const MAX_PTS      = 30;
const STALE_MS     = 3 * 60 * 1000; // 3 minutes = stale

export function useTransformerData() {
  const [data, setData]         = useState(null);
  const [connected, setConnected] = useState(false);
  const [stale, setStale]       = useState(false);
  const [lastPacketMs, setLastPacketMs] = useState(null);
  const [history, setHistory]   = useState({
    labels: [],
    series: { current: [], oil_temp: [], winding_temp: [], vibration: [], oil_level: [] }
  });
  const [prevValues, setPrevValues] = useState(null);

  const historyRef  = useRef(history);
  const prevRef     = useRef(null);
  const staleTimer  = useRef(null);

  historyRef.current = history;

  // ── stale watchdog ──
  const resetStaleTimer = () => {
    clearTimeout(staleTimer.current);
    setStale(false);
    staleTimer.current = setTimeout(() => setStale(true), STALE_MS);
  };

  useEffect(() => {
    const dbRef = ref(db, LATEST_PATH);
    const unsub = onValue(
      dbRef,
      (snap) => {
        const val = snap.val();
        if (!val) { setConnected(false); return; }

        // save previous for trend arrows
        if (prevRef.current) setPrevValues({ ...prevRef.current });
        prevRef.current = val.latest_values || {};

        setData(val);
        setConnected(true);
        setLastPacketMs(Date.now());
        resetStaleTimer();

        // in-memory chart history
        const lv   = val.latest_values || {};
        const prev = historyRef.current;
        const newLabels = [...prev.labels, new Date().toLocaleTimeString()];
        const newSeries = {
          current:      [...prev.series.current,      Number(lv.current      ?? null)],
          oil_temp:     [...prev.series.oil_temp,     Number(lv.oil_temp     ?? null)],
          winding_temp: [...prev.series.winding_temp, Number(lv.winding_temp ?? null)],
          vibration:    [...prev.series.vibration,    Number(lv.vibration    ?? null)],
          oil_level:    [...prev.series.oil_level,    Number(lv.oil_level    ?? null)],
        };
        if (newLabels.length > MAX_PTS) {
          newLabels.shift();
          Object.keys(newSeries).forEach(k => newSeries[k].shift());
        }
        setHistory({ labels: newLabels, series: newSeries });

        // write to RTDB history (only key fields to save memory)
        push(ref(db, HISTORY_PATH), {
          ts:           Math.floor(Date.now() / 1000),
          current:      Number(lv.current      ?? 0),
          oil_temp:     Number(lv.oil_temp     ?? 0),
          winding_temp: Number(lv.winding_temp ?? 0),
          vibration:    Number(lv.vibration    ?? 0),
          oil_level:    Number(lv.oil_level    ?? 0),
          health:       Number(val.health       ?? 0),
          recon_error:  Number(val.reconstruction_error ?? 0),
          threshold:    Number(val.adaptive_threshold   ?? 0),
        });
      },
      (err) => {
        console.error('Firebase error:', err);
        setConnected(false);
      }
    );
    return () => { unsub(); clearTimeout(staleTimer.current); };
  }, []);

  // ── fetch historical data by range ──
  const fetchHistory = async (rangeMs) => {
    const cutoff = Math.floor((Date.now() - rangeMs) / 1000);
    // limitToLast 2000 pts max to avoid huge reads
    const q = query(ref(db, HISTORY_PATH), orderByChild('ts'), limitToLast(2000));
    return new Promise((resolve) => {
      onValue(q, (snap) => {
        const raw = snap.val();
        if (!raw) { resolve([]); return; }
        const pts = Object.values(raw)
          .filter(p => p.ts >= cutoff)
          .sort((a, b) => a.ts - b.ts);
        resolve(pts);
      }, { onlyOnce: true });
    });
  };

  return { data, connected, stale, lastPacketMs, history, prevValues, fetchHistory };
}