from flask import Flask, request, jsonify, send_from_directory
import os
import time
import traceback
from collections import deque

import numpy as np
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

try:
    import firebase_admin
    from firebase_admin import credentials, db as rtdb
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

import onnxruntime as ort

app = Flask(__name__)

@app.route("/dashboard")
def dashboard():
    return send_from_directory("dashboard", "index.html")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ============================================================
# FILE PATHS
# ============================================================
MODEL_PATH     = os.getenv("MODEL_PATH",     "lstm_autoencoder.onnx")
SCALER_PATH    = os.getenv("SCALER_PATH",    "scaler.save")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.npy")

# ============================================================
# INPUT SETTINGS
# ============================================================
WINDOW_SIZE   = 20
NUM_FEATURES  = 5
FEATURE_NAMES = ["current", "oil_temp", "winding_temp", "vibration", "oil_level"]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# ============================================================
# HEALTH / RUL SETTINGS
# ============================================================
ERROR_HISTORY_SIZE   = 30
MIN_RUL_POINTS       = 6
EMA_ALPHA            = 0.35
FAILURE_MULTIPLIER   = 12.0
MAX_RUL_HOURS        = 100.0
SAMPLE_INTERVAL_SECONDS = float(os.getenv("SAMPLE_INTERVAL_SECONDS", "10"))
SAMPLE_INTERVAL_HOURS   = SAMPLE_INTERVAL_SECONDS / 3600.0

INSUFFICIENT_HISTORY_HEALTH_WEIGHT = 0.6
STABLE_HEALTH_WEIGHT               = 0.8
DEGRADING_PROJECTED_WEIGHT         = 0.7
DEGRADING_HEALTH_WEIGHT            = 0.3

# ============================================================
# ADAPTIVE THRESHOLD SETTINGS
# ============================================================
ADAPTIVE_THRESHOLD_ENABLED    = os.getenv("ADAPTIVE_THRESHOLD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ADAPTIVE_HISTORY_SIZE         = int(os.getenv("ADAPTIVE_HISTORY_SIZE",        "100"))
ADAPTIVE_MIN_HEALTHY_POINTS   = int(os.getenv("ADAPTIVE_MIN_HEALTHY_POINTS",  "20"))
ADAPTIVE_BLEND                = float(os.getenv("ADAPTIVE_BLEND",             "0.35"))
ADAPTIVE_STD_MULTIPLIER       = float(os.getenv("ADAPTIVE_STD_MULTIPLIER",    "3.0"))
ADAPTIVE_PERCENTILE           = float(os.getenv("ADAPTIVE_PERCENTILE",        "95.0"))
ADAPTIVE_MIN_RATIO            = float(os.getenv("ADAPTIVE_MIN_RATIO",         "0.80"))
ADAPTIVE_MAX_RATIO            = float(os.getenv("ADAPTIVE_MAX_RATIO",         "1.50"))
ADAPTIVE_UPDATE_MAX_ERROR_RATIO = float(os.getenv("ADAPTIVE_UPDATE_MAX_ERROR_RATIO", "0.90"))
ADAPTIVE_UPDATE_MIN_HEALTH    = float(os.getenv("ADAPTIVE_UPDATE_MIN_HEALTH", "75.0"))
ADAPTIVE_MAX_OOD_FOR_UPDATE   = float(os.getenv("ADAPTIVE_MAX_OOD_FOR_UPDATE","0.12"))

# ============================================================
# PRESCRIPTIVE LAYER SETTINGS
# ============================================================
DOMINANT_CONTRIBUTION_THRESHOLD = float(os.getenv("DOMINANT_CONTRIBUTION_THRESHOLD", "40.0"))
PERSISTENCE_LOOKBACK            = int(os.getenv("PERSISTENCE_LOOKBACK",              "5"))
OPERATING_BAND_MARGIN_RATIO     = float(os.getenv("OPERATING_BAND_MARGIN_RATIO",     "0.05"))
WARMUP_TEMP_MARGIN_RATIO        = float(os.getenv("WARMUP_TEMP_MARGIN_RATIO",        "0.03"))
MIN_LOAD_CURRENT                = float(os.getenv("MIN_LOAD_CURRENT",                "0.05"))
WARMUP_HEALTH_FLOOR             = float(os.getenv("WARMUP_HEALTH_FLOOR",             "75.0"))
WARMUP_RUL_HOURS                = float(os.getenv("WARMUP_RUL_HOURS",                "80.0"))

MPS_WEIGHTS = {
    "anomaly_severity":  0.28,
    "health_degradation":0.22,
    "rul_risk":          0.22,
    "trend":             0.15,
    "persistence":       0.13,
}

URGENCY_ORDER = ["NORMAL", "WARNING", "PLAN_MAINTENANCE", "URGENT", "CRITICAL"]

# ============================================================
# TRANSFORMER-SPECIFIC PRESCRIPTION RULES
# ============================================================
PRESCRIPTION_RULES = {
    ("current", "LOW"): {
        "type": "electrical_underload",
        "label": "Underload / Open-Circuit Check",
        "context": "Low load current state",
        "actions": [
            "Verify that the transformer is supplying the expected load — check downstream breakers and contactors",
            "Inspect current transformer (CT) wiring and connections for open circuit",
            "Confirm the load has not been disconnected or shifted to an alternate supply",
            "Check current sensor calibration and signal conditioning circuit",
        ],
        "urgent_extra": "Investigate immediately if transformer should be on full load — an open secondary winding is a protection concern",
        "critical_extra": "Isolate and perform winding continuity test if no-load condition persists unexpectedly",
        "normal_title":  "Normal load current — continue monitoring",
        "warning_title": "Warning — low load current observed",
        "plan_title":    "Plan inspection for persistent low current state",
        "urgent_title":  "Urgent — investigate low current / open-circuit condition",
        "critical_title":"Critical — winding continuity test required",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_inspection",
        "auto_urgent":  "inspect_winding_and_supply",
        "auto_critical":"isolate_and_test",
    },
    ("current", "HIGH"): {
        "type": "electrical_overload",
        "label": "Overload / Overcurrent Maintenance",
        "context": "High load current / overload state",
        "actions": [
            "Check total connected load against transformer nameplate KVA rating",
            "Inspect for downstream fault or short-circuit condition",
            "Review load growth trend and compare against rated full-load current",
            "Check protection relay settings — confirm overcurrent relay is armed",
            "Inspect busbar and terminal connections for signs of heating or arcing",
        ],
        "urgent_extra": "Shed non-critical loads immediately to reduce thermal stress on windings",
        "critical_extra": "Trip transformer and perform detailed winding insulation test — prolonged overload causes irreversible insulation damage",
        "normal_title":  "Normal load current — continue monitoring",
        "warning_title": "Warning — elevated load current",
        "plan_title":    "Plan load audit and capacity review",
        "urgent_title":  "Urgent — overload condition, reduce load immediately",
        "critical_title":"Critical — trip and inspect for winding damage",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_load_audit",
        "auto_urgent":  "shed_load",
        "auto_critical":"trip_transformer",
    },
    ("oil_temp", "LOW"): {
        "type": "oil_cold",
        "label": "Cold Oil / Pre-Heat Management",
        "context": "Sub-normal oil temperature state",
        "actions": [
            "Confirm transformer is operating under load — cold oil at full load is unusual and requires investigation",
            "Check oil temperature indicator (OTI) calibration and wiring",
            "Inspect oil circulation pump if forced-oil cooled — pump failure causes local hotspots and false cold bulk readings",
            "Verify ambient temperature conditions and check for unusually cold environment affecting sensor",
        ],
        "urgent_extra": "Investigate oil circulation system immediately if transformer is on load — bulk cold reading with high winding temp is a dangerous mismatch",
        "critical_extra": "Take transformer offline and perform oil sample analysis if temperature mismatch with winding sensor persists",
        "normal_title":  "Normal oil temperature — continue monitoring",
        "warning_title": "Warning — oil temperature below normal band",
        "plan_title":    "Plan OTI calibration check",
        "urgent_title":  "Urgent — inspect cooling and circulation system",
        "critical_title":"Critical — isolate and perform oil and cooling analysis",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_oti_check",
        "auto_urgent":  "inspect_cooling_system",
        "auto_critical":"isolate_and_sample_oil",
    },
    ("oil_temp", "HIGH"): {
        "type": "oil_overheating",
        "label": "Oil Thermal Maintenance",
        "context": "Oil overheating state — cooling system fault likely",
        "actions": [
            "Inspect radiator fins for blockage, dirt, or bent fins reducing airflow",
            "Check cooling fans — confirm all fans are running at rated speed",
            "Verify oil level is within normal range — low oil reduces heat dissipation capacity",
            "Inspect oil circulation pump (if OFAF/ONAF cooled) for flow and pressure",
            "Check load current against nameplate rating — overload causes disproportionate oil heating",
            "Review dissolved gas analysis (DGA) sample if available — oil overheating produces CO and CO2",
        ],
        "urgent_extra": "Reduce transformer load immediately and force-activate backup cooling if available",
        "critical_extra": "Trip transformer — sustained oil overheating above 95°C accelerates insulation pyrolysis and can lead to thermal runaway",
        "normal_title":  "Normal oil temperature — continue monitoring",
        "warning_title": "Warning — elevated oil temperature",
        "plan_title":    "Plan cooling system inspection and oil sample",
        "urgent_title":  "Urgent — overheating, reduce load and inspect cooling",
        "critical_title":"Critical — trip transformer, risk of thermal runaway",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_cooling_inspection",
        "auto_urgent":  "reduce_load_activate_cooling",
        "auto_critical":"trip_transformer",
    },
    ("winding_temp", "LOW"): {
        "type": "winding_cold",
        "label": "Cold Winding / WTI Check",
        "context": "Sub-normal winding temperature state",
        "actions": [
            "Check winding temperature indicator (WTI) sensor and its thermal image current source",
            "Verify transformer is under adequate load — no-load operation may produce genuinely low winding temp",
            "Compare winding temp reading against oil temp — if oil is normal but winding is anomalously low, the WTI is suspect",
            "Inspect RTD/thermocouple wiring for open circuit or poor contact",
        ],
        "urgent_extra": "Perform WTI calibration check — a failed-low winding temp sensor masks real overheating events",
        "critical_extra": "Replace or bypass faulty WTI sensor; do not rely on it for thermal protection until confirmed accurate",
        "normal_title":  "Normal winding temperature — continue monitoring",
        "warning_title": "Warning — winding temperature below normal band",
        "plan_title":    "Plan WTI sensor calibration",
        "urgent_title":  "Urgent — WTI sensor suspected faulty, calibrate immediately",
        "critical_title":"Critical — replace WTI sensor, thermal protection compromised",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_wti_calibration",
        "auto_urgent":  "inspect_and_calibrate_wti",
        "auto_critical":"replace_wti_sensor",
    },
    ("winding_temp", "HIGH"): {
        "type": "winding_overheating",
        "label": "Winding Thermal / Insulation Risk",
        "context": "Winding overheating — insulation degradation risk",
        "actions": [
            "Confirm load current is within rated limits — winding overheating under normal load suggests cooling failure",
            "Check top oil temperature — if oil is cool but winding is hot, suspect localised hotspot or blocked duct",
            "Inspect cooling system fans, radiators, and oil circulation pump",
            "Collect an oil sample for dissolved gas analysis — winding overheating produces ethylene and acetylene",
            "Review winding temperature trip setpoint against actual reading — confirm protection is armed at correct threshold",
        ],
        "urgent_extra": "Reduce load immediately and activate supplementary cooling — winding insulation degrades exponentially above rated temperature (Montsinger rule: each 6°C rise halves insulation life)",
        "critical_extra": "Trip transformer immediately — hotspot above thermal limit causes irreversible cellulose degradation and risk of inter-turn fault",
        "normal_title":  "Normal winding temperature — continue monitoring",
        "warning_title": "Warning — winding temperature rising, monitor closely",
        "plan_title":    "Plan winding thermal inspection and DGA sample",
        "urgent_title":  "Urgent — winding overheating, reduce load now",
        "critical_title":"Critical — trip immediately, winding insulation at risk",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_dga_and_inspection",
        "auto_urgent":  "reduce_load_activate_cooling",
        "auto_critical":"trip_transformer",
    },
    ("vibration", "LOW"): {
        "type": "vibration_low",
        "label": "Low Vibration / Sensor Check",
        "context": "Sub-normal vibration — lightly loaded or sensor fault",
        "actions": [
            "Confirm transformer is energised and carrying load — de-energised transformers produce minimal vibration",
            "Check vibration sensor mounting — loose mounting reduces coupling and gives falsely low readings",
            "Inspect vibration sensor wiring and signal conditioning module",
            "Compare against baseline vibration signature recorded at commissioning",
        ],
        "urgent_extra": "Inspect sensor integrity immediately if transformer should be on full load — a dead sensor masks real mechanical events",
        "critical_extra": "Replace vibration sensor if fault is confirmed — mechanical protection is compromised without valid vibration data",
        "normal_title":  "Normal vibration — continue monitoring",
        "warning_title": "Warning — vibration below normal band, check sensor",
        "plan_title":    "Plan vibration sensor and mounting inspection",
        "urgent_title":  "Urgent — sensor integrity check required",
        "critical_title":"Critical — replace vibration sensor, protection compromised",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_sensor_inspection",
        "auto_urgent":  "inspect_sensor_chain",
        "auto_critical":"replace_sensor",
    },
    ("vibration", "HIGH"): {
        "type": "mechanical_fault",
        "label": "Mechanical / Core Fault Maintenance",
        "context": "Elevated vibration — core or tank mechanical fault",
        "actions": [
            "Inspect tank wall, core clamps, and core tie bolts for looseness — core looseness is the primary cause of abnormal transformer vibration",
            "Check for loose accessories: bushings, conservator fittings, radiator connections, Buchholz relay mounting",
            "Inspect foundation bolts and anti-vibration pads for deterioration",
            "Record vibration frequency spectrum if equipment available — 100/120 Hz dominance indicates core magnetostriction; subharmonics indicate core looseness",
            "Check load current for DC bias or harmonic distortion which can excite abnormal vibration modes",
        ],
        "urgent_extra": "Reduce load to limit magnetostrictive forces — do not allow transformer to run at high load with confirmed mechanical looseness",
        "critical_extra": "De-energise and perform internal inspection — progressive core looseness can cause winding displacement and inter-turn short circuit",
        "normal_title":  "Normal vibration — continue monitoring",
        "warning_title": "Warning — elevated vibration, schedule inspection",
        "plan_title":    "Plan mechanical inspection of core and tank fittings",
        "urgent_title":  "Urgent — mechanical fault likely, reduce load and inspect",
        "critical_title":"Critical — de-energise and perform internal inspection",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_mechanical_inspection",
        "auto_urgent":  "reduce_load_and_inspect",
        "auto_critical":"deenergise_and_inspect",
    },
    ("oil_level", "LOW"): {
        "type": "oil_level_low",
        "label": "Oil Level / Leak Investigation",
        "context": "Low oil level — insulation and cooling at risk",
        "actions": [
            "Inspect transformer tank, gaskets, drain valve, and radiator connections for oil leaks",
            "Check conservator tank oil level gauge and breather condition",
            "Inspect Buchholz relay — gas accumulation may indicate internal fault alongside oil loss",
            "Do not re-energise at reduced oil level — uncovered windings lose dielectric protection",
            "Arrange topping up with compatible dielectric oil of verified quality (IEC 60296)",
        ],
        "urgent_extra": "Sample oil and test for breakdown voltage and moisture before topping up — contaminated or wet oil must not be introduced to the tank",
        "critical_extra": "Isolate transformer immediately — exposed windings or core lose dielectric protection and risk flashover or insulation failure",
        "normal_title":  "Normal oil level — continue monitoring",
        "warning_title": "Warning — oil level below normal, check for leak",
        "plan_title":    "Plan leak inspection and oil top-up",
        "urgent_title":  "Urgent — significant oil loss, inspect and prepare top-up",
        "critical_title":"Critical — isolate immediately, oil level critically low",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_leak_inspection",
        "auto_urgent":  "inspect_and_prepare_oil_topup",
        "auto_critical":"isolate_transformer",
    },
    ("oil_level", "HIGH"): {
        "type": "oil_level_high",
        "label": "Oil Overfill / Thermal Expansion Check",
        "context": "High oil level — possible overfill or blocked breather",
        "actions": [
            "Check if transformer was recently topped up — overfilling causes pressure buildup in the conservator",
            "Inspect conservator breather (silica gel) — a blocked breather causes pressure differential and oil expansion anomalies",
            "Confirm oil temperature is within normal range — high oil temp causes genuine thermal expansion of dielectric oil",
            "Inspect pressure relief device (PRD) to confirm it has not operated",
        ],
        "urgent_extra": "Release excess pressure through drain valve if conservator is confirmed overfilled — do not allow excess pressure to build",
        "critical_extra": "Operate pressure relief manually only if PRD has not operated and pressure is confirmed dangerous — then de-energise and investigate",
        "normal_title":  "Normal oil level — continue monitoring",
        "warning_title": "Warning — oil level above normal band",
        "plan_title":    "Plan conservator and breather inspection",
        "urgent_title":  "Urgent — inspect conservator and release excess oil if confirmed overfill",
        "critical_title":"Critical — de-energise and inspect conservator system",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_conservator_inspection",
        "auto_urgent":  "inspect_conservator_pressure",
        "auto_critical":"deenergise_and_inspect",
    },
    ("mixed", "MIXED"): {
        "type": "general_inspection",
        "label": "General Transformer Inspection",
        "context": "Mixed or unclear dominant condition",
        "actions": [
            "Perform full visual inspection: oil level, temperature indicators, cooling fans, bushing condition",
            "Check all sensor readings against expected values for current load level",
            "Review Buchholz relay and pressure relief device for any prior operation",
            "Collect oil sample for dissolved gas analysis (DGA) — DGA is the most powerful diagnostic for internal transformer faults",
        ],
        "urgent_extra": "Reduce load while general inspection is in progress",
        "critical_extra": "De-energise if condition continues deteriorating after general inspection",
        "normal_title":  "Normal operation — continue monitoring",
        "warning_title": "Warning — general inspection recommended",
        "plan_title":    "Plan comprehensive transformer inspection",
        "urgent_title":  "Urgent — general inspection required, reduce load",
        "critical_title":"Critical — de-energise and perform full inspection",
        "auto_normal":  "monitor",
        "auto_warning": "monitor",
        "auto_plan":    "schedule_inspection",
        "auto_urgent":  "reduce_load_and_inspect",
        "auto_critical":"deenergise_and_inspect",
    },
    ("observe", "NORMAL"): {
        "type": "observe",
        "label": "Observe / No Maintenance Required",
        "context": "Normal operating state",
        "actions": [
            "Continue normal transformer operation",
            "Maintain scheduled monitoring of oil temp, winding temp, oil level, current, and vibration",
        ],
        "urgent_extra": "",
        "critical_extra": "",
        "normal_title":  "Normal operation — continue monitoring",
        "warning_title": "Warning — continue close monitoring",
        "plan_title":    "Plan follow-up monitoring",
        "urgent_title":  "Urgent — review monitoring trends",
        "critical_title":"Critical — review required",
        "auto_normal":  "none",
        "auto_warning": "monitor",
        "auto_plan":    "monitor",
        "auto_urgent":  "inspect",
        "auto_critical":"inspect",
    },
}

# ============================================================
# GLOBALS
# ============================================================
model         = None
scaler        = None
threshold     = None
startup_error = None
is_loaded     = False

raw_error_history              = deque(maxlen=ERROR_HISTORY_SIZE)
smooth_error_history           = deque(maxlen=ERROR_HISTORY_SIZE)
adaptive_healthy_error_history = deque(maxlen=ADAPTIVE_HISTORY_SIZE)
adaptive_threshold_history     = deque(maxlen=ERROR_HISTORY_SIZE)
last_adaptive_threshold        = None

firebase_ref = None

# ============================================================
# FIREBASE INIT
# ============================================================
def init_firebase():
    global firebase_ref
    if not FIREBASE_AVAILABLE:
        print("firebase_admin not installed — Firebase write disabled.")
        return

    db_url   = os.getenv("FIREBASE_DB_URL", "")
    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON", "")

    if not db_url:
        print("FIREBASE_DB_URL not set — Firebase write disabled.")
        return

    try:
        import json
        if cred_json:
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})
        else:
            firebase_admin.initialize_app(options={"databaseURL": db_url})

        firebase_ref = rtdb.reference("/transformer_monitor/latest")
        print("Firebase RTDB initialised. Writing to /transformer_monitor/latest")
    except Exception as e:
        print(f"Firebase init failed: {e}")
        firebase_ref = None

def write_to_firebase(payload: dict):
    if firebase_ref is None:
        return
    try:
        firebase_ref.set(payload)
    except Exception as e:
        print(f"Firebase write error: {e}")

def save_history_to_firebase():
    if firebase_ref is None:
        return
    try:
        history_ref = rtdb.reference("/transformer_monitor/scorer_state")
        history_ref.set({
            "raw_error_history":              list(raw_error_history),
            "smooth_error_history":           list(smooth_error_history),
            "adaptive_healthy_error_history": list(adaptive_healthy_error_history),
            "adaptive_threshold_history":     list(adaptive_threshold_history),
            "last_adaptive_threshold":        last_adaptive_threshold,
            "saved_at":                       int(time.time()),
        })
    except Exception as e:
        print(f"Firebase history save error: {e}")

def load_history_from_firebase():
    global raw_error_history, smooth_error_history
    global adaptive_healthy_error_history, adaptive_threshold_history
    global last_adaptive_threshold
    if firebase_ref is None:
        return
    try:
        history_ref = rtdb.reference("/transformer_monitor/scorer_state")
        state = history_ref.get()
        if not state:
            print("No saved scorer state found in Firebase — starting fresh.")
            return
        def restore(key, dest):
            vals = state.get(key, [])
            if isinstance(vals, list):
                for v in vals:
                    dest.append(float(v))
            elif isinstance(vals, dict):
                for k in sorted(vals.keys(), key=lambda x: int(x)):
                    dest.append(float(vals[k]))
        restore("raw_error_history",              raw_error_history)
        restore("smooth_error_history",           smooth_error_history)
        restore("adaptive_healthy_error_history", adaptive_healthy_error_history)
        restore("adaptive_threshold_history",     adaptive_threshold_history)
        if state.get("last_adaptive_threshold") is not None:
            last_adaptive_threshold = float(state["last_adaptive_threshold"])
        print(f"Scorer state restored — smooth_history={len(smooth_error_history)} pts, "
              f"adaptive_history={len(adaptive_healthy_error_history)} pts, "
              f"last_threshold={last_adaptive_threshold}")
    except Exception as e:
        print(f"Firebase history load error: {e}")

# ============================================================
# HELPERS
# ============================================================
def clamp(value, low, high):
    return max(low, min(high, value))

def validate_input(readings):
    if not isinstance(readings, list):
        return False, "Field 'readings' must be a list."
    if len(readings) != WINDOW_SIZE:
        return False, f"'readings' must contain exactly {WINDOW_SIZE} rows."
    for i, row in enumerate(readings):
        if not isinstance(row, list):
            return False, f"Row {i} must be a list."
        if len(row) != NUM_FEATURES:
            return False, f"Row {i} must contain exactly {NUM_FEATURES} values (current, oil_temp, winding_temp, vibration, oil_level)."
        for j, value in enumerate(row):
            try:
                float(value)
            except Exception:
                return False, f"Value at row {i}, column {j} is not numeric."
    return True, "OK"

def backend_ready():
    return model is not None and scaler is not None and threshold is not None

def compute_total_error(x_true, x_pred):
    return float(np.mean(np.square(x_true - x_pred)))

def compute_feature_errors(x_true, x_pred):
    return np.mean(np.square(x_true[0] - x_pred[0]), axis=0)

def compute_sensor_contributions(feature_errors):
    total = float(np.sum(feature_errors))
    if total <= 1e-12:
        return {name: 0.0 for name in FEATURE_NAMES}, "unknown"
    perc = (feature_errors / total) * 100.0
    contributions = {name: float(perc[i]) for i, name in enumerate(FEATURE_NAMES)}
    main_cause = FEATURE_NAMES[int(np.argmax(perc))]
    return contributions, main_cause

def update_smoothed_error(new_error):
    raw_error_history.append(float(new_error))
    if len(smooth_error_history) == 0:
        smooth = float(new_error)
    else:
        smooth = EMA_ALPHA * float(new_error) + (1.0 - EMA_ALPHA) * smooth_error_history[-1]
    smooth_error_history.append(float(smooth))
    return float(smooth)

def compute_adaptive_threshold(base_threshold, healthy_errors):
    if not ADAPTIVE_THRESHOLD_ENABLED:
        return float(base_threshold), False
    count = len(healthy_errors)
    if count < ADAPTIVE_MIN_HEALTHY_POINTS:
        return float(base_threshold), False
    values = np.array(healthy_errors, dtype=np.float64)
    local_mean       = float(np.mean(values))
    local_std        = float(np.std(values))
    local_percentile = float(np.percentile(values, ADAPTIVE_PERCENTILE))
    candidate = max(local_mean + ADAPTIVE_STD_MULTIPLIER * local_std, local_percentile)
    blended = (1.0 - ADAPTIVE_BLEND) * float(base_threshold) + ADAPTIVE_BLEND * candidate
    lower_bound = float(base_threshold) * ADAPTIVE_MIN_RATIO
    upper_bound = float(base_threshold) * ADAPTIVE_MAX_RATIO
    adaptive_threshold = clamp(blended, lower_bound, upper_bound)
    return float(adaptive_threshold), True

def get_adaptive_history_summary(healthy_errors, base_threshold):
    if len(healthy_errors) == 0:
        return {"count": 0, "ready": False, "mean": None, "std": None, "percentile": None, "base_threshold": float(base_threshold)}
    values = np.array(healthy_errors, dtype=np.float64)
    return {
        "count":          int(len(values)),
        "ready":          bool(len(values) >= ADAPTIVE_MIN_HEALTHY_POINTS),
        "mean":           float(np.mean(values)),
        "std":            float(np.std(values)),
        "percentile":     float(np.percentile(values, ADAPTIVE_PERCENTILE)),
        "base_threshold": float(base_threshold),
    }

def compute_health(smoothed_error, anomaly_threshold, warmup_like=False):
    if warmup_like:
        return WARMUP_HEALTH_FLOOR
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER
    if smoothed_error <= anomaly_threshold:
        return 100.0
    if smoothed_error >= failure_threshold:
        return 0.0
    health = 100.0 * (1.0 - (smoothed_error - anomaly_threshold) / (failure_threshold - anomaly_threshold))
    return float(clamp(health, 0.0, 100.0))

def estimate_trend():
    if len(smooth_error_history) < MIN_RUL_POINTS:
        return None
    y = np.array(smooth_error_history, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64) * SAMPLE_INTERVAL_HOURS
    try:
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return None

def estimate_rul(smoothed_error, anomaly_threshold, warmup_like=False):
    if warmup_like:
        return WARMUP_RUL_HOURS, "warmup_idle"
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER
    health = compute_health(smoothed_error, anomaly_threshold)
    slope  = estimate_trend()
    if smoothed_error >= failure_threshold or health <= 0:
        return 0.0, "failed"
    distance_to_failure = failure_threshold - smoothed_error
    health_reserve = (health / 100.0) * MAX_RUL_HOURS
    if slope is None:
        rul = INSUFFICIENT_HISTORY_HEALTH_WEIGHT * health_reserve
        return float(clamp(rul, 0.0, MAX_RUL_HOURS)), "insufficient_history"
    if slope <= 0:
        rul = STABLE_HEALTH_WEIGHT * health_reserve
        return float(clamp(rul, 0.0, MAX_RUL_HOURS)), "stable"
    projected_hours = distance_to_failure / slope
    projected_hours = clamp(projected_hours, 0.0, MAX_RUL_HOURS)
    rul = DEGRADING_PROJECTED_WEIGHT * projected_hours + DEGRADING_HEALTH_WEIGHT * health_reserve
    return float(clamp(rul, 0.0, MAX_RUL_HOURS)), "degrading"

def compute_ood_score(raw_window, scaler_obj):
    if scaler_obj is None:
        return None, {}, {}, None
    mins = getattr(scaler_obj, "data_min_", None)
    maxs = getattr(scaler_obj, "data_max_", None)
    if mins is None or maxs is None:
        return None, {}, {}, None
    latest = np.array(raw_window[-1], dtype=np.float64)
    mins   = np.array(mins, dtype=np.float64)
    maxs   = np.array(maxs, dtype=np.float64)
    span   = np.maximum(maxs - mins, 1e-6)
    below     = np.maximum((mins - latest) / span, 0.0)
    above     = np.maximum((latest - maxs) / span, 0.0)
    violation = below + above
    details           = {name: float(violation[i]) for i, name in enumerate(FEATURE_NAMES)}
    direction_details = {}
    for i, name in enumerate(FEATURE_NAMES):
        if below[i] > 0:
            direction_details[name] = "LOW"
        elif above[i] > 0:
            direction_details[name] = "HIGH"
        else:
            direction_details[name] = "NORMAL"
    main_ood_feature = FEATURE_NAMES[int(np.argmax(violation))] if np.max(violation) > 0 else None
    ood_score = float(np.mean(violation))
    return ood_score, details, direction_details, main_ood_feature

def compute_trend_instability():
    if len(smooth_error_history) < MIN_RUL_POINTS:
        return 1.0
    y     = np.array(smooth_error_history, dtype=np.float64)
    diffs = np.diff(y)
    if len(diffs) == 0:
        return 1.0
    diff_std  = float(np.std(diffs))
    diff_mean = float(np.mean(np.abs(diffs))) + 1e-6
    return float(clamp(diff_std / diff_mean, 0.0, 1.0))

def compute_error_fluctuation(anomaly_threshold):
    if len(raw_error_history) < 2:
        return 1.0
    err_std = float(np.std(np.array(raw_error_history, dtype=np.float64)))
    scale   = max(0.5 * anomaly_threshold, 1e-6)
    return float(clamp(err_std / scale, 0.0, 1.0))

def compute_confidence(ood_score, anomaly_threshold):
    error_fluctuation    = compute_error_fluctuation(anomaly_threshold)
    trend_instability    = compute_trend_instability()
    out_of_distribution  = float(clamp((ood_score or 0.0) / 0.30, 0.0, 1.0))
    limited_history      = float(1.0 - min(len(smooth_error_history) / float(MIN_RUL_POINTS), 1.0))
    total_penalty        = (0.35 * error_fluctuation + 0.30 * trend_instability + 0.20 * out_of_distribution + 0.15 * limited_history)
    confidence_score     = float(clamp(100.0 * (1.0 - total_penalty), 0.0, 100.0))
    if confidence_score >= 75:
        confidence_level = "high"
    elif confidence_score >= 45:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    sources = {
        "error_fluctuation":   round(error_fluctuation,   4),
        "trend_instability":   round(trend_instability,   4),
        "out_of_distribution": round(out_of_distribution, 4),
        "limited_history":     round(limited_history,     4),
    }
    return confidence_score, confidence_level, sources

def compute_rul_range(rul, confidence_score, health, rul_state):
    if rul <= 0:
        return 0.0, 0.0, 0.0
    uncertainty_factor = 1.0 - (confidence_score / 100.0)
    if rul_state == "insufficient_history":
        spread_ratio = 0.45 + 0.30 * uncertainty_factor
    elif rul_state == "stable":
        spread_ratio = 0.20 + 0.25 * uncertainty_factor
    else:
        spread_ratio = 0.25 + 0.35 * uncertainty_factor
    if health < 40:
        spread_ratio += 0.10
    spread_ratio = float(clamp(spread_ratio, 0.10, 0.85))
    rul_min = max(0.0, rul * (1.0 - spread_ratio))
    rul_max = min(MAX_RUL_HOURS, rul * (1.0 + spread_ratio))
    rul_std = (rul_max - rul_min) / 4.0
    return float(rul_min), float(rul_max), float(rul_std)

def build_uncertainty_reason(ood_score, ood_feature, confidence_sources):
    ranked   = sorted(confidence_sources.items(), key=lambda x: x[1], reverse=True)
    top_name, top_value = ranked[0]
    if ood_score is not None and ood_score > 0.10 and ood_feature:
        return f"{ood_feature.replace('_', ' ').capitalize()} is outside the training range, reducing prediction confidence."
    if top_name == "limited_history" and top_value > 0.5:
        return "Not enough recent history is available yet for a stable RUL estimate."
    if top_name == "trend_instability" and top_value > 0.4:
        return "Recent degradation trend is unstable — RUL estimate carries higher uncertainty."
    if top_name == "error_fluctuation" and top_value > 0.4:
        return "Reconstruction error is fluctuating — RUL confidence interval is wider than usual."
    return "Prediction is based on recent transformer behaviour and is relatively stable."

def derive_status_and_led(is_anomaly, health_value, urgency_level=None):
    if urgency_level in {"CRITICAL", "URGENT"}:
        return "Anomaly", "RED"
    if urgency_level in {"WARNING", "PLAN_MAINTENANCE"}:
        return "Warning", "YELLOW"
    if is_anomaly or health_value <= 20:
        return "Anomaly", "RED"
    if health_value <= 60:
        return "Warning", "YELLOW"
    return "Normal", "GREEN"

# ============================================================
# OPERATING REGION HELPERS
# ============================================================
def env_float(names, default_value):
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            try:
                return float(value)
            except Exception:
                pass
    return float(default_value)

def get_operating_bands(scaler_obj):
    mins = getattr(scaler_obj, "data_min_", None)
    maxs = getattr(scaler_obj, "data_max_", None)
    if mins is None or maxs is None:
        return {name: {"low": 0.0, "high": 9999.0} for name in FEATURE_NAMES}
    bands = {}
    for name in FEATURE_NAMES:
        idx      = FEATURE_INDEX[name]
        data_min = float(mins[idx])
        data_max = float(maxs[idx])
        span     = max(data_max - data_min, 1e-6)
        default_low  = data_min - OPERATING_BAND_MARGIN_RATIO * span
        default_high = data_max + OPERATING_BAND_MARGIN_RATIO * span
        env_name = name.upper().replace("_", "")
        low  = env_float([f"{env_name}_NORMAL_LOW",  f"{name.upper()}_NORMAL_LOW"],  default_low)
        high = env_float([f"{env_name}_NORMAL_HIGH", f"{name.upper()}_NORMAL_HIGH"], default_high)
        if low > high:
            low, high = high, low
        bands[name] = {"low": float(low), "high": float(high)}
    return bands

def classify_state(value, low, high):
    if value < low:  return "LOW"
    if value > high: return "HIGH"
    return "NORMAL"

def compute_sensor_states(latest_values, bands):
    states    = {}
    distances = {}
    for name in FEATURE_NAMES:
        value = float(latest_values[name])
        low   = float(bands[name]["low"])
        high  = float(bands[name]["high"])
        states[name] = classify_state(value, low, high)
        if value < low:
            distances[name] = float(low - value)
        elif value > high:
            distances[name] = float(value - high)
        else:
            distances[name] = 0.0
    return states, distances

def compute_condition_warmup_flag(raw_window, bands):
    oil_temp_idx  = FEATURE_INDEX["oil_temp"]
    temp_low      = float(bands["oil_temp"]["low"])
    temp_high     = float(bands["oil_temp"]["high"])
    span          = max(temp_high - temp_low, 1e-6)
    warmup_exit   = temp_low + WARMUP_TEMP_MARGIN_RATIO * span
    latest_temp   = float(raw_window[-1][oil_temp_idx])
    recent_mean   = float(np.mean(raw_window[-5:, oil_temp_idx]))
    warmup_active = latest_temp < warmup_exit and recent_mean < warmup_exit
    return bool(warmup_active), float(warmup_exit)

def should_update_adaptive_history(raw_window, reconstruction_error, active_threshold, health, ood_score, scaler_obj, exempt_from_warmup=False):
    if not ADAPTIVE_THRESHOLD_ENABLED:
        return False, "adaptive_disabled", False, None
    operating_bands = get_operating_bands(scaler_obj)
    warmup_like, warmup_exit_temp = compute_condition_warmup_flag(raw_window, operating_bands)
    # Block adaptive update during warmup unless overridden by exempt sensors
    effective_warmup_for_adaptive = warmup_like and not exempt_from_warmup
    if effective_warmup_for_adaptive:
        return False, "warmup_like_condition", True, float(warmup_exit_temp)
    # If current or vibration is HIGH, it is a fault condition — never train adaptive baseline on fault data
    if exempt_from_warmup:
        return False, "exempt_sensor_fault_active", False, float(warmup_exit_temp)
    if health < ADAPTIVE_UPDATE_MIN_HEALTH:
        return False, "health_below_update_limit", False, float(warmup_exit_temp)
    if reconstruction_error > active_threshold * ADAPTIVE_UPDATE_MAX_ERROR_RATIO:
        return False, "error_too_close_to_threshold", False, float(warmup_exit_temp)
    if ood_score is not None and ood_score > ADAPTIVE_MAX_OOD_FOR_UPDATE:
        return False, "out_of_distribution", False, float(warmup_exit_temp)
    return True, "accepted", False, float(warmup_exit_temp)

def maybe_update_adaptive_history(raw_window, reconstruction_error, active_threshold, health, ood_score, scaler_obj, exempt_from_warmup=False):
    should_update, reason, warmup_like, warmup_exit_temp = should_update_adaptive_history(
        raw_window=raw_window, reconstruction_error=reconstruction_error,
        active_threshold=active_threshold, health=health,
        ood_score=ood_score, scaler_obj=scaler_obj,
        exempt_from_warmup=exempt_from_warmup,
    )
    if should_update:
        adaptive_healthy_error_history.append(float(reconstruction_error))
    return {
        "applied":                 bool(should_update),
        "reason":                  reason,
        "warmup_like":             bool(warmup_like),
        "warmup_exit_temperature": warmup_exit_temp,
        "history_count":           len(adaptive_healthy_error_history),
        "history_ready":           len(adaptive_healthy_error_history) >= ADAPTIVE_MIN_HEALTHY_POINTS,
    }

def compute_anomaly_severity(smoothed_error, anomaly_threshold):
    failure_threshold = anomaly_threshold * FAILURE_MULTIPLIER
    if smoothed_error <= anomaly_threshold: return 0.0
    if smoothed_error >= failure_threshold: return 1.0
    return float(clamp((smoothed_error - anomaly_threshold) / (failure_threshold - anomaly_threshold), 0.0, 1.0))

def compute_persistence_factor(anomaly_threshold):
    history = list(smooth_error_history)[-PERSISTENCE_LOOKBACK:]
    if not history: return 0.0
    count_above = sum(1 for v in history if v > anomaly_threshold)
    return float(clamp(count_above / float(len(history)), 0.0, 1.0))

def compute_trend_factor(anomaly_threshold, slope):
    if slope is None or slope <= 0: return 0.0
    horizon_hours      = max(MIN_RUL_POINTS * SAMPLE_INTERVAL_HOURS, SAMPLE_INTERVAL_HOURS)
    projected_increase = slope * horizon_hours
    scale = max(0.75 * anomaly_threshold, 1e-6)
    return float(clamp(projected_increase / scale, 0.0, 1.0))

def compute_maintenance_priority(health, rul, smoothed_error, anomaly_threshold, slope):
    anomaly_severity   = compute_anomaly_severity(smoothed_error, anomaly_threshold)
    health_degradation = float(clamp(1.0 - (health / 100.0), 0.0, 1.0))
    rul_risk           = float(clamp(1.0 - (rul / MAX_RUL_HOURS), 0.0, 1.0))
    trend_factor       = compute_trend_factor(anomaly_threshold, slope)
    persistence_factor = compute_persistence_factor(anomaly_threshold)
    weighted = (
        MPS_WEIGHTS["anomaly_severity"]   * anomaly_severity
        + MPS_WEIGHTS["health_degradation"] * health_degradation
        + MPS_WEIGHTS["rul_risk"]           * rul_risk
        + MPS_WEIGHTS["trend"]              * trend_factor
        + MPS_WEIGHTS["persistence"]        * persistence_factor
    )
    mps = 100.0 * weighted
    factors = {
        "anomaly_severity":   round(anomaly_severity,   4),
        "health_degradation": round(health_degradation, 4),
        "rul_risk":           round(rul_risk,           4),
        "trend_factor":       round(trend_factor,       4),
        "persistence_factor": round(persistence_factor, 4),
    }
    return float(clamp(mps, 0.0, 100.0)), factors

def determine_urgency_level(mps, health, rul, anomaly_severity):
    if mps >= 85 or health <= 15 or rul <= 4 or anomaly_severity >= 0.90: return "CRITICAL"
    if mps >= 65 or health <= 30 or rul <= 12:                             return "URGENT"
    if mps >= 45 or health <= 55 or rul <= 30:                             return "PLAN_MAINTENANCE"
    if mps >= 20 or health <= 75 or anomaly_severity > 0:                  return "WARNING"
    return "NORMAL"

def cap_urgency(urgency_level, max_allowed):
    current_index = URGENCY_ORDER.index(urgency_level)
    max_index     = URGENCY_ORDER.index(max_allowed)
    return URGENCY_ORDER[min(current_index, max_index)]

def determine_dominant_feature(contributions, is_anomaly, mps):
    if not contributions:
        return "observe", 0.0
    dominant_feature = max(contributions, key=contributions.get)
    dominant_pct     = float(contributions.get(dominant_feature, 0.0))
    if not is_anomaly and mps < 20:
        return "observe", dominant_pct
    if dominant_pct < DOMINANT_CONTRIBUTION_THRESHOLD:
        return "mixed", dominant_pct
    return dominant_feature, dominant_pct

def determine_operating_region(is_anomaly, sensor_states, dominant_feature, dominant_state, warmup_like):
    if warmup_like and sensor_states.get("oil_temp") == "LOW":
        return "WARMUP", "Cold oil / transformer warm-up state"
    if not is_anomaly and all(sensor_states.get(n) == "NORMAL" for n in FEATURE_NAMES):
        return "NORMAL_OPERATION", "Normal operating region"
    region_map = {
        ("oil_temp",      "LOW"):  ("COLD_OIL",        "Cold oil temperature state"),
        ("oil_temp",      "HIGH"): ("OIL_OVERHEAT",    "Oil overheating state"),
        ("winding_temp",  "LOW"):  ("COLD_WINDING",    "Cold winding / WTI fault state"),
        ("winding_temp",  "HIGH"): ("WINDING_OVERHEAT","Winding overheating — insulation risk"),
        ("current",       "LOW"):  ("UNDERLOAD",       "Low load current / open-circuit risk"),
        ("current",       "HIGH"): ("OVERLOAD",        "Overload condition"),
        ("vibration",     "LOW"):  ("LOW_VIBRATION",   "Low vibration / sensor fault state"),
        ("vibration",     "HIGH"): ("HIGH_VIBRATION",  "Mechanical / core looseness fault"),
        ("oil_level",     "LOW"):  ("LOW_OIL",         "Low oil level — insulation risk"),
        ("oil_level",     "HIGH"): ("HIGH_OIL",        "High oil level — conservator issue"),
    }
    key = (dominant_feature, dominant_state)
    if key in region_map:
        return region_map[key]
    return "GENERAL_DIAGNOSTIC", "Mixed operating state"

def contextualise_priority(mps, urgency_level, dominant_feature, dominant_state, operating_region):
    adjusted_mps     = float(mps)
    adjusted_urgency = urgency_level
    if operating_region == "WARMUP":
        # Current HIGH or vibration HIGH during warmup is a real fault — never cap these
        if dominant_feature in {"current", "vibration"} and dominant_state == "HIGH":
            pass  # full urgency preserved
        else:
            adjusted_mps     = min(adjusted_mps, 35.0)
            adjusted_urgency = cap_urgency(adjusted_urgency, "WARNING")
    elif dominant_state == "LOW" and dominant_feature in {"oil_temp", "current", "vibration"}:
        adjusted_mps     = min(adjusted_mps, 55.0)
        adjusted_urgency = cap_urgency(adjusted_urgency, "PLAN_MAINTENANCE")
    elif dominant_feature == "oil_level" and dominant_state == "HIGH":
        adjusted_mps     = min(adjusted_mps, 55.0)
        adjusted_urgency = cap_urgency(adjusted_urgency, "PLAN_MAINTENANCE")
    return float(adjusted_mps), adjusted_urgency

def select_rule_key(dominant_feature, dominant_state, is_anomaly, sensor_states):
    if dominant_feature == "observe":
        return ("observe", "NORMAL")
    if dominant_feature == "mixed":
        return ("mixed", "MIXED")
    if dominant_state not in {"LOW", "HIGH"}:
        return ("mixed", "MIXED") if is_anomaly else ("observe", "NORMAL")
    return (dominant_feature, dominant_state)

def build_actions(rule, urgency_level):
    actions = list(rule["actions"])
    if urgency_level == "URGENT" and rule.get("urgent_extra"):
        actions.append(rule["urgent_extra"])
    if urgency_level == "CRITICAL":
        if rule.get("urgent_extra") and rule["urgent_extra"] not in actions:
            actions.append(rule["urgent_extra"])
        if rule.get("critical_extra"):
            actions.append(rule["critical_extra"])
    return actions

def build_prescription_title(rule, urgency_level):
    key_map = {
        "NORMAL":           "normal_title",
        "WARNING":          "warning_title",
        "PLAN_MAINTENANCE": "plan_title",
        "URGENT":           "urgent_title",
        "CRITICAL":         "critical_title",
    }
    return rule.get(key_map.get(urgency_level, "normal_title"), "--")

def build_auto_action(rule, urgency_level):
    key_map = {
        "NORMAL":           "auto_normal",
        "WARNING":          "auto_warning",
        "PLAN_MAINTENANCE": "auto_plan",
        "URGENT":           "auto_urgent",
        "CRITICAL":         "auto_critical",
    }
    return rule.get(key_map.get(urgency_level, "auto_normal"), "monitor")

def build_prescription_reason(urgency_level, mps, dominant_feature, dominant_pct, dominant_state, operating_region, health, rul, factors, warmup_like):
    if dominant_feature == "observe":
        return (f"Transformer condition is stable. Operating region: {operating_region}. "
                f"MPS: {mps:.2f}, health: {health:.1f}%, RUL: {rul:.1f} h. No immediate action required.")
    state_text = dominant_state.lower() if dominant_state else "unknown"
    dom_text = (
        f"{dominant_feature.replace('_',' ').capitalize()} is the dominant contributor ({dominant_pct:.1f}%) in a {state_text} state. "
        if dominant_feature not in {"mixed", "observe"}
        else "No single sensor is strongly dominant — a general inspection is recommended. "
    )
    warmup_text = "Cold oil / warm-up condition detected. " if warmup_like else ""
    return (warmup_text + dom_text
            + f"Operating region: {operating_region}. "
            + f"Urgency: {urgency_level} | MPS: {mps:.2f} | Health: {health:.1f}% | RUL: {rul:.1f} h. "
            + f"Score drivers — anomaly severity: {factors['anomaly_severity']:.2f}, "
            + f"health degradation: {factors['health_degradation']:.2f}, "
            + f"RUL risk: {factors['rul_risk']:.2f}, "
            + f"trend: {factors['trend_factor']:.2f}, "
            + f"persistence: {factors['persistence_factor']:.2f}.")

def compute_prescriptive_layer(raw_window, contributions, is_anomaly, health, rul, smoothed_error, anomaly_threshold, slope, scaler_obj):
    latest_values    = {name: float(raw_window[-1][FEATURE_INDEX[name]]) for name in FEATURE_NAMES}
    operating_bands  = get_operating_bands(scaler_obj)
    sensor_states, band_distances = compute_sensor_states(latest_values, operating_bands)
    warmup_like, warmup_exit_temp = compute_condition_warmup_flag(raw_window, operating_bands)

    mps, mps_factors = compute_maintenance_priority(
        health=health, rul=rul, smoothed_error=smoothed_error,
        anomaly_threshold=anomaly_threshold, slope=slope)

    dominant_feature, dominant_pct = determine_dominant_feature(contributions, is_anomaly, mps)
    dominant_state = sensor_states.get(dominant_feature, "NORMAL") if dominant_feature in FEATURE_NAMES else (
        "MIXED" if dominant_feature == "mixed" else "NORMAL")

    operating_region, prescription_context = determine_operating_region(
        is_anomaly=is_anomaly, sensor_states=sensor_states,
        dominant_feature=dominant_feature, dominant_state=dominant_state,
        warmup_like=warmup_like)

    urgency_level = determine_urgency_level(mps=mps, health=health, rul=rul, anomaly_severity=mps_factors["anomaly_severity"])
    adjusted_mps, urgency_level = contextualise_priority(
        mps=mps, urgency_level=urgency_level,
        dominant_feature=dominant_feature, dominant_state=dominant_state,
        operating_region=operating_region)

    rule_key = select_rule_key(dominant_feature, dominant_state, is_anomaly, sensor_states)
    rule     = PRESCRIPTION_RULES[rule_key]

    return {
        "maintenance_priority_score":   round(float(adjusted_mps), 2),
        "urgency_level":                urgency_level,
        "prescription_type":            rule["type"],
        "prescription_category_label":  rule["label"],
        "prescription_title":           build_prescription_title(rule, urgency_level),
        "prescription_actions":         build_actions(rule, urgency_level),
        "prescription_reason":          build_prescription_reason(urgency_level, adjusted_mps, dominant_feature,
                                            dominant_pct, dominant_state, operating_region, health, rul, mps_factors, warmup_like),
        "prescription_context":         prescription_context,
        "auto_action":                  build_auto_action(rule, urgency_level),
        "anomaly_severity":             round(float(mps_factors["anomaly_severity"]),   4),
        "health_degradation":           round(float(mps_factors["health_degradation"]), 4),
        "rul_risk":                     round(float(mps_factors["rul_risk"]),           4),
        "trend_factor":                 round(float(mps_factors["trend_factor"]),       4),
        "persistence_factor":           round(float(mps_factors["persistence_factor"]), 4),
        "current_state":                sensor_states["current"],
        "oil_temp_state":               sensor_states["oil_temp"],
        "winding_temp_state":           sensor_states["winding_temp"],
        "vibration_state":              sensor_states["vibration"],
        "oil_level_state":              sensor_states["oil_level"],
        "dominant_feature":             dominant_feature,
        "dominant_state":               dominant_state,
        "operating_region":             operating_region,
        "condition_warmup_flag":        bool(warmup_like),
        "warmup_exit_temperature":      round(float(warmup_exit_temp), 4),
        "operating_bands":              {
            name: {"low": round(float(operating_bands[name]["low"]), 4),
                   "high": round(float(operating_bands[name]["high"]), 4)}
            for name in FEATURE_NAMES
        },
        "state_distance": {name: round(float(band_distances[name]), 4) for name in FEATURE_NAMES},
    }

# ============================================================
# MODEL LOADING
# ============================================================
def load_all():
    global model, scaler, threshold, is_loaded, last_adaptive_threshold
    if is_loaded:
        return
    for path, label in [(MODEL_PATH,"Model"),(SCALER_PATH,"Scaler"),(THRESHOLD_PATH,"Threshold")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
    print("Loading model...")
    model_local = ort.InferenceSession(MODEL_PATH)
    print("Loading scaler...")
    scaler_local = joblib.load(SCALER_PATH)
    print("Loading threshold...")
    threshold_local = float(np.load(THRESHOLD_PATH, allow_pickle=True))
    print(f"Threshold: {threshold_local}")
    model     = model_local
    scaler    = scaler_local
    threshold = threshold_local
    last_adaptive_threshold = threshold_local
    is_loaded = True

def ensure_loaded():
    global startup_error
    if backend_ready():
        return True
    try:
        load_all()
        startup_error = None
        return True
    except Exception as e:
        startup_error = str(e)
        traceback.print_exc()
        return False

# ============================================================
# ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message":    "Transformer Health Monitor backend running.",
        "ready":      backend_ready(),
        "features":   FEATURE_NAMES,
        "window_size":WINDOW_SIZE,
        "threshold":  last_adaptive_threshold if last_adaptive_threshold is not None else threshold,
        "adaptive_threshold_enabled": ADAPTIVE_THRESHOLD_ENABLED,
        "startup_error": startup_error,
    })

@app.route("/health", methods=["GET"])
def health_check():
    code = 200 if backend_ready() else 503
    return jsonify({"ok": backend_ready(), "startup_error": startup_error}), code

@app.route("/batch", methods=["POST"])
def batch_predict():
    global last_adaptive_threshold
    if not ensure_loaded():
        return jsonify({"error": "Backend not ready", "details": startup_error or "Model not loaded."}), 503

    try:
        t0 = time.time()
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Missing or invalid JSON body."}), 400

        readings = payload.get("readings")
        valid, message = validate_input(readings)
        if not valid:
            return jsonify({"error": message}), 400

        raw_window = np.array(readings, dtype=np.float32)

        # ── Early warmup / idle detection (before LSTM inference) ──
        operating_bands_early          = get_operating_bands(scaler)
        warmup_like_early, warmup_exit = compute_condition_warmup_flag(raw_window, operating_bands_early)
        latest_current                 = float(raw_window[-1][FEATURE_INDEX["current"]])
        latest_vibration               = float(raw_window[-1][FEATURE_INDEX["vibration"]])
        is_idle                        = latest_current < MIN_LOAD_CURRENT

        # Current or vibration genuinely HIGH overrides warmup/idle suppression
        current_high       = latest_current   > operating_bands_early["current"]["high"]
        vibration_high     = latest_vibration > operating_bands_early["vibration"]["high"]
        exempt_from_warmup = current_high or vibration_high

        # Effective warmup: suppressed only if no critical electrical/mechanical fault present
        effective_warmup = (warmup_like_early or is_idle) and not exempt_from_warmup

        scaled_window = scaler.transform(raw_window)
        x_input       = np.expand_dims(scaled_window, axis=0)
        x_pred        = model.run(None, {"input": x_input})[0]

        base_threshold     = float(threshold)
        adaptive_threshold, adaptive_ready = compute_adaptive_threshold(base_threshold, adaptive_healthy_error_history)
        active_threshold   = float(adaptive_threshold)
        last_adaptive_threshold = active_threshold
        adaptive_threshold_history.append(active_threshold)

        reconstruction_error = compute_total_error(x_input, x_pred)
        is_anomaly           = reconstruction_error > active_threshold and not effective_warmup
        smoothed_error       = update_smoothed_error(reconstruction_error)
        health               = compute_health(smoothed_error, active_threshold, warmup_like=effective_warmup)

        feature_errors            = compute_feature_errors(x_input, x_pred)
        contributions, main_cause = compute_sensor_contributions(feature_errors)

        rul, rul_state = estimate_rul(smoothed_error, active_threshold, warmup_like=effective_warmup)
        slope          = estimate_trend()

        ood_score, ood_details, ood_direction_details, ood_feature = compute_ood_score(raw_window, scaler)
        confidence_score, confidence_level, confidence_sources     = compute_confidence(ood_score, active_threshold)
        rul_min, rul_max, rul_std = compute_rul_range(rul, confidence_score, health, rul_state)
        uncertainty_reason        = build_uncertainty_reason(ood_score, ood_feature, confidence_sources)

        prescriptive = compute_prescriptive_layer(
            raw_window=raw_window, contributions=contributions, is_anomaly=is_anomaly,
            health=health, rul=rul, smoothed_error=smoothed_error,
            anomaly_threshold=active_threshold, slope=slope, scaler_obj=scaler)

        adaptive_update = maybe_update_adaptive_history(
            raw_window=raw_window,
            reconstruction_error=reconstruction_error,
            active_threshold=active_threshold,
            health=health,
            ood_score=ood_score,
            scaler_obj=scaler,
            exempt_from_warmup=exempt_from_warmup)
        adaptive_summary = get_adaptive_history_summary(adaptive_healthy_error_history, base_threshold)

        status, led_status = derive_status_and_led(is_anomaly, health, prescriptive["urgency_level"])
        latest = raw_window[-1]

        response = {
            "is_anomaly":           bool(is_anomaly),
            "status":               status,
            "led_status":           led_status,
            "health":               round(float(health), 2),
            "rul_hours":            round(float(rul),    2),
            "rul_state":            rul_state,
            "rul_min":              round(float(rul_min), 2),
            "rul_max":              round(float(rul_max), 2),
            "rul_std":              round(float(rul_std), 2),
            "main_cause":           main_cause,
            "sensor_contributions": {
                name: round(float(contributions[name]), 2) for name in FEATURE_NAMES
            },
            "latest_values": {
                "current":      round(float(latest[FEATURE_INDEX["current"]]),      4),
                "oil_temp":     round(float(latest[FEATURE_INDEX["oil_temp"]]),     4),
                "winding_temp": round(float(latest[FEATURE_INDEX["winding_temp"]]), 4),
                "vibration":    round(float(latest[FEATURE_INDEX["vibration"]]),    4),
                "oil_level":    round(float(latest[FEATURE_INDEX["oil_level"]]),    4),
            },
            "reconstruction_error":       round(float(reconstruction_error), 6),
            "smoothed_error":             round(float(smoothed_error),       6),
            "base_threshold":             round(float(base_threshold),       6),
            "adaptive_threshold":         round(float(active_threshold),     6),
            "failure_threshold":          round(float(active_threshold * FAILURE_MULTIPLIER), 6),
            "threshold_mode":             "adaptive" if adaptive_ready and ADAPTIVE_THRESHOLD_ENABLED else "fixed_base_threshold",
            "adaptive_threshold_enabled": bool(ADAPTIVE_THRESHOLD_ENABLED),
            "adaptive_threshold_ready":   bool(adaptive_ready),
            "adaptive_history_count":     adaptive_summary["count"],
            "adaptive_history_ready":     adaptive_summary["ready"],
            "adaptive_update_applied":    bool(adaptive_update["applied"]),
            "adaptive_update_reason":     adaptive_update["reason"],
            "adaptive_warmup_block":      bool(adaptive_update["warmup_like"]),
            "adaptive_warmup_exit_temperature": round(float(adaptive_update["warmup_exit_temperature"]), 4) if adaptive_update["warmup_exit_temperature"] is not None else None,
            "degradation_rate":           round(float(slope), 6) if slope is not None else None,
            "confidence_level":           confidence_level,
            "confidence_score":           round(float(confidence_score), 2),
            "ood_score":                  round(float(ood_score), 6) if ood_score is not None else None,
            "ood_direction_details":      ood_direction_details,
            "uncertainty_reason":         uncertainty_reason,
            "uncertainty_sources":        confidence_sources,
            **prescriptive,
            "analysis_timestamp": int(time.time()),
        }

        write_to_firebase(response)
        save_history_to_firebase()
        print(f"/batch done in {round(time.time()-t0,3)}s | anomaly={is_anomaly} | health={health:.1f}% | urgency={prescriptive['urgency_level']}")
        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# ============================================================
# STARTUP
# ============================================================
init_firebase()
ensure_loaded()
load_history_from_firebase()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
