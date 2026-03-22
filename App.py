import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diesel Engine Leak Detection", page_icon="🔧", layout="wide")
st.title("🔧 Diesel Engine Leak Detection & Isolation System")
st.markdown("**Caterpillar Tech Challenge — AI Syndicate**")
st.markdown("---")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'airflow_meter':        np.random.normal(450, 10, n),
        'compressor_inlet_p':   np.random.normal(101, 2,  n),
        'compressor_outlet_p':  np.random.normal(180, 5,  n),
        'intercooler_inlet_p':  np.random.normal(178, 5,  n),
        'intercooler_outlet_p': np.random.normal(175, 5,  n),
        'intake_manifold_p':    np.random.normal(170, 5,  n),
        'exhaust_manifold_p':   np.random.normal(120, 5,  n),
        'turbine_inlet_p':      np.random.normal(118, 5,  n),
        'turbine_outlet_p':     np.random.normal(105, 3,  n),
        'tailpipe_p':           np.random.normal(102, 2,  n),
        'intake_temp':          np.random.normal(45,  3,  n),
        'exhaust_temp':         np.random.normal(380, 20, n),
        'engine_speed':         np.random.normal(1800,50, n),
        'fuel_flow':            np.random.normal(25,  1,  n),
    })
    feats = list(df.columns)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats])
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)
    return model, scaler, feats

model, scaler, features = train_model()

ACTIONS = {
    1: 'Inspect ducting between airflow meter and compressor. Check clamps and seals.',
    2: 'Inspect charge air cooler piping, intercooler hoses and intake manifold gaskets.',
    3: 'Inspect exhaust manifold gaskets, turbine flanges and DPF/SCR connections.',
    4: 'Inspect test cell intake and exhaust ducting connections and interface seals.',
}
ZONES = {
    1: 'Zone 1 — Airflow meter to compressor inlet',
    2: 'Zone 2 — Compressor outlet to intake ports',
    3: 'Zone 3 — Exhaust manifold to aftertreatment',
    4: 'Zone 4 — Test cell ducting interfaces',
}

def detect(row):
    z1 = ((450 - row['airflow_meter']) / 450 * 0.6 + (101 - row['compressor_inlet_p']) / 101 * 0.4) * 100
    z2 = (max(0, (row['compressor_outlet_p'] - row['intake_manifold_p']) - 10) / row['compressor_outlet_p']) * 100
    z3 = (max(0, 20 - (row['exhaust_manifold_p'] - row['tailpipe_p'])) / row['exhaust_manifold_p']) * 100
    z4 = (abs(row['tailpipe_p'] - 101) / 101 * 0.5 + max(0, 450 - row['airflow_meter']) / 450 * 0.5) * 100
    scores = {1: round(z1,2), 2: round(z2,2), 3: round(z3,2), 4: round(z4,2)}
    top_zone = max(scores, key=scores.get)
    top_score = scores[top_zone]
    X = scaler.transform([[row[f] for f in features]])
    ml_pred = model.predict(X)[0]
    ml_score_val = model.decision_function(X)[0]
    ml_conf = round(min(99, max(1, (-ml_score_val + 0.2) * 100)), 1)
    is_anom = ml_pred == -1
    physics_leak = top_score >= 8.0
    if physics_leak and is_anom:
        flag = '🔴 LEAK CONFIRMED'
        conf = round((min(99, top_score * 2.5) + ml_conf) / 2, 1)
        color = 'red'
    elif physics_leak or is_anom:
        flag = '🟡 POSSIBLE LEAK'
        conf = round(min(99, top_score * 2.5) * 0.7, 1)
        color = 'orange'
    else:
        flag = '🟢 NO LEAK'
        conf = round((100 - top_score * 2) * 0.9, 1)
        color = 'green'
    return {
        'flag': flag, 'confidence': conf, 'color': color,
        'location': ZONES[top_zone], 'action': ACTIONS[top_zone],
        'scores': scores, 'ml_anomaly': is_anom, 'ml_conf': ml_conf
    }

st.sidebar.header("🎛️ Sensor Inputs")
mode = st.sidebar.radio("Input mode", [
    "Normal operation",
    "Simulate Zone 1 leak",
    "Simulate Zone 2 leak",
    "Simulate Zone 3 leak",
    "Simulate Zone 4 leak"
])

presets = {
    "Normal operation":     [450,101,180,178,175,170,120,118,105,102,45,380,1800,25],
    "Simulate Zone 1 leak": [380, 88,180,178,175,170,120,118,105,102,45,380,1800,25],
    "Simulate Zone 2 leak": [450,101,140,135,128,120,120,118,105,102,45,380,1800,25],
    "Simulate Zone 3 leak": [450,101,180,178,175,170, 90, 85,100,102,45,380,1800,25],
    "Simulate Zone 4 leak": [390,101,180,178,175,170,120,118,105, 98,45,380,1800,25],
}
d = presets[mode]

airflow      = st.sidebar.slider("Airflow meter (kg/hr)",       300, 550, d[0])
comp_in_p    = st.sidebar.slider("Compressor inlet P (kPa)",     80, 115, d[1])
comp_out_p   = st.sidebar.slider("Compressor outlet P (kPa)",   120, 220, d[2])
ic_in_p      = st.sidebar.slider("Intercooler inlet P (kPa)",   118, 218, d[3])
ic_out_p     = st.sidebar.slider("Intercooler outlet P (kPa)",  115, 215, d[4])
intake_p     = st.sidebar.slider("Intake manifold P (kPa)",     110, 210, d[5])
exh_p        = st.sidebar.slider("Exhaust manifold P (kPa)",     80, 160, d[6])
turb_in_p    = st.sidebar.slider("Turbine inlet P (kPa)",        78, 158, d[7])
turb_out_p   = st.sidebar.slider("Turbine outlet P (kPa)",       90, 130, d[8])
tail_p       = st.sidebar.slider("Tailpipe P (kPa)",             95, 115, d[9])
intake_temp  = st.sidebar.slider("Intake temp (°C)",             20,  80, d[10])
exhaust_temp = st.sidebar.slider("Exhaust temp (°C)",           300, 500, d[11])
eng_speed    = st.sidebar.slider("Engine speed (RPM)",         1400,2200, d[12])
fuel_flow    = st.sidebar.slider("Fuel flow (kg/hr)",            18,  35, d[13])

row = {
    'airflow_meter': airflow, 'compressor_inlet_p': comp_in_p,
    'compressor_outlet_p': comp_out_p, 'intercooler_inlet_p': ic_in_p,
    'intercooler_outlet_p': ic_out_p, 'intake_manifold_p': intake_p,
    'exhaust_manifold_p': exh_p, 'turbine_inlet_p': turb_in_p,
    'turbine_outlet_p': turb_out_p, 'tailpipe_p': tail_p,
    'intake_temp': intake_temp, 'exhaust_temp': exhaust_temp,
    'engine_speed': eng_speed, 'fuel_flow': fuel_flow,
}

result = detect(row)

c1, c2, c3 = st.columns(3)
c1.metric("🚨 STATUS",      result['flag'])
c2.metric("📊 CONFIDENCE",  str(result['confidence']) + "%")
c3.metric("🤖 ML ANOMALY",  "YES ⚠️" if result['ml_anomaly'] else "NO ✅")

st.markdown("---")

c4, c5 = st.columns(2)
with c4:
    if result['color'] == 'red':
        st.error("📍 LOCATION: " + result['location'])
    elif result['color'] == 'orange':
        st.warning("📍 LOCATION: " + result['location'])
    else:
        st.success("📍 LOCATION: " + result['location'])
with c5:
    st.info("🔧 ACTION: " + result['action'])

st.markdown("---")
st.subheader("📈 Zone Leak Scores")

zone_labels = [ZONES[i].split('—')[1].strip() for i in range(1,5)]
zone_scores = [result['scores'][i] for i in range(1,5)]
bar_colors  = ['red' if s >= 8 else 'orange' if s >= 4 else 'green' for s in zone_scores]

fig1 = go.Figure(go.Bar(
    x=zone_labels, y=zone_scores,
    marker_color=bar_colors,
    text=[str(round(s,1)) for s in zone_scores],
    textposition='outside'
))
fig1.add_hline(y=8, line_dash="dash", line_color="red", annotation_text="Leak threshold (8.0)")
fig1.update_layout(yaxis_title="Leak Score", height=350,
                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌡️ Pressure Profile Across Engine Path")

sensor_labels = ['Comp inlet','Comp outlet','IC inlet','IC outlet',
                 'Intake manifold','Exhaust manifold','Turbine inlet','Turbine outlet','Tailpipe']
sensor_keys   = ['compressor_inlet_p','compressor_outlet_p','intercooler_inlet_p',
                 'intercooler_outlet_p','intake_manifold_p','exhaust_manifold_p',
                 'turbine_inlet_p','turbine_outlet_p','tailpipe_p']
normal_vals   = [101,180,178,175,170,120,118,105,102]
current_vals  = [row[k] for k in sensor_keys]

line_color = 'red' if result['color']=='red' else 'orange' if result['color']=='orange' else 'blue'
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sensor_labels, y=normal_vals,
    name='Normal baseline', line=dict(color='green', dash='dash'), mode='lines+markers'))
fig2.add_trace(go.Scatter(x=sensor_labels, y=current_vals,
    name='Current reading', line=dict(color=line_color), mode='lines+markers'))
fig2.update_layout(yaxis_title="Pressure (kPa)", height=350,
                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Caterpillar Tech Challenge 2026 · AI Syndicate · Physics + ML Hybrid Detection · Non-invasive · Real-time · No hardware changes required")
