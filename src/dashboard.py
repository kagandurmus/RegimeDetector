# REGIME DETECTOR - PUBLIC INTERFACE
import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="RegimeDetector", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for compact layout
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
    .stMetric { background-color: #161616; padding: 5px; border-radius: 5px; }
    [data-testid="stExpander"] { background-color: #161616 !important; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Data Loading
JSON_PATH = "data/dashboard/latest_stats.json"
PARQUET_PATH = "data/processed/regime_data.parquet"

def load_latest():
    if not os.path.exists(JSON_PATH): return None
    with open(JSON_PATH, "r") as f: return json.load(f)

def load_history():
    if not os.path.exists(PARQUET_PATH): return None
    # Load last 30 days from your processed parquet file
    df = pd.read_parquet(PARQUET_PATH)
    return df.tail(30)

data = load_latest()
history_df = load_history()

if data:
    # --- HEADER ---
    h_col1, h_col2 = st.columns([4, 1])
    h_col1.title("Institutional Regime Dashboard")
    if h_col2.button("Refresh", use_container_width=True): st.rerun()
    st.caption(f"Last updated: {data['timestamp']}")

    # --- TOP SECTION: NEWS (3 ROWS STACKED) ---
    st.subheader("Market Catalysts")
    for news in data['catalysts'][:3]:
        with st.expander(f"{news['source'].upper()}: {news['headline']}", expanded=False):
            st.write(f"{news['age_minutes']}m ago | Timestamp: {news['timestamp']}")

    st.divider()

    # --- MIDDLE SECTION: METRICS & CONFIDENCE ---
    col_sent, col_ctx, col_conf = st.columns([1, 1, 1.5])

    with col_sent:
        st.subheader("Market Sentiment")
        fg_val = data['indicators']['fear_greed']
        color = "#ef553b" if fg_val < 45 else "#00cc96"
        st.markdown(f"""
            <div style='background-color:#161616; padding: 20px; border-radius: 8px; text-align: center; border-left: 5px solid {color};'>
                <h2 style='color:{color}; margin:0;'>{fg_val}</h2>
                <p style='margin:0; font-size: 12px;'>{ "Extreme Fear" if fg_val <= 24 else "Fear" if fg_val <=40 else "Neutral" if fg_val <=50 else "Greed" if fg_val <=75 else "Extreme Greed" }</p>
            </div>
        """, unsafe_allow_html=True)

    with col_ctx:
        st.subheader("Indicators")
        m1, m2 = st.columns(2)
        m1.metric("RSI", data['indicators']['rsi'])
        m1.metric("Volatility", f"{data['indicators']['volatility']:.4f}")
        m2.metric("VPA", f"{data['indicators']['vpa']:.4f}")
        m2.metric("Regime", data['prediction']['regime'])

    with col_conf:
        st.subheader("XGBoost Confidence")
        conf = data['prediction']['confidence']
        fig_conf = go.Figure(go.Bar(
            x=[conf['Bull_Prob'], conf['Neutral_Prob'], conf['Bear_Prob']],
            y=['Bull', 'Neut', 'Bear'],
            orientation='h', marker_color=['#00cc96', '#636efa', '#ef553b'],
            text=[f"{x*100:.1f}%" for x in [conf['Bull_Prob'], conf['Neutral_Prob'], conf['Bear_Prob']]],
            textposition='auto'
        ))
        fig_conf.update_layout(height=160, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_conf, use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # --- BOTTOM SECTION: SHAP & TREND (SIDE BY SIDE) ---
    bot_left, bot_right = st.columns([1, 1])

    with bot_left:
        st.subheader("SHAP Impact")
        shap_df = pd.DataFrame(list(data['shap'].items()), columns=['F', 'V']).sort_values('V')
        fig_shap = px.bar(shap_df, x='V', y='F', orientation='h', 
                          color='V', color_continuous_scale=['#ef553b', '#00cc96'])
        fig_shap.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, coloraxis_showscale=False,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               xaxis=dict(title=None), yaxis=dict(title=None))
        st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})

    with bot_right:
        st.subheader("30-Day Regime Trend")
        if history_df is not None:
            fig_trend = px.line(history_df, y='Regime', render_mode="svg")
            fig_trend.update_traces(line_color='#636efa', line_width=2)
            fig_trend.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0),
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(showgrid=False, title=None), yaxis=dict(gridcolor='#333', title=None))
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No historical parquet data found.")