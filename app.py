
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud_config import CLASS_DESCRIPTION, FEATURES, FRAUD_SIGNAL_MAP

st.set_page_config(
    page_title='FedSecure: A Privacy Preserving Fraud Detection using Federated Learning',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded',
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / 'artifacts'
DATA_PATH = BASE_DIR / 'dataset.csv'
MODEL_PATH = ARTIFACT_DIR / "fraud_model_artifact.joblib"
METRICS_PATH = ARTIFACT_DIR / 'metrics.json'
RESEARCH_PATH = ARTIFACT_DIR / 'research_metadata.json'

FED_METRICS_PATH = ARTIFACT_DIR / "federated_metrics.json"



@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error('Artifacts not found. Run generate_synthetic_data.py and train_models.py first.')
    st.stop()

artifact = load_model_artifact()
df = load_data()
metrics = load_json(METRICS_PATH)
research = load_json(RESEARCH_PATH)

pipeline = artifact['model']
label_encoder = artifact['label_encoder']
best_model_name = artifact.get('best_model_name', 'Unknown')



st.sidebar.title('Navigation')

page = st.sidebar.radio(
    'Choose section',
    [
        'Federated Learning (FedAvg & FedSecure)',
        'Executive Summary',
        'Data Observatory',
        'Algorithm Benchmarking',
        'Single Transaction Scoring',
        'Batch Scoring'
    ],
)

st.sidebar.markdown('---')
st.sidebar.info(
    f"Best trained model: **{best_model_name}**\n\n"
    f"Classes: **{len(artifact['class_labels'])}**\n\n"
    f"Features: **{len(FEATURES)}**"
)



if page == 'Federated Learning (FedAvg & FedSecure)':

    st.title(' Federated Learning (FedAvg & FedSecure)')

    st.markdown("###  Real Federated Training")

    if FED_METRICS_PATH.exists():
        fed_metrics = load_json(FED_METRICS_PATH)

        col1, col2, col3 = st.columns(3)
        col1.metric("Clients", fed_metrics["num_clients"])
        col2.metric("Rounds", fed_metrics["rounds"])
        col3.metric("Aggregation", fed_metrics["aggregation_type"])

        history_df = pd.DataFrame(fed_metrics["history"])

        fig = px.line(
            history_df,
            x="round",
            y=["accuracy", "f1_macro"],
            title="Federated Training Performance",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Run federated_train.py to generate results")

    st.markdown('---')




elif page == 'Executive Summary':
    st.title('FedSecure: A Privacy Preserving Fraud Detection using Federated Learning')
    st.markdown(
        'A research-style Streamlit system for synthetic **credit card fraud**, **identity theft**, '
        '**money laundering**, **transaction laundering**, **UPI fraud**, and **ATM skimming fraud** detection.'
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Dataset Size', f"{len(df):,}")
    col2.metric('Fraud Classes', len(artifact['class_labels']) - 1)
    col3.metric('Best Model', best_model_name)
    col4.metric('Macro F1', f"{metrics[best_model_name]['f1_macro']:.3f}")

    left, right = st.columns([1.2, 1])
    with left:
        class_counts = df['fraud_type'].value_counts().reset_index()
        class_counts.columns = ['fraud_type', 'count']
        fig = px.pie(class_counts, names='fraud_type', values='count', hole=0.48)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader('Research Framing')
        st.write(research['problem_scope'])
        st.write(f"**Hypothesis:** {research['recommended_hypothesis']}")

elif page == 'Data Observatory':
    st.title(' Data Observatory')

    feature = st.selectbox('Select feature', FEATURES)
    fig = px.histogram(df, x=feature, color='fraud_type')
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Algorithm Benchmarking':
    st.title(' Algorithm Benchmarking')

    results_df = pd.DataFrame(metrics).T.reset_index()
    results_df = results_df.rename(columns={'index': 'algorithm'})
    fig = px.bar(results_df, x='algorithm', y='f1_macro')
    st.plotly_chart(fig)

elif page == 'Single Transaction Scoring':
    st.title(' Single Transaction Scoring')

    user_input = {f: st.number_input(f, value=1.0) for f in FEATURES}

    if st.button('Predict'):
        pred = pipeline.predict(pd.DataFrame([user_input]))[0]
        label = label_encoder.inverse_transform([pred])[0]
        st.success(f'Prediction: {label}')

elif page == 'Batch Scoring':
    st.title(' Batch Scoring')

    uploaded = st.file_uploader('Upload CSV')

    if uploaded:
        data = pd.read_csv(uploaded)
        preds = pipeline.predict(data[FEATURES])
        data['Prediction'] = label_encoder.inverse_transform(preds)
        st.dataframe(data.head())

