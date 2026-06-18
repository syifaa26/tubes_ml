from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_THRESHOLD,
    FEATURE_HELP,
    FEATURE_LABELS,
    MODEL_OPTIONS,
    NUMERIC_FEATURES,
)
from src.model_utils import load_model, load_scaler, predict_transactions


st.set_page_config(
    page_title="Fraud Detection",
    page_icon="FD",
    layout="wide",
)


@st.cache_resource
def get_model(model_name: str):
    return load_model(model_name)


@st.cache_resource
def get_scaler():
    return load_scaler()


def yes_no_input(label: str, help_text: str = "") -> int:
    choice = st.segmented_control(
        label,
        options=["Tidak", "Ya"],
        default="Tidak",
        help=help_text,
    )
    return 1 if choice == "Ya" else 0


def build_single_transaction_form() -> dict[str, float | int]:
    st.subheader("Input Transaksi")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        distance_from_home = st.number_input(
            FEATURE_LABELS["distance_from_home"],
            min_value=0.0,
            value=10.0,
            step=1.0,
            help=FEATURE_HELP["distance_from_home"],
        )
        repeat_retailer = yes_no_input(
            FEATURE_LABELS["repeat_retailer"],
            FEATURE_HELP["repeat_retailer"],
        )

    with col_b:
        distance_from_last_transaction = st.number_input(
            FEATURE_LABELS["distance_from_last_transaction"],
            min_value=0.0,
            value=2.0,
            step=1.0,
            help=FEATURE_HELP["distance_from_last_transaction"],
        )
        used_chip = yes_no_input(
            FEATURE_LABELS["used_chip"],
            FEATURE_HELP["used_chip"],
        )

    with col_c:
        ratio_to_median_purchase_price = st.number_input(
            FEATURE_LABELS["ratio_to_median_purchase_price"],
            min_value=0.0,
            value=1.5,
            step=0.1,
            help=FEATURE_HELP["ratio_to_median_purchase_price"],
        )
        used_pin_number = yes_no_input(
            FEATURE_LABELS["used_pin_number"],
            FEATURE_HELP["used_pin_number"],
        )
        online_order = yes_no_input(
            FEATURE_LABELS["online_order"],
            FEATURE_HELP["online_order"],
        )

    return {
        "distance_from_home": distance_from_home,
        "distance_from_last_transaction": distance_from_last_transaction,
        "ratio_to_median_purchase_price": ratio_to_median_purchase_price,
        "repeat_retailer": repeat_retailer,
        "used_chip": used_chip,
        "used_pin_number": used_pin_number,
        "online_order": online_order,
    }


def show_prediction(result: pd.DataFrame) -> None:
    row = result.iloc[0]
    probability = float(row["fraud_probability"])
    label = str(row["prediction_label"])

    left, right = st.columns([1, 2])
    with left:
        st.metric("Risiko fraud", f"{probability * 100:.2f}%")
        st.metric("Keputusan", label)

    with right:
        if label == "Fraud":
            st.error("INDIKASI FRAUD: transaksi perlu ditinjau manual.")
        else:
            st.success("TRANSAKSI AMAN: tidak terdeteksi sebagai fraud.")

    st.dataframe(result, use_container_width=True, hide_index=True)


st.title("Fraud Detection Transaction Checker")
st.caption("Deployment model deteksi fraud berbasis file .pkl.")

with st.sidebar:
    st.header("Pengaturan")
    model_name = st.selectbox(
        "Model",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL_NAME),
    )
    threshold = st.slider(
        "Threshold fraud",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
    )
    st.divider()
    st.write("Kolom numerik:")
    st.code("\n".join(NUMERIC_FEATURES))

model = get_model(model_name)
scaler = get_scaler()

tab_single, tab_batch = st.tabs(["Prediksi Tunggal", "Prediksi CSV"])

with tab_single:
    transaction = build_single_transaction_form()
    if st.button("Cek Transaksi", type="primary", use_container_width=True):
        prediction = predict_transactions(model, transaction, threshold=threshold, scaler=scaler)
        show_prediction(prediction)

with tab_batch:
    st.subheader("Upload File CSV")
    uploaded_file = st.file_uploader("CSV harus memiliki 7 kolom fitur yang sama.", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        predictions = predict_transactions(model, data, threshold=threshold, scaler=scaler)

        total = len(predictions)
        fraud_count = int((predictions["prediction"] == 1).sum())
        safe_count = total - fraud_count

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Total transaksi", total)
        metric_b.metric("Fraud", fraud_count)
        metric_c.metric("Non-Fraud", safe_count)

        st.dataframe(predictions, use_container_width=True, hide_index=True)
        st.download_button(
            "Download hasil prediksi",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
