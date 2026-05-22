import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.title("WEB PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG")

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # =========================
    # ĐỌC FILE AN TOÀN
    # =========================

    try:

        df = pd.read_csv(
            uploaded_file,
            encoding='utf-8',
            low_memory=False
        )

    except:

        df = pd.read_csv(
            uploaded_file,
            encoding='latin1',
            low_memory=False
        )

    st.success("Đọc file thành công")

    st.dataframe(df.head())

    # =========================
    # COPY DATA
    # =========================

    model_df = df.copy()

    # Encode type
    if 'type' in model_df.columns:

        le = LabelEncoder()

        model_df['type'] = le.fit_transform(
            model_df['type']
        )

    # Xóa text columns
    remove_cols = []

    for col in ['nameOrig', 'nameDest']:

        if col in model_df.columns:
            remove_cols.append(col)

    model_df = model_df.drop(
        columns=remove_cols
    )

    # Tách X
    if 'isFraud' in model_df.columns:

        X = model_df.drop(
            columns=['isFraud']
        )

    else:
        X = model_df

    # Chuẩn hóa
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # AI model
    st.subheader(
        "🤖 AI đang phân tích dữ liệu..."
    )

    model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42
    )

    model.fit(X_scaled)

    pred = model.predict(X_scaled)

    model_df['anomaly'] = pd.Series(pred).map({
        1: 0,
        -1: 1
    })

    # Kết quả
    anomaly_count = model_df[
        'anomaly'
    ].sum()

    st.success(
        f"Phát hiện {anomaly_count} giao dịch bất thường"
    )

    anomaly_df = model_df[
        model_df['anomaly'] == 1
    ]

    st.subheader(
        "🚨 Giao dịch bất thường"
    )

    st.dataframe(
        anomaly_df.head(50)
    )

else:

    st.info(
        "Vui lòng upload file CSV"
    )