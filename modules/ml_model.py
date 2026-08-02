import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(ttl=3600)
def train_linear_regression(df, feature_cols, target_col):
    """
    Huấn luyện mô hình Linear Regression dựa trên các thuộc tính chọn lọc.
    """
    clean_df = df[feature_cols + [target_col]].dropna()

    if clean_df.empty or len(clean_df) < 5:
        return None, None, None

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "intercept": model.intercept_,
        "coefficients": dict(zip(feature_cols, model.coef_)),
        "n_samples": len(clean_df)
    }

    results_df = clean_df.copy()
    results_df["Predicted"] = y_pred
    results_df["Residual"] = y - y_pred

    return model, metrics, results_df

def evaluate_new_data(model, new_df, feature_cols, target_col=None):
    """
    Dự báo trên file dữ liệu mới. Nếu file mới có sẵn cột Target, tính toán sai số thực tế.
    """
    # Lọc bỏ dòng thiếu dữ liệu ở các cột Features
    required_cols = list(feature_cols)
    if target_col and target_col in new_df.columns:
        required_cols.append(target_col)

    clean_new_df = new_df.dropna(subset=required_cols).copy()

    if clean_new_df.empty:
        return None, None

    X_new = clean_new_df[feature_cols]
    preds = model.predict(X_new)
    clean_new_df["Predicted"] = preds

    test_metrics = None
    if target_col and target_col in clean_new_df.columns:
        y_true = clean_new_df[target_col]
        clean_new_df["Error_Diff"] = clean_new_df["Predicted"] - y_true

        r2 = r2_score(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = np.mean(np.abs(clean_new_df["Error_Diff"]))

        test_metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "n_samples": len(clean_new_df)
        }

    return clean_new_df, test_metrics


def create_regression_plot(df_results, feature_col, target_col):
    """
    Vẽ biểu đồ Hồi quy tuyến tính đơn (Actual vs Predicted / Trendline).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_results[feature_col],
        y=df_results[target_col],
        mode='markers',
        name='Thực tế (Actual)',
        marker=dict(color='#1f77b4', opacity=0.7)
    ))

    sorted_df = df_results.sort_values(by=feature_col)
    fig.add_trace(go.Scatter(
        x=sorted_df[feature_col],
        y=sorted_df["Predicted"],
        mode='lines',
        name='Đường Hồi quy (Fit Line)',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=f"Mô hình Hồi quy: {target_col} theo {feature_col}",
        xaxis_title=feature_col,
        yaxis_title=target_col,
        hovermode="closest",
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig


def create_actual_vs_predicted_chart(df_results, target_col, time_col=None):
    """
    Vẽ biểu đồ so sánh Thực tế (Actual) vs Dự báo (Predicted) cho file Test mới.
    """
    fig = go.Figure()

    x_axis = df_results[time_col] if (time_col and time_col in df_results.columns) else df_results.index

    # Đường giá trị Thực tế
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df_results[target_col],
        mode='lines+markers',
        name=f'Thực tế ({target_col})',
        line=dict(color='#2ca02c', width=2)
    ))

    # Đường giá trị Dự báo từ Model
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df_results["Predicted"],
        mode='lines+markers',
        name='Mô hình Dự báo (Predicted)',
        line=dict(color='#d62728', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"So sánh Giá trị Thực tế vs Mô hình Dự báo trên Dữ liệu Mới",
        xaxis_title="Thời gian / Mẫu" if time_col else "Chỉ số Mẫu",
        yaxis_title=target_col,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig