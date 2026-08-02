import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_line_chart(df: pd.DataFrame, time_col: str, val_col: str, group_col: str = None):
    """Vẽ biểu đồ đường thể hiện biến đổi theo thời gian."""
    fig = px.line(
        df,
        x=time_col,
        y=val_col,
        color=group_col,
        title=f"Biểu đồ đường: {val_col} theo {time_col}" + (f" (Phân loại: {group_col})" if group_col else ""),
        markers=False
    )
    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


def create_moving_average_chart(df: pd.DataFrame, time_col: str, val_col: str, window: int = 7):
    """Vẽ biểu đồ Moving Average (Trung bình động) so với dữ liệu gốc."""
    df_sorted = df.sort_values(by=time_col).copy()
    df_sorted[f'MA_{window}'] = df_sorted[val_col].rolling(window=window).mean()

    fig = go.Figure()
    # Đường giá trị thực tế
    fig.add_trace(go.Scatter(
        x=df_sorted[time_col], y=df_sorted[val_col],
        mode='lines', name='Giá trị thực', opacity=0.4, line=dict(color='gray')
    ))
    # Đường trung bình động
    fig.add_trace(go.Scatter(
        x=df_sorted[time_col], y=df_sorted[f'MA_{window}'],
        mode='lines', name=f'Moving Avg (Window={window})', line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title=f"Biểu đồ Moving Average ({window} chu kỳ) cho {val_col}",
        xaxis_title=time_col, yaxis_title=val_col,
        hovermode="x unified", template="plotly_white"
    )
    return fig


def create_boxplot_chart(df: pd.DataFrame, val_col: str, group_type: str = "Month", time_col: str = None,
                         cat_col: str = None):
    """Vẽ Boxplot phân tích độ phân tán theo Chu kỳ thời gian hoặc Categorical."""
    df_plot = df.copy()

    if group_type in ["Month", "Quarter", "Year"] and time_col:
        if group_type == "Month":
            df_plot["Period"] = df_plot[time_col].dt.to_period("M").astype(str)
        elif group_type == "Quarter":
            df_plot["Period"] = df_plot[time_col].dt.to_period("Q").astype(str)
        elif group_type == "Year":
            df_plot["Period"] = df_plot[time_col].dt.year.astype(str)
        x_axis = "Period"
        title_str = f"Boxplot {val_col} theo {group_type}"
    else:
        x_axis = cat_col
        title_str = f"Boxplot {val_col} theo nhóm {cat_col}"

    fig = px.box(
        df_plot,
        x=x_axis,
        y=val_col,
        points="outliers",
        title=title_str,
        color=x_axis
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_histogram_chart(df: pd.DataFrame, val_col: str, n_bins: int = 30):
    """Vẽ biểu đồ phân bố tần suất Histogram."""
    fig = px.histogram(
        df,
        x=val_col,
        nbins=n_bins,
        marginal="rug",  # Thêm mật độ vạch ở trục dưới
        title=f"Phân bố tần suất (Histogram) của {val_col}",
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(template="plotly_white", yaxis_title="Tần suất (Count)")
    return fig


def create_correlation_heatmap(df: pd.DataFrame, num_cols: list):
    """Vẽ Heatmap ma trận tương quan giữa các thông số đo dạng số."""
    if len(num_cols) < 2:
        return None

    corr_matrix = df[num_cols].corr().round(2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Ma trận tương quan Heatmap giữa các biến đo"
    )
    fig.update_layout(template="plotly_white")
    return fig