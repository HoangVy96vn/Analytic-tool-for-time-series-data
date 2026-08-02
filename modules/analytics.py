import pandas as pd
import numpy as np
import plotly.graph_objects as go


def calculate_cpk(data_series, usl, lsl):
    """
    Tính toán các chỉ số thống kê cơ bản và năng lực quy trình (Cp, CpK).
    """
    # Loại bỏ các giá trị null
    clean_series = data_series.dropna()

    if clean_series.empty:
        return None

    mean = float(clean_series.mean())
    std = float(clean_series.std(ddof=1))  # Sample standard deviation

    if std == 0 or np.isnan(std):
        return {
            "mean": mean,
            "std": 0.0,
            "cpu": np.nan,
            "cpl": np.nan,
            "cp": np.nan,
            "cpk": np.nan,
            "status": "Không thể tính (Độ lệch chuẩn bằng 0)"
        }

    # Tính Cp, CPU, CPL
    cp = (usl - lsl) / (6 * std) if (usl is not None and lsl is not None) else np.nan
    cpu = (usl - mean) / (3 * std) if usl is not None else np.nan
    cpl = (mean - lsl) / (3 * std) if lsl is not None else np.nan

    # CpK là giá trị nhỏ hơn giữa CPU và CPL
    cpk = min(filter(lambda x: not np.isnan(x), [cpu, cpl]), default=np.nan)

    # Đánh giá quy trình
    if np.isnan(cpk):
        status = "Chưa đủ dữ liệu giới hạn (LSL/USL)"
    elif cpk >= 1.33:
        status = "Đạt năng lực tốt (Capable)"
    elif cpk >= 1.0:
        status = "Mức cảnh báo (Marginal)"
    else:
        status = "Không đạt năng lực (Incapable)"

    return {
        "mean": mean,
        "std": std,
        "cpu": cpu,
        "cpl": cpl,
        "cp": cp,
        "cpk": cpk,
        "status": status
    }


def create_spc_control_chart(df, val_col, time_col=None, usl=None, lsl=None, sigma_level=3):
    """
    Vẽ biểu đồ kiểm soát SPC (Control Chart) với đường Mean, UCL, LCL, USL, LSL.
    """
    df_plot = df.copy()

    if val_col not in df_plot.columns:
        return go.Figure(), 0

    clean_series = df_plot[val_col].dropna()

    if clean_series.empty:
        return go.Figure(), 0

    mean_val = clean_series.mean()
    std_val = clean_series.std(ddof=1)

    # Tính toán giới hạn kiểm soát (Control Limits)
    ucl = mean_val + sigma_level * std_val
    lcl = mean_val - sigma_level * std_val

    # Xác định trục X
    x_axis = df_plot[time_col] if (time_col and time_col in df_plot.columns) else df_plot.index

    fig = go.Figure()

    # Đường dữ liệu chính
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df_plot[val_col],
        mode='lines+markers',
        name='Giá trị đo',
        line=dict(color='#1f77b4', width=1.5),
        marker=dict(size=5)
    ))

    # Phát hiện điểm Out-of-Control (vượt UCL/LCL)
    out_of_control = df_plot[(df_plot[val_col] > ucl) | (df_plot[val_col] < lcl)]
    out_count = len(out_of_control)

    if out_count > 0:
        x_out = out_of_control[time_col] if (time_col and time_col in out_of_control.columns) else out_of_control.index
        fig.add_trace(go.Scatter(
            x=x_out,
            y=out_of_control[val_col],
            mode='markers',
            name=f'Bất thường (>{sigma_level}σ)',
            marker=dict(color='red', size=9, symbol='x')
        ))

    # Đường Trung bình (Mean)
    fig.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text=f"Mean: {mean_val:.2f}")

    # Đường UCL / LCL (Control Limits)
    fig.add_hline(y=ucl, line_dash="dot", line_color="orange", annotation_text=f"UCL ({sigma_level}σ): {ucl:.2f}")
    fig.add_hline(y=lcl, line_dash="dot", line_color="orange", annotation_text=f"LCL ({sigma_level}σ): {lcl:.2f}")

    # Đường USL / LSL (Specification Limits) nếu người dùng nhập
    if usl is not None:
        fig.add_hline(y=usl, line_dash="solid", line_color="red", annotation_text=f"USL: {usl:.2f}")
    if lsl is not None:
        fig.add_hline(y=lsl, line_dash="solid", line_color="red", annotation_text=f"LSL: {lsl:.2f}")

    fig.update_layout(
        title=f"Biểu đồ Kiểm soát SPC - {val_col}",
        xaxis_title="Thời gian / Mẫu" if time_col else "Chỉ số Mẫu",
        yaxis_title=val_col,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig, out_count