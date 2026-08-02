import streamlit as st
import pandas as pd
import numpy as np

from modules.data_processor import load_data, auto_detect_columns
from modules.charts import (
    create_line_chart,
    create_moving_average_chart,
    create_boxplot_chart,
    create_histogram_chart,
    create_correlation_heatmap
)
from modules.analytics import calculate_cpk, create_spc_control_chart
from modules.ml_model import (
    train_linear_regression,
    create_regression_plot,
    evaluate_new_data,
    create_actual_vs_predicted_chart
)
from modules.pdf_generator import generate_pdf_report

# ---------------- CONFIG TRANG ----------------
st.set_page_config(
    page_title="Industrial Time-Series Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Industrial Time-Series & Quality Control Platform")

# ---------------- KHỞI TẠO SESSION STATE ----------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

# ---------------- SIDEBAR: UPLOAD & EXPORT ----------------
st.sidebar.header("📂 1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Chọn file Excel hoặc CSV", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if st.session_state.df_raw is None or st.sidebar.button("🔄 Re-load Data"):
        st.session_state.df_raw = load_data(uploaded_file)
        st.session_state.column_mapping = auto_detect_columns(st.session_state.df_raw)

# XUẤT BÁO CÁO PDF Ở SIDEBAR (Tối ưu nút tải 1-click)
if st.session_state.df_raw is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("📄 2. Export Report")

    # Lấy dữ liệu phân tích hiện tại từ Session State
    cpk_data_to_pdf = st.session_state.get("last_cpk_metrics", None)
    ml_data_to_pdf = st.session_state.get("last_ml_metrics", None)

    summary_df = None
    if "final_mapping" in st.session_state and "df_processed" in st.session_state:
        metrics = st.session_state.final_mapping.get("metrics", [])
        if metrics:
            summary_df = st.session_state.df_processed[metrics].describe().T[['mean', 'std', 'min', '50%', 'max']]
            summary_df = summary_df.reset_index().rename(columns={'index': 'Metric', '50%': 'Median'})

    # Tạo PDF Data Bytes
    pdf_bytes = generate_pdf_report(
        summary_stats=summary_df,
        cpk_metrics=cpk_data_to_pdf,
        ml_metrics=ml_data_to_pdf
    )

    # Nút tải trực tiếp 1-Click
    st.sidebar.download_button(
        label="📥 Tải Báo cáo PDF",
        data=pdf_bytes,
        file_name="Industrial_Analytics_Report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

# ---------------- HIỂN THỊ NỘI DUNG CHÍNH ----------------
if st.session_state.df_raw is not None:
    df = st.session_state.df_raw

    # Tạo các Tabs chức năng
    tab_data, tab_eda, tab_spc, tab_ml = st.tabs([
        "📊 Data Inspection & Mapping",
        "📈 Exploratory Analysis",
        "⚙️ SPC & CpK Analysis",
        "🤖 Linear Regression & Prediction"
    ])

    # ---------------- TAB 1: DATA INSPECTION & MAPPING ----------------
    with tab_data:
        st.subheader("Xem trước Dữ liệu Uploaded")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("⚙️ Tùy chỉnh & Xác nhận Phân loại Cột (Column Mapping)")

        col1, col2, col3 = st.columns(3)

        all_cols = list(df.columns)
        detected = st.session_state.column_mapping

        with col1:
            selected_time_col = st.selectbox(
                "📅 Cột Thời gian (Time / Datetime):",
                options=["None"] + all_cols,
                index=all_cols.index(detected["time"]) + 1 if detected.get("time") in all_cols else 0
            )

        with col2:
            selected_cat_cols = st.multiselect(
                "🏷️ Cột Phân loại (Machine, Product, Shift...):",
                options=all_cols,
                default=[c for c in detected.get("categories", []) if c in all_cols]
            )

        with col3:
            selected_num_cols = st.multiselect(
                "📏 Cột Giá trị đo (ValueA, ValueB, Temp...):",
                options=all_cols,
                default=[c for c in detected.get("metrics", []) if c in all_cols]
            )

        if st.button("✅ Xác nhận Mapping Dữ liệu", type="primary"):
            df_processed = df.copy()
            if selected_time_col != "None":
                df_processed[selected_time_col] = pd.to_datetime(df_processed[selected_time_col], errors='coerce')
                df_processed = df_processed.sort_values(by=selected_time_col)

            st.session_state.df_processed = df_processed
            st.session_state.final_mapping = {
                "time": selected_time_col if selected_time_col != "None" else None,
                "categories": selected_cat_cols,
                "metrics": selected_num_cols
            }
            st.success("Dữ liệu đã được chuẩn hóa thành công! Hãy chuyển sang các Tab tiếp theo để phân tích.")

    # ---------------- TAB 2: EXPLORATORY ANALYSIS ----------------
    with tab_eda:
        if "final_mapping" not in st.session_state or "df_processed" not in st.session_state:
            st.warning("⚠️ Vui lòng bấm nút '✅ Xác nhận Mapping Dữ liệu' ở Tab 1 trước khi sang bước này!")
        else:
            df_p = st.session_state.df_processed
            mapping = st.session_state.final_mapping

            time_col = mapping["time"]
            cat_cols = mapping["categories"]
            num_cols = mapping["metrics"]

            if not num_cols:
                st.error("Không có cột thông số đo (Value) nào được chọn để phân tích.")
            else:
                st.subheader("📌 Bộ lọc dữ liệu nhanh (Quick Filter)")

                df_filtered = df_p.copy()
                if cat_cols:
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        filter_cat = st.selectbox("Lọc theo cột phân loại:", options=["None"] + cat_cols)
                    with f_col2:
                        if filter_cat != "None":
                            unique_vals = list(df_p[filter_cat].dropna().unique())
                            selected_vals = st.multiselect(
                                f"Chọn giá trị thuộc {filter_cat}:",
                                options=unique_vals,
                                default=unique_vals
                            )
                            if selected_vals:
                                df_filtered = df_filtered[df_filtered[filter_cat].isin(selected_vals)]

                st.markdown("---")

                # --- SECTION 1: LINE CHART & MOVING AVERAGE ---
                st.subheader("1. Biến đổi theo thời gian (Time-Series Trends)")

                c1, c2 = st.columns([1, 2])
                with c1:
                    selected_metric = st.selectbox("Chọn thông số đo:", options=num_cols, key="eda_metric")
                    group_by_cat = st.selectbox(
                        "Phân màu nhóm (Color Group):",
                        options=["None"] + cat_cols,
                        key="eda_group"
                    )
                    group_var = group_by_cat if group_by_cat != "None" else None

                    ma_window = st.slider(
                        "Khung thời gian Moving Average (Window):",
                        min_value=2,
                        max_value=50,
                        value=7
                    )

                with c2:
                    if time_col:
                        st.plotly_chart(
                            create_line_chart(df_filtered, time_col, selected_metric, group_var),
                            use_container_width=True
                        )
                        st.plotly_chart(
                            create_moving_average_chart(df_filtered, time_col, selected_metric, window=ma_window),
                            use_container_width=True
                        )
                    else:
                        st.info("Cần chọn cột Thời gian ở Tab 1 để xem biểu đồ Time-series.")

                st.markdown("---")

                # --- SECTION 2: BOXPLOT & HISTOGRAM ---
                st.subheader("2. Phân bố Dữ liệu & Phân tán (Distribution Analysis)")
                d1, d2 = st.columns(2)

                with d1:
                    box_group_type = st.radio(
                        "Phân loại Boxplot theo:",
                        options=["Tháng (Month)", "Quý (Quarter)", "Năm (Year)", "Nhóm Categorical"],
                        horizontal=True
                    )

                    if "Tháng" in box_group_type:
                        fig_box = create_boxplot_chart(df_filtered, selected_metric, group_type="Month", time_col=time_col)
                    elif "Quý" in box_group_type:
                        fig_box = create_boxplot_chart(df_filtered, selected_metric, group_type="Quarter", time_col=time_col)
                    elif "Năm" in box_group_type:
                        fig_box = create_boxplot_chart(df_filtered, selected_metric, group_type="Year", time_col=time_col)
                    else:
                        selected_box_cat = st.selectbox("Chọn cột nhóm:", options=cat_cols) if cat_cols else None
                        if selected_box_cat:
                            fig_box = create_boxplot_chart(
                                df_filtered, selected_metric, group_type="Category", cat_col=selected_box_cat
                            )
                        else:
                            fig_box = None

                    if fig_box:
                        st.plotly_chart(fig_box, use_container_width=True)

                with d2:
                    nbins = st.slider("Số lượng cột Histogram (Bins):", min_value=10, max_value=100, value=30)
                    st.plotly_chart(
                        create_histogram_chart(df_filtered, selected_metric, n_bins=nbins),
                        use_container_width=True
                    )

                st.markdown("---")

                # --- SECTION 3: HEATMAP TƯƠNG QUAN ---
                st.subheader("3. Tương quan giữa các giá trị đo (Metrics Correlation)")
                if len(num_cols) >= 2:
                    selected_corr_cols = st.multiselect(
                        "Chọn các biến đưa vào Ma trận tương quan:",
                        options=num_cols,
                        default=num_cols
                    )
                    if len(selected_corr_cols) >= 2:
                        fig_heat = create_correlation_heatmap(df_filtered, selected_corr_cols)
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.warning("Cần chọn ít nhất 2 biến dạng số để tính tương quan.")
                else:
                    st.info("Dữ liệu cần có ít nhất 2 cột giá trị đo (Value) dạng số để vẽ Heatmap.")

    # ---------------- TAB 3: SPC & CPK ANALYSIS ----------------
    with tab_spc:
        if "final_mapping" not in st.session_state or "df_processed" not in st.session_state:
            st.warning("⚠️ Vui lòng bấm nút '✅ Xác nhận Mapping Dữ liệu' ở Tab 1 trước khi sang bước này!")
        else:
            df_p = st.session_state.df_processed
            mapping = st.session_state.final_mapping

            time_col = mapping["time"]
            cat_cols = mapping["categories"]
            num_cols = mapping["metrics"]

            if not num_cols:
                st.error("Không có cột thông số đo (Value) nào để thực hiện phân tích SPC/CpK.")
            else:
                st.subheader("⚙️ Cấu hình thông số Kiểm soát Chất lượng (Quality Control Settings)")

                c_cfg1, c_cfg2, c_cfg3 = st.columns([1, 1, 1])

                with c_cfg1:
                    spc_metric = st.selectbox("Chọn giá trị cần đánh giá SPC/CpK:", options=num_cols, key="spc_metric")

                data_series = df_p[spc_metric].dropna()
                data_mean = float(data_series.mean()) if not data_series.empty else 50.0
                data_std = float(data_series.std()) if not data_series.empty else 1.0

                with c_cfg2:
                    usl_val = st.number_input(
                        "Nhập USL (Upper Specification Limit):",
                        value=round(data_mean + 3 * data_std, 2),
                        step=0.1
                    )

                with c_cfg3:
                    lsl_val = st.number_input(
                        "Nhập LSL (Lower Specification Limit):",
                        value=round(data_mean - 3 * data_std, 2),
                        step=0.1
                    )

                st.markdown("---")

                # Bảng Thống kê & Tính toán CpK
                st.subheader("📊 Kết quả Năng lực Quy trình (Process Capability - CpK)")

                metrics_res = calculate_cpk(data_series, usl=usl_val, lsl=lsl_val)
                st.session_state.last_cpk_metrics = metrics_res

                if metrics_res:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("GTM (Mean - X̄)", f"{metrics_res['mean']:.3f}")
                    m2.metric("Độ lệch chuẩn (Std - σ)", f"{metrics_res['std']:.3f}")
                    m3.metric("Chỉ số Cp", f"{metrics_res['cp']:.3f}" if not np.isnan(metrics_res['cp']) else "N/A")
                    m4.metric("Chỉ số CpK", f"{metrics_res['cpk']:.3f}" if not np.isnan(metrics_res['cpk']) else "N/A")
                    m5.metric("Đánh giá Quy trình", metrics_res['status'])

                    if metrics_res['cpk'] < 1.0:
                        st.error("⚠️ Quy trình hiện tại **KHÔNG ĐẢM BẢO NĂNG LỰC** (CpK < 1.0). Tỷ lệ lỗi/phế phẩm có thể cao.")
                    elif metrics_res['cpk'] < 1.33:
                        st.warning("⚠️ Quy trình ở mức **CẢNH BÁO** (1.0 <= CpK < 1.33). Cần giám sát chặt chẽ.")
                    else:
                        st.success("✅ Quy trình **ĐẠT NĂNG LỰC TỐT** (CpK >= 1.33).")

                st.markdown("---")

                # Biểu đồ Kiểm soát SPC
                st.subheader("📈 Biểu đồ Kiểm soát SPC (Control Chart)")

                sigma_choice = st.slider("Cấu hình ngưỡng SPC Sigma Level:", min_value=1, max_value=6, value=3)

                fig_spc, out_count = create_spc_control_chart(
                    df_p,
                    val_col=spc_metric,
                    time_col=time_col,
                    usl=usl_val,
                    lsl=lsl_val,
                    sigma_level=sigma_choice
                )

                st.plotly_chart(fig_spc, use_container_width=True)

                if out_count > 0:
                    st.warning(f"🚨 Phát hiện **{out_count} điểm bất thường** vượt ngoài giới hạn kiểm soát {sigma_choice}σ (Out-of-Control)!")
                else:
                    st.info(f"✅ Tất cả các điểm dữ liệu đều nằm trong giới hạn kiểm soát {sigma_choice}σ.")

    # ---------------- TAB 4: LINEAR REGRESSION & PREDICTION ----------------
    with tab_ml:
        if "final_mapping" not in st.session_state or "df_processed" not in st.session_state:
            st.warning("⚠️ Vui lòng bấm nút '✅ Xác nhận Mapping Dữ liệu' ở Tab 1 trước khi sang bước này!")
        else:
            df_p = st.session_state.df_processed
            mapping = st.session_state.final_mapping
            num_cols = mapping["metrics"]

            if len(num_cols) < 2:
                st.error("Cần ít nhất 2 cột giá trị đo (Numeric Metrics) để xây dựng mô hình Hồi quy.")
            else:
                st.subheader("🤖 1. Huấn luyện Mô hình Hồi quy (Train Model)")

                c_ml1, c_ml2 = st.columns(2)

                with c_ml1:
                    target_var = st.selectbox(
                        "🎯 Chọn Biến Mục tiêu cần Dự báo (Target - Y):",
                        options=num_cols,
                        key="target_var"
                    )

                with c_ml2:
                    available_features = [c for c in num_cols if c != target_var]
                    feature_vars = st.multiselect(
                        "📌 Chọn Biến Đầu vào (Features - X):",
                        options=available_features,
                        default=[available_features[0]] if available_features else []
                    )

                if feature_vars:
                    model, metrics, df_res = train_linear_regression(df_p, feature_vars, target_var)
                    st.session_state.last_ml_metrics = metrics

                    if metrics:
                        # Hiển thị chỉ số mô hình huấn luyện
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Hệ số xác định (R² Score)", f"{metrics['r2']:.3f}")
                        m2.metric("Sai số RMSE (Train)", f"{metrics['rmse']:.3f}")
                        m3.metric("Số mẫu Huấn luyện", f"{metrics['n_samples']}")

                        coef_str = " + ".join([f"({val:.4f} * {col})" for col, val in metrics['coefficients'].items()])
                        eq_str = f"**{target_var}** = {metrics['intercept']:.4f} + {coef_str}"
                        st.info(f"📐 **Phương trình Hồi quy:** {eq_str}")

                        if len(feature_vars) == 1:
                            fig_reg = create_regression_plot(df_res, feature_vars[0], target_var)
                            st.plotly_chart(fig_reg, use_container_width=True)

                        st.markdown("---")

                        # SUB-SECTION A: DỰ BÁO NHANH (SIMULATOR)
                        with st.expander("🔮 Dự báo Nhanh theo Tham số tùy chỉnh (Single Prediction)", expanded=False):
                            st.write("Thử thay đổi thông số đầu vào để dự báo điểm:")
                            input_data = {}
                            pred_cols = st.columns(len(feature_vars))
                            for idx, col in enumerate(feature_vars):
                                default_val = float(df_p[col].mean())
                                with pred_cols[idx]:
                                    input_data[col] = st.number_input(
                                        f"Nhập {col}:",
                                        value=round(default_val, 2),
                                        key=f"input_{col}"
                                    )

                            input_df = pd.DataFrame([input_data])
                            pred_val = model.predict(input_df)[0]
                            st.success(f"💡 **Giá trị dự báo của {target_var}:** `{pred_val:.3f}`")

                        st.markdown("---")

                        # SUB-SECTION B: UPLOAD FILE MỚI ĐỂ DỰ BÁO VÀ TEST DỮ LIỆU THỰC TẾ
                        st.subheader("📂 2. Dự báo & Kiểm định Mô hình trên File Dữ liệu Mới")
                        st.write("Upload file CSV/Excel mới chứa các cột Features để xuất kết quả dự báo hàng loạt, hoặc so sánh với giá trị thực tế.")

                        test_file = st.file_uploader(
                            "Chọn file dữ liệu Mới (CSV/Excel)",
                            type=["csv", "xlsx", "xls"],
                            key="test_file_uploader"
                        )

                        if test_file is not None:
                            # Đọc file dữ liệu mới
                            test_df = load_data(test_file)

                            # Kiểm tra xem file mới có đủ các cột Features không
                            missing_features = [col for col in feature_vars if col not in test_df.columns]

                            if missing_features:
                                st.error(f"❌ File mới thiếu các cột đầu vào bắt buộc: {missing_features}")
                            else:
                                # Tiến hành dự báo trên dữ liệu mới
                                test_res_df, test_metrics = evaluate_new_data(
                                    model, test_df, feature_vars, target_var
                                )

                                if test_res_df is not None:
                                    st.success(f"✅ Đã chạy dự báo thành công cho {len(test_res_df)} dòng dữ liệu!")

                                    # TH1: Trong file mới CÓ cột Target thực tế -> Đánh giá độ lệch
                                    if test_metrics:
                                        st.subheader("📊 Đánh giá Độ chênh lệch giữa Dữ liệu Thực tế & Mô hình")
                                        tm1, tm2, tm3 = st.columns(3)
                                        tm1.metric("R² Score (Tập Test)", f"{test_metrics['r2']:.3f}")
                                        tm2.metric("Sai số RMSE (Test)", f"{test_metrics['rmse']:.3f}")
                                        tm3.metric("Sai số tuyệt đối trung bình (MAE)", f"{test_metrics['mae']:.3f}")

                                        # Vẽ đồ thị so sánh Actual vs Predicted
                                        time_col_test = mapping["time"] if (mapping["time"] and mapping["time"] in test_res_df.columns) else None
                                        fig_test = create_actual_vs_predicted_chart(
                                            test_res_df, target_var, time_col=time_col_test
                                        )
                                        st.plotly_chart(fig_test, use_container_width=True)

                                    # TH2: Hiển thị bảng kết quả & Nút Tải file Kết quả Dự báo
                                    st.subheader("📋 Bảng Kết quả Dự báo Hàng loạt")
                                    st.dataframe(test_res_df.head(20), use_container_width=True)

                                    # Nút Tải file CSV kết quả về máy
                                    csv_data = test_res_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 Tải về File Kết quả Dự báo (CSV)",
                                        data=csv_data,
                                        file_name="predicted_results.csv",
                                        mime="text/csv",
                                        type="primary"
                                    )

                    else:
                        st.error("Dữ liệu không đủ hoặc có quá nhiều ô trống (NaN) để huấn luyện mô hình.")
                else:
                    st.warning("Vui lòng chọn ít nhất 1 biến đầu vào (Feature X).")
else:
    st.info("👋 Vui lòng upload file CSV hoặc Excel ở thanh bên trái để bắt đầu.")