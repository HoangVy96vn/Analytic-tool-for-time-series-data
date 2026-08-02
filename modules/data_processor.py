import pandas as pd
import numpy as np


def load_data(uploaded_file) -> pd.DataFrame:
    """Đọc file CSV hoặc Excel tải lên từ Streamlit."""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df


def auto_detect_columns(df: pd.DataFrame) -> dict:
    """Tự động phân loại các cột trong dataframe."""
    time_cols = []
    cat_cols = []
    num_cols = []

    for col in df.columns:
        # 1. Kiểm tra Datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_cols.append(col)
            continue

        # Thử ép kiểu thời gian nếu tên cột chứa từ khóa thời gian
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'day', 'timestamp']):
            try:
                pd.to_datetime(df[col].dropna().iloc[:100])
                time_cols.append(col)
                continue
            except Exception:
                pass

        # 2. Kiểm tra Categorical vs Numerical
        if pd.api.types.is_numeric_dtype(df[col]):
            # Nếu là số nhưng số lượng giá trị duy nhất quá ít -> Coi là danh mục (Machine ID, Shift, v.v.)
            if df[col].nunique() < 10 and not pd.api.types.is_float_dtype(df[col]):
                cat_cols.append(col)
            else:
                num_cols.append(col)
        else:
            cat_cols.append(col)

    return {
        "time": time_cols[0] if time_cols else None,
        "categories": cat_cols,
        "metrics": num_cols
    }