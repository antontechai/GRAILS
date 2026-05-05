import streamlit as st
import pandas as pd
from io import StringIO

st.set_page_config(
    page_title="Generic Fairness Tool",
    layout="wide"
)

# STYLING
st.markdown("""
<style>
    .title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
        max-width: 900px;
    }
    .soft-box {
        background: rgba(240, 242, 246, 0.45);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        background: rgba(240, 242, 246, 0.5);
    }
    .placeholder-chart {
        height: 280px;
        border-radius: 14px;
        background: rgba(240, 242, 246, 0.45);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #666;
        font-size: 1rem;
        text-align: center;
        padding: 1rem;
    }
    .explanation-card {
        background: rgba(240, 242, 246, 0.45);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# SESSION STATE
if "df" not in st.session_state:
    st.session_state.df = None

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False


# HELPERS
def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
        else:
            return None
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        return None


def file_format_valid(uploaded_file):
    if uploaded_file is None:
        return False
    return uploaded_file.name.lower().endswith((".csv", ".xlsx"))


def dataset_summary(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric": len(numeric_cols),
        "categorical": len(categorical_cols),
        "missing": int(df.isna().sum().sum())
    }


def check_data_cleanliness(df):
    issues = []

    if df is None or df.empty:
        issues.append("Dataset is empty.")
        return issues

    for col in df.columns:
        series = df[col].astype(str)

        # Unknown placeholder symbols
        suspicious_values = {"?", "??", "unknown", "Unknown", "UNKNOWN", "N/A", "n/a", "na", "NA", "-", "--"}
        if series.isin(suspicious_values).any():
            issues.append(f"Column '{col}' contains unknown placeholder values.")

        # Mixed types in object columns can hint inconsistency
        non_null = df[col].dropna()
        if len(non_null) > 0:
            detected_types = non_null.map(lambda x: type(x).__name__).nunique()
            if detected_types > 1:
                issues.append(f"Column '{col}' may contain inconsistent value formats.")

    return issues


# HEADER
st.markdown('<div class="title"> Generic Fairness Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'This tool helps users inspect datasets for potential bias by comparing model behavior '
    'with and without sensitive columns. It is designed as a fairness-focused dashboard that '
    'supports clearer, more responsible decision-making.'
    '</div>',
    unsafe_allow_html=True
)

# STEP 1: UPLOAD
st.subheader("1. Upload dataset")

uploaded_file = st.file_uploader(
    "Upload a .csv or .xlsx dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    if not file_format_valid(uploaded_file):
        st.error("Invalid file format. Please upload a .csv or .xlsx file.")
    else:
        df = read_uploaded_file(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success("Valid dataset uploaded successfully.")

# STEP 2: PREVIEW + CLEANLINESS
if st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("2. Dataset preview")

    summary = dataset_summary(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", summary["rows"])
    c2.metric("Columns", summary["columns"])
    c3.metric("Numeric", summary["numeric"])
    c4.metric("Categorical", summary["categorical"])
    c5.metric("Missing values", summary["missing"])

    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Show column details"):
        details = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_values": [int(df[col].isna().sum()) for col in df.columns],
            "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns]
        })
        st.dataframe(details, use_container_width=True)

    st.subheader("3. Data cleanliness check")

    cleanliness_issues = check_data_cleanliness(df)

    if len(cleanliness_issues) == 0:
        st.success("Dataset passed the basic cleanliness check.")
    else:
        st.error("Dataset may not be clean enough for analysis.")
        for issue in cleanliness_issues:
            st.markdown(f"- {issue}")

    # STEP 3: COLUMN SELECTION
    st.subheader("4. Select columns")

    all_columns = df.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox(
            "Select target / outcome column",
            options=all_columns
        )

        sensitive_cols = st.multiselect(
            "Select sensitive columns",
            options=[c for c in all_columns if c != target_col],
            help="Example: gender, race, religion, age"
        )

    with col2:
        feature_cols = st.multiselect(
            "Select feature columns",
            options=[c for c in all_columns if c != target_col],
            default=[c for c in all_columns if c != target_col]
        )

    valid = True

    if target_col in feature_cols:
        st.error("Target column cannot be included in feature columns.")
        valid = False

    if len(sensitive_cols) == 0:
        st.warning("Please select at least one sensitive column.")
        valid = False

    if len(cleanliness_issues) > 0:
        valid = False

    # STEP 4: RUN
    st.subheader("5. Run fairness analysis")

    run_clicked = st.button("Run Analysis", disabled=not valid)

    if run_clicked:
        st.session_state.analysis_started = True

# STEP 5: RESULTS PLACEHOLDER
if st.session_state.analysis_started:
    st.subheader("6. Results")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### With sensitive columns")
        st.markdown("**Loan approval likelihood percentage**")
        st.markdown(
            '<div class="placeholder-chart">Pie chart placeholder<br>Example: Men vs Women</div>',
            unsafe_allow_html=True
        )

    with chart_col2:
        st.markdown("#### Without sensitive columns")
        st.markdown("**Loan approval likelihood percentage**")
        st.markdown(
            '<div class="placeholder-chart">Pie chart placeholder<br>Example: Men vs Women</div>',
            unsafe_allow_html=True
        )

    st.markdown("### Bias Result")
    st.markdown(
        '<div class="result-box">BIASED / NOT BIASED</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Detailed Explanation")

    if st.session_state.df is not None:
        # show one explanation placeholder card per selected sensitive column
        try:
            for col in sensitive_cols:
                st.markdown(
                    f"""
                    <div class="explanation-card">
                        <strong>Sensitive column:</strong> {col}<br><br>
                        <strong>Impact:</strong> —<br>
                        <strong>Explanation:</strong> This impacts the bias score by X.<br>
                        <strong>Recommendation:</strong> You can prevent this by ...
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception:
            st.markdown(
                '<div class="explanation-card">Explanation placeholders will appear here for each selected sensitive column.</div>',
                unsafe_allow_html=True
            )

else:
    st.info("Upload a valid dataset, pass the cleanliness check, and select the required columns to continue.")