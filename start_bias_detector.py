import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

st.set_page_config(layout="wide")
st.header("Bias analysis - Loan dataset")

st.sidebar.header("Upload dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload datafile.")
    st.stop()


# Universal function to load data based on file type
# CSV separator detection works for comma AND semicolon
def load_data(file):
    filename = file.name.lower()

    try:
        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)

        elif filename.endswith(".csv"):
            df = pd.read_csv(file, sep=None, engine="python")

        else:
            st.error("Unsupported file type.")
            st.stop()

        # Clean column names to avoid KeyErrors
        df.columns = df.columns.astype(str).str.strip()
        df.columns = df.columns.str.replace(" ", "_", regex=False)
        df.columns = df.columns.str.replace("-", "_", regex=False)

        # Remove empty strings and empty rows properly
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df = df.replace(["None", "none", "NONE", "nan", "NaN", "NAN", ""], np.nan)

        # Remove completely empty rows and columns
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        # Reset index after cleaning
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        st.error("Failed to read file.")
        st.error(e)
        st.stop()


# Function to detect useless identifier columns
# These columns usually break or weaken the model
def detect_id_columns(df):
    id_cols = []

    for col in df.columns:
        col_lower = col.lower()

        if col_lower in ["id", "loan_id", "customer_id", "applicant_id"]:
            id_cols.append(col)

        elif col_lower.endswith("_id"):
            id_cols.append(col)

        elif df[col].nunique(dropna=True) == len(df) and len(df) > 20:
            id_cols.append(col)

    return list(dict.fromkeys(id_cols))


# Function to check if a column can be used as a binary target
def is_binary_target(series):
    cleaned = series.dropna().astype(str).str.strip()
    unique_values = cleaned.unique()

    return len(unique_values) == 2


# Function to convert target column to numeric 0 and 1
def convert_target_to_numeric(series):
    cleaned = series.astype(str).str.strip().str.lower()

    mapping = {
        "y": 1,
        "yes": 1,
        "approved": 1,
        "approve": 1,
        "accepted": 1,
        "accept": 1,
        "true": 1,
        "1": 1,

        "n": 0,
        "no": 0,
        "rejected": 0,
        "reject": 0,
        "declined": 0,
        "decline": 0,
        "false": 0,
        "0": 0
    }

    converted = cleaned.map(mapping)

    # If normal mapping fails, still allow any binary target
    if converted.isna().all():
        unique_values = cleaned.dropna().unique()

        if len(unique_values) == 2:
            converted = cleaned.map({
                unique_values[0]: 0,
                unique_values[1]: 1
            })

    return converted


# Function to detect numeric-looking text columns
# Example: "128", "360", "1.0" stored as object should become numeric
def try_convert_numeric_columns(df_model, feature_cols):
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_model[col]):
            converted = pd.to_numeric(df_model[col], errors="coerce")

            # Convert to numeric if most non-empty values are numeric
            original_not_missing = df_model[col].notna().sum()
            converted_not_missing = converted.notna().sum()

            if original_not_missing > 0:
                numeric_ratio = converted_not_missing / original_not_missing

                if numeric_ratio >= 0.80:
                    df_model[col] = converted

    return df_model


# Function to safely clean selected feature columns
def clean_feature_columns(df_model, feature_cols):
    df_model = try_convert_numeric_columns(df_model, feature_cols)

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_model[col]):
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
            df_model[col] = df_model[col].replace([np.inf, -np.inf], np.nan)

            if df_model[col].notna().sum() > 0:
                df_model[col] = df_model[col].fillna(df_model[col].median())
            else:
                df_model[col] = 0

        else:
            df_model[col] = df_model[col].astype(str).str.strip()
            df_model[col] = df_model[col].replace(["nan", "None", "NaN", ""], "Missing")
            df_model[col] = df_model[col].fillna("Missing")

    return df_model


# Function to clean the bias column for grouping
# Missing values are kept as "Missing" instead of being removed
def clean_bias_column(series):
    cleaned = series.copy()
    cleaned = cleaned.astype(str).str.strip()
    cleaned = cleaned.replace(["nan", "None", "NaN", "NAN", "", "<NA>"], "Missing")
    cleaned = cleaned.fillna("Missing")

    return cleaned


# Function to prepare model data
def prepare_model_data(df, y_col, x_cols, bias_col):
    df_model = df.copy()

    # Remove empty strings again just to be safe
    df_model = df_model.replace(r"^\s*$", np.nan, regex=True)
    df_model = df_model.replace(["None", "none", "NONE", "nan", "NaN", "NAN", ""], np.nan)

    # Remove rows where target is missing
    df_model = df_model.dropna(subset=[y_col])

    # Convert target to 0 and 1
    df_model[y_col] = convert_target_to_numeric(df_model[y_col])

    # Remove rows where target could not be converted
    df_model = df_model.dropna(subset=[y_col])

    # Keep only binary target values
    df_model = df_model[df_model[y_col].isin([0, 1])]

    if df_model.empty:
        return None, None, None, "No usable rows left after target conversion. Choose a real binary target, like Loan_Status."

    # Save original rows for bias analysis
    original_rows = df.loc[df_model.index].copy()

    # Clean bias column for plots
    # Important: null values are not removed, they become "Missing"
    original_rows[bias_col] = clean_bias_column(original_rows[bias_col])

    # Keep selected X columns only
    selected_cols = list(dict.fromkeys(x_cols))
    selected_cols = [col for col in selected_cols if col != y_col]

    if len(selected_cols) == 0:
        return None, None, None, "No feature columns selected."

    # Keep only required columns
    df_model = df_model[[y_col] + selected_cols].copy()

    # Clean selected feature columns
    df_model = clean_feature_columns(df_model, selected_cols)

    # Encode categorical columns
    categorical_cols = df_model[selected_cols].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Important:
    # drop_first=True prevents dummy-variable trap and Singular matrix errors
    df_encoded = pd.get_dummies(
        df_model,
        columns=categorical_cols,
        drop_first=True,
        dtype=float
    )

    # Create mapping from original feature names to encoded feature names
    feature_map = {}

    for col in selected_cols:
        if col in categorical_cols:
            encoded_cols = [
                new_col for new_col in df_encoded.columns
                if new_col.startswith(col + "_")
            ]

            # If a categorical column has only one usable category, no dummy is made
            # In that case it is not useful for the model
            feature_map[col] = encoded_cols

        else:
            feature_map[col] = [col]

    return df_encoded, original_rows, feature_map, None


# Function to expand selected original columns into encoded columns
def expand_features(selected_features, feature_map):
    expanded = []

    for feature in selected_features:
        if feature in feature_map:
            expanded.extend(feature_map[feature])

    expanded = list(dict.fromkeys(expanded))

    return expanded


# Function to remove duplicated and mathematically useless columns
def clean_model_matrix(X):
    # Remove columns with only one value
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] == 0:
        return X

    # Remove duplicate columns
    X = X.T.drop_duplicates().T

    if X.shape[1] == 0:
        return X

    # Remove columns that are perfectly correlated
    corr_matrix = X.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    columns_to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > 0.999999)
    ]

    X = X.drop(columns=columns_to_drop, errors="ignore")

    return X


# Function to safely fit logistic regression model
def fit_logit_model(df_encoded, y_col, encoded_x_cols):
    encoded_x_cols = [col for col in encoded_x_cols if col in df_encoded.columns]

    if len(encoded_x_cols) == 0:
        return None, None, "No usable feature columns left."

    model_data = df_encoded[[y_col] + encoded_x_cols].copy()

    # Convert everything to numeric
    for col in model_data.columns:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    # Replace infinite values
    model_data = model_data.replace([np.inf, -np.inf], np.nan)

    # Drop rows with missing values after conversion
    model_data = model_data.dropna()

    if model_data.empty:
        return None, None, "No usable rows left after cleaning."

    y = model_data[y_col].astype(float)
    X = model_data[encoded_x_cols].astype(float)

    # Target must contain both 0 and 1
    if y.nunique() != 2:
        return None, None, "Target must contain both 0 and 1 after cleaning."

    # Remove duplicate, constant and perfectly correlated columns
    X = clean_model_matrix(X)

    if X.shape[1] == 0:
        return None, None, "No usable feature columns left after cleaning the model matrix."

    # Add constant
    X = sm.add_constant(X, has_constant="add")

    try:
        # This is normal logistic regression, not linear regression
        model = sm.Logit(y, X).fit(disp=False, maxiter=300)
        predictions = pd.Series(model.predict(X), index=X.index)

        return model, predictions, None

    except Exception:
        try:
            # Fallback logistic regression with regularization
            # This handles singular matrix and separation issues better
            model = sm.Logit(y, X).fit_regularized(
                disp=False,
                maxiter=500,
                alpha=0.01
            )

            predictions = pd.Series(model.predict(X), index=X.index)

            return model, predictions, None

        except Exception:
            try:
                # Final fallback: Binomial GLM
                # This is still logistic-type binary modelling, not linear regression
                model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
                predictions = pd.Series(model.predict(X), index=X.index)

                return model, predictions, None

            except Exception as e:
                return None, None, str(e)


# Function to calculate approval rate per bias category
def calculate_bias_table(original_rows, predictions, bias_col):
    temp = original_rows.copy()
    temp = temp.loc[predictions.index].copy()

    temp[bias_col] = clean_bias_column(temp[bias_col])
    temp["prediction"] = predictions

    bias_table = (
        temp.groupby(bias_col, dropna=False)["prediction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    bias_table.columns = ["Category", "Predicted approval rate", "Prediction std", "Group size"]

    bias_table["Prediction std"] = bias_table["Prediction std"].fillna(0)

    return bias_table

# Function to add bias classification function
def classify_bias(difference_pp, threshold_pp):
    abs_diff = abs(difference_pp)

    if abs_diff <= threshold_pp:
        return "Not biased"
    elif abs_diff <= 2 * threshold_pp:
        return "Borderline"
    else:
        return "Biased"

def calculate_dynamic_threshold(std_1, n_1, std_2, n_2, min_floor_pp=3.0, z=1.96):
    """
    Computes a dynamic threshold in percentage points for the difference
    between two group mean predicted approval rates.

    std_1, std_2 are standard deviations of predictions (0 to 1 scale)
    n_1, n_2 are group sizes
    min_floor_pp is the minimum threshold in percentage points
    """
    if n_1 <= 0 or n_2 <= 0:
        return min_floor_pp

    se_diff = np.sqrt((std_1 ** 2) / n_1 + (std_2 ** 2) / n_2)

    # convert from probability scale to percentage points
    threshold_pp = z * se_diff * 100

    # enforce a minimum practical threshold
    return max(min_floor_pp, threshold_pp)

# Function to show note under charts
# Explains missing categories clearly
def show_missing_note(bias_col, categories):
    categories_as_text = [str(category) for category in categories]

    if "Missing" in categories_as_text:
        st.caption(
            f'"Missing" means the dataset contains null or blank values in the "{bias_col}" column. '
            f'These rows are kept in the analysis because we do not want to exclude applicants with missing demographic data.'
        )


# Function to show explanation under Model 1 and Model 2 charts
def explain_approval_chart(model_name, bias_col, categories):
    st.caption(
        f"This chart shows the average predicted approval rate from {model_name} for each group in the selected bias column: {bias_col}."
    )

    show_missing_note(bias_col, categories)


# Function to explain grouped chart
def explain_grouped_chart(bias_col, categories):
    st.caption(
        f"This chart compares Model 1 and Model 2 predicted approval rates for each {bias_col} group. "
        f"The values are percentages, so 70 means a predicted approval rate of 70%."
    )

    show_missing_note(bias_col, categories)


# Function to explain difference chart
def explain_difference_chart(bias_col, categories):
    st.caption(
        f"This chart shows Model 1 minus Model 2 for each {bias_col} group. "
        f"The unit is percentage points, written as pp. "
        f"For example, +2.00 pp means Model 1 is 2 percentage points higher than Model 2 for that group."
    )

    show_missing_note(bias_col, categories)


# Function to plot approval rate
# Fancy compact Plotly chart
def plot_approval_rate(bias_df, title, bias_col, model_name):
    plot_df = bias_df.copy()
    plot_df["Category"] = plot_df["Category"].astype(str)
    plot_df["Predicted approval rate percentage"] = plot_df["Predicted approval rate"] * 100

    fig = px.bar(
        plot_df,
        x="Category",
        y="Predicted approval rate percentage",
        text="Predicted approval rate percentage",
        title=title,
        template="plotly_dark"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_width=1,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Predicted approval rate: %{y:.2f}%"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        height=320,
        title_font_size=21,
        title_x=0.02,
        margin=dict(l=30, r=30, t=70, b=35),
        xaxis_title=bias_col,
        yaxis_title="Predicted approval rate (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)
    explain_approval_chart(model_name, bias_col, plot_df["Category"].tolist())


# Function to compare Model 1 and Model 2 approval rates
# This gives a cleaner grouped chart
def plot_model_comparison(bias_df_1, bias_df_2, bias_col):
    model_1 = bias_df_1.copy()
    model_2 = bias_df_2.copy()

    model_1["Model"] = "Model 1 - All features"
    model_2["Model"] = "Model 2 - Essential features"

    combined = pd.concat([model_1, model_2], ignore_index=True)
    combined["Category"] = combined["Category"].astype(str)
    combined["Predicted approval rate percentage"] = combined["Predicted approval rate"] * 100

    fig = px.bar(
        combined,
        x="Category",
        y="Predicted approval rate percentage",
        color="Model",
        barmode="group",
        text="Predicted approval rate percentage",
        title="Predicted approval rate by group",
        template="plotly_dark"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_width=1,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{fullData.name}<br>"
            "Predicted approval rate: %{y:.2f}%"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        height=370,
        title_font_size=21,
        title_x=0.02,
        margin=dict(l=30, r=30, t=70, b=35),
        xaxis_title=bias_col,
        yaxis_title="Predicted approval rate (%)",
        yaxis=dict(range=[0, 100]),
        legend_title_text="Model",
        bargap=0.25,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)
    explain_grouped_chart(bias_col, combined["Category"].unique().tolist())


# Function to plot difference
# Fancy compact Plotly chart
def plot_difference(comparison, bias_col):
    plot_df = comparison.copy()
    plot_df["Category"] = plot_df["Category"].astype(str)

    # Create rounded text so labels do not show long ugly decimals
    plot_df["Difference label"] = plot_df["Difference_percentage_points"].apply(
        lambda value: f"{value:+.2f} pp"
    )

    fig = px.bar(
        plot_df,
        x="Category",
        y="Difference_percentage_points",
        text="Difference label",
        title="Prediction shift: Model 1 compared to Model 2",
        template="plotly_dark"
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=1,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Difference: %{y:+.2f} percentage points"
            "<extra></extra>"
        )
    )

    fig.add_hline(
        y=0,
        line_width=2,
        line_dash="dash"
    )

    fig.update_layout(
        height=330,
        title_font_size=21,
        title_x=0.02,
        margin=dict(l=30, r=30, t=70, b=35),
        xaxis_title=bias_col,
        yaxis_title="Difference in percentage points",
        showlegend=False,
        bargap=0.35,
        uniformtext_minsize=10,
        uniformtext_mode="show",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)
    explain_difference_chart(bias_col, plot_df["Category"].tolist())


# Function to show clean metric cards
def show_bias_metrics(comparison):
    if comparison.empty:
        return

    max_shift = comparison["Difference_percentage_points"].abs().max()
    average_shift = comparison["Difference_percentage_points"].abs().mean()

    highest_group_row = comparison.loc[
        comparison["Difference_percentage_points"].abs().idxmax()
    ]

    highest_group = highest_group_row["Category"]
    highest_group_shift = highest_group_row["Difference_percentage_points"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Largest group shift",
        f"{max_shift:.2f} pp"
    )

    col2.metric(
        "Average group shift",
        f"{average_shift:.2f} pp"
    )

    col3.metric(
        "Most affected group",
        f"{highest_group}",
        f"{highest_group_shift:+.2f} pp"
    )


# Function to show clean dataframe
def show_table(df_to_show):
    st.dataframe(
        df_to_show,
        use_container_width=True,
        hide_index=True
    )


df = load_data(uploaded_file)

st.subheader("Dataset preview")
st.dataframe(df.head(), use_container_width=True)

st.sidebar.header("Model settings")

columns = df.columns.tolist()
id_columns = detect_id_columns(df)

usable_columns = [col for col in columns if col not in id_columns]

if len(usable_columns) == 0:
    st.error("No usable columns found.")
    st.stop()

# Prefer Loan_Status as target if it exists
default_target_index = 0

if "Loan_Status" in usable_columns:
    default_target_index = usable_columns.index("Loan_Status")

y_col = st.sidebar.selectbox(
    "Choose target column (y)",
    usable_columns,
    index=default_target_index
)

# Validate target before modelling
if not is_binary_target(df[y_col]):
    st.error("Target column must be binary.")
    st.write("Detected target values:")
    st.write(df[y_col].dropna().unique())
    st.stop()

possible_features = [col for col in usable_columns if col != y_col]

if len(possible_features) == 0:
    st.error("No possible feature columns found.")
    st.stop()

x_cols = st.sidebar.multiselect(
    "Choose ALL features (Model 1)",
    possible_features,
    default=possible_features
)

# Prefer a smaller reasonable default for Model 2
default_essential = [
    col for col in x_cols
    if col in [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]
]

x_essential = st.sidebar.multiselect(
    "Choose ESSENTIAL features (Model 2)",
    x_cols,
    default=default_essential
)

# Bias column is only used for grouping the results
# It tells the app which groups to compare
default_bias_index = 0

if "Gender" in possible_features:
    default_bias_index = possible_features.index("Gender")

bias_calculator = st.sidebar.selectbox(
    "Choose bias column",
    possible_features,
    index=default_bias_index
)


# Validation before models run
if len(x_cols) == 0:
    st.warning("Select X columns for Model 1.")
    st.stop()

if len(x_essential) == 0:
    st.warning("Select essential features for Model 2.")
    st.stop()

if y_col in x_cols:
    st.error("Target column cannot be in X.")
    st.stop()

if y_col in x_essential:
    st.error("Target column cannot be essential feature.")
    st.stop()

if bias_calculator == y_col:
    st.error("Target cannot be bias column.")
    st.stop()


st.info(
    "This app uses logistic regression with statsmodels Logit. "
    "The bias column is not the target. It is only used to compare predicted approval rates between groups."
)


# Show missing value information for the selected bias column
raw_missing_bias_count = df[bias_calculator].isna().sum()

if raw_missing_bias_count > 0:
    st.warning(
        f'The selected bias column "{bias_calculator}" contains {raw_missing_bias_count} missing values. '
        f'These rows are kept and shown as "Missing" in the charts.'
    )


# Prepare model data once
df_encoded, original_rows, feature_map, prepare_error = prepare_model_data(
    df=df,
    y_col=y_col,
    x_cols=x_cols,
    bias_col=bias_calculator
)

if prepare_error is not None:
    st.error(prepare_error)
    st.stop()

# Expand original selected columns into dummy encoded columns
x_cols_encoded = expand_features(x_cols, feature_map)
x_essential_encoded = expand_features(x_essential, feature_map)

# Remove missing encoded columns safely
x_cols_encoded = [col for col in x_cols_encoded if col in df_encoded.columns]
x_essential_encoded = [col for col in x_essential_encoded if col in df_encoded.columns]


# Model 1 : All features included
st.header("Model 1 - All features")

model_1, predictions_1, error_1 = fit_logit_model(
    df_encoded=df_encoded,
    y_col=y_col,
    encoded_x_cols=x_cols_encoded
)

if error_1 is not None:
    st.error("Model 1 failed.")
    st.error(error_1)
    st.stop()

bias_df_1 = calculate_bias_table(
    original_rows=original_rows,
    predictions=predictions_1,
    bias_col=bias_calculator
)

bias_df_1_show = bias_df_1.copy()
bias_df_1_show["Predicted approval rate"] = bias_df_1_show["Predicted approval rate"] * 100
bias_df_1_show["Predicted approval rate"] = bias_df_1_show["Predicted approval rate"].round(2)
bias_df_1_show = bias_df_1_show.rename(
    columns={"Predicted approval rate": "Predicted approval rate (%)"}
)

show_table(bias_df_1_show)


# Model 2 : Essential features only
st.header("Model 2 - Essential features only")

model_2, predictions_2, error_2 = fit_logit_model(
    df_encoded=df_encoded,
    y_col=y_col,
    encoded_x_cols=x_essential_encoded
)

if error_2 is not None:
    st.error("Model 2 failed.")
    st.error(error_2)
    st.stop()

bias_df_2 = calculate_bias_table(
    original_rows=original_rows,
    predictions=predictions_2,
    bias_col=bias_calculator
)

bias_df_2_show = bias_df_2.copy()
bias_df_2_show["Predicted approval rate"] = bias_df_2_show["Predicted approval rate"] * 100
bias_df_2_show["Predicted approval rate"] = bias_df_2_show["Predicted approval rate"].round(2)
bias_df_2_show = bias_df_2_show.rename(
    columns={"Predicted approval rate": "Predicted approval rate (%)"}
)

show_table(bias_df_2_show)


# Bias comparison between both models
st.header("Bias comparison")

comparison = pd.merge(
    bias_df_1,
    bias_df_2,
    on="Category",
    how="outer",
    suffixes=("_Model1", "_Model2")
)

comparison["Difference"] = (
    comparison["Predicted approval rate_Model1"] -
    comparison["Predicted approval rate_Model2"]
)

comparison["Difference_percentage_points"] = comparison["Difference"] * 100

comparison["Dynamic_threshold_pp"] = comparison.apply(
    lambda row: calculate_dynamic_threshold(
        std_1=row["Prediction std_Model1"],
        n_1=row["Group size_Model1"],
        std_2=row["Prediction std_Model2"],
        n_2=row["Group size_Model2"],
        min_floor_pp=3.0
    ),
    axis=1
)

comparison["Bias_flag"] = comparison.apply(
    lambda row: classify_bias(
        difference_pp=row["Difference_percentage_points"],
        threshold_pp=row["Dynamic_threshold_pp"]
    ),
    axis=1
)

comparison["Difference_percentage_points"] = comparison["Difference"] * 100

comparison_show = comparison.copy()

comparison_show["Predicted approval rate_Model1"] = comparison_show["Predicted approval rate_Model1"] * 100
comparison_show["Predicted approval rate_Model2"] = comparison_show["Predicted approval rate_Model2"] * 100

comparison_show["Predicted approval rate_Model1"] = comparison_show["Predicted approval rate_Model1"].round(2)
comparison_show["Predicted approval rate_Model2"] = comparison_show["Predicted approval rate_Model2"].round(2)
comparison_show["Difference_percentage_points"] = comparison_show["Difference_percentage_points"].round(2)
comparison_show["Dynamic_threshold_pp"] = comparison_show["Dynamic_threshold_pp"].round(2)

comparison_show = comparison_show.rename(
    columns={
        "Predicted approval rate_Model1": "Model 1 approval rate (%)",
        "Predicted approval rate_Model2": "Model 2 approval rate (%)",
        "Difference_percentage_points": "Difference (percentage points)",
        "Dynamic_threshold_pp": "Dynamic threshold (pp)",
        "Bias_flag": "Bias result"
    }
)

comparison_show = comparison_show[
    [
        "Category",
        "Model 1 approval rate (%)",
        "Model 2 approval rate (%)",
        "Difference (percentage points)",
        "Dynamic threshold (pp)",
        "Bias result"
    ]
]

show_bias_metrics(comparison)
st.subheader("Overall bias decision")

if "Biased" in comparison["Bias_flag"].values:
    overall_result = "BIASED"
    st.error(f"Overall result: {overall_result}")
elif "Borderline" in comparison["Bias_flag"].values:
    overall_result = "BORDERLINE"
    st.warning(f"Overall result: {overall_result}")
else:
    overall_result = "NOT BIASED"
    st.success(f"Overall result: {overall_result}")
show_table(comparison_show)

plot_model_comparison(bias_df_1, bias_df_2, bias_calculator)
plot_difference(comparison, bias_calculator)


# Clear explanation under the full section
st.subheader("What the comparison means")

st.write(
    "The grouped chart compares the predicted approval rate from Model 1 and Model 2 "
    "for each group in the selected bias column."
)

st.write(
    "The difference chart shows Model 1 predicted approval rate minus "
    "Model 2 predicted approval rate for each group."
)

st.write(
    "A positive value means Model 1 predicts a higher approval rate for that group. "
    "A negative value means Model 1 predicts a lower approval rate for that group."
)

st.write(
    "The difference values are shown in percentage points, written as pp. "
    "For example, +2.00 pp means Model 1 is 2 percentage points higher than Model 2 for that group."
)

st.write(
    'If a chart contains "Missing", that means the selected bias column contains null or blank values. '
    "Those rows are included because excluding them would remove applicants from the analysis."
)


# Optional individual charts
with st.expander("Show individual model charts"):
    plot_approval_rate(
        bias_df_1,
        "Bias per category - Model 1",
        bias_calculator,
        "Model 1"
    )

    plot_approval_rate(
        bias_df_2,
        "Bias per category - Model 2",
        bias_calculator,
        "Model 2"
    )


# Show model summaries
st.header("Model summaries")

with st.expander("Model 1 summary"):
    st.text(model_1.summary())

with st.expander("Model 2 summary"):
    st.text(model_2.summary())
