import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from pathlib import Path
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ClimateStat",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
            max-width: 1400px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            line-height: 1.6 !important;
            overflow: visible !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.8rem !important;
        }
        [data-testid="stSidebar"] {
            padding-top: 0.5rem;
        }
        [data-testid="stPlotlyChart"] {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. DASHBOARD TITLE
# ============================================================
st.title("🌦️ ClimateStat")
st.subheader("Regional Weather Prediction Engine")
st.write(
    "Analyze historical weather patterns using Probability and Statistics."
)
# Use India local time rather than the computer/server's default timezone.
now_india = datetime.now(ZoneInfo("Asia/Kolkata"))
today_india = now_india.date()
st.caption(f"📅 Today's date (India): **{today_india.strftime('%A, %d %B %Y')}**")

# Rerun the whole dashboard shortly after midnight India time so the displayed
# date and date-dependent API request are refreshed even if the tab stays open.
if st_autorefresh is not None:
    next_midnight = datetime.combine(
        today_india + timedelta(days=1), datetime.min.time(), tzinfo=ZoneInfo("Asia/Kolkata")
    )
    milliseconds_until_midnight = max(
        1_000, int((next_midnight - now_india).total_seconds() * 1000) + 5_000
    )
    refresh_count = st_autorefresh(
        interval=milliseconds_until_midnight,
        limit=None,
        key="climatestat_daily_midnight_refresh",
    )
    previous_refresh_count = st.session_state.get("climatestat_last_refresh_count", refresh_count)
    if refresh_count != previous_refresh_count:
        st.session_state["climatestat_last_refresh_count"] = refresh_count
        st.cache_data.clear()
        st.rerun()
    st.session_state["climatestat_last_refresh_count"] = refresh_count
else:
    st.info(
        "Automatic midnight refresh is unavailable because streamlit-autorefresh is not installed. "
        "Install the project requirements to enable it; you can still use the manual refresh button."
    )
st.divider()

# ============================================================
# 4. CITY SELECTION FOR THE ONLINE WEATHER API
# ============================================================
cities = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
}

city = st.sidebar.selectbox("Select a city for API analysis", list(cities.keys()))
latitude, longitude = cities[city]
st.sidebar.write(f"Selected region: {city}")
st.sidebar.caption(f"Today's date in India: {today_india.strftime('%d %b %Y')}")
if st.sidebar.button("🔄 Refresh weather data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("API period: 1 January 2024 through the latest available historical observation")
st.sidebar.info(
    "The dashboard includes online API analysis and separate analysis of your three CSV datasets."
)

# ============================================================
# 5. LOAD DOWNLOADED DATASETS
# ============================================================
DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_csv(filename):
    """Load a CSV file from the project's data folder."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{filename}'. Make sure it is inside: {DATA_DIR}"
        )
    return pd.read_csv(path)


@st.cache_data
def load_all_datasets():
    daily = load_csv("india_2000_2024_daily_weather.csv")
    annual = load_csv("Rainfall_Data_LL.csv")
    statewise = load_csv("rainfall_statewise_daily_imd.csv")

    if "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    if "YEAR" in annual.columns:
        annual["YEAR"] = pd.to_numeric(annual["YEAR"], errors="coerce")
    if "ANNUAL" in annual.columns:
        annual["ANNUAL"] = pd.to_numeric(annual["ANNUAL"], errors="coerce")

    if "Date" in statewise.columns:
        statewise["Date"] = pd.to_datetime(statewise["Date"], errors="coerce")

    for column in ["Daily Actual", "Daily Normal"]:
        if column in statewise.columns:
            statewise[column] = pd.to_numeric(
                statewise[column], errors="coerce"
            )

    return daily, annual, statewise


# ============================================================
# 6. FETCH HISTORICAL WEATHER DATA FROM OPEN-METEO
# ============================================================
@st.cache_data(ttl=3600)
def get_weather(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    # Archive data can lag by a few days. The actual prediction date below is
    # therefore based on the latest date returned by the API, not today's date.
    yesterday = datetime.now(ZoneInfo("Asia/Kolkata")).date() - timedelta(days=1)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": yesterday.isoformat(),
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "relative_humidity_2m_mean",
        ],
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    result = response.json()

    if "daily" not in result:
        raise ValueError("The weather API did not return daily data.")

    weather = result["daily"]
    df = pd.DataFrame(
        {
            "Date": weather["time"],
            "Mean Temperature": weather["temperature_2m_mean"],
            "Maximum Temperature": weather["temperature_2m_max"],
            "Minimum Temperature": weather["temperature_2m_min"],
            "Rainfall": weather["precipitation_sum"],
            "Wind Speed": weather["wind_speed_10m_max"],
            "Humidity": weather["relative_humidity_2m_mean"],
        }
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_columns = [
        "Mean Temperature",
        "Maximum Temperature",
        "Minimum Temperature",
        "Rainfall",
        "Wind Speed",
        "Humidity",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# ============================================================
# 7. API WEATHER ANALYSIS
# ============================================================
try:
    with st.spinner(f"Loading historical weather data for {city}..."):
        df = get_weather(latitude, longitude)

    df = df.dropna(subset=["Date", "Mean Temperature", "Rainfall"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        st.error("No usable weather records were returned by the API.")
        st.stop()

    st.success(f"Historical API weather data loaded for {city}.")

    temperature = df["Mean Temperature"]
    rainfall = df["Rainfall"]
    average_temp = temperature.mean()
    median_temp = temperature.median()
    min_temp = temperature.min()
    max_temp = temperature.max()
    std_temp = temperature.std()
    variance_temp = temperature.var()
    average_rain = rainfall.mean()
    total_rain = rainfall.sum()
    average_humidity = df["Humidity"].mean()
    total_days = len(df)
    rainy_days = int((rainfall > 0).sum())
    dry_days = total_days - rainy_days
    rain_probability = rainy_days / total_days * 100 if total_days else 0

    # --------------------------------------------------------
    # 7A. WEATHER OVERVIEW
    # --------------------------------------------------------
    st.header("Weather Overview")
    st.caption(f"Region: {city} | Usable daily records: {total_days}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Temperature", f"{average_temp:.2f} °C")
    col2.metric("Median Temperature", f"{median_temp:.2f} °C")
    col3.metric("Temperature Std. Deviation", f"{std_temp:.2f} °C")
    col4.metric("Historical Rainy-Day Frequency", f"{rain_probability:.2f}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Minimum Temperature", f"{min_temp:.2f} °C")
    col2.metric("Maximum Temperature", f"{max_temp:.2f} °C")
    col3.metric("Average Daily Rainfall", f"{average_rain:.2f} mm")
    col4.metric("Total Rainfall", f"{total_rain:.2f} mm")
    st.divider()

    # --------------------------------------------------------
    # 7B. TEMPERATURE TREND
    # --------------------------------------------------------
    st.header("Temperature Analysis")
    fig_temp = px.line(
        df,
        x="Date",
        y=["Mean Temperature", "Maximum Temperature", "Minimum Temperature"],
        title="Daily Temperature Trends",
        labels={
            "Date": "Date",
            "value": "Temperature (°C)",
            "variable": "Temperature Type",
        },
    )
    fig_temp.update_layout(
        hovermode="x unified",
        legend_title_text="Temperature Type",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # --------------------------------------------------------
    # 7C. DAILY AND MONTHLY RAINFALL
    # --------------------------------------------------------
    st.header("Rainfall Analysis")
    fig_rain = px.bar(
        df,
        x="Date",
        y="Rainfall",
        title="Daily Rainfall",
        labels={"Date": "Date", "Rainfall": "Rainfall (mm)"},
    )
    fig_rain.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_rain, use_container_width=True)

    monthly_rainfall = (
        df.set_index("Date")["Rainfall"].resample("MS").sum().reset_index()
    )
    monthly_rainfall["Month"] = monthly_rainfall["Date"].dt.strftime("%b %Y")
    fig_monthly = px.bar(
        monthly_rainfall,
        x="Month",
        y="Rainfall",
        title="Total Rainfall by Month",
        labels={"Month": "Month", "Rainfall": "Total Rainfall (mm)"},
    )
    fig_monthly.update_layout(
        xaxis_tickangle=-45, margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # --------------------------------------------------------
    # 7D. PROBABILITY AND STATISTICS
    # --------------------------------------------------------
    st.header("Probability and Statistics")
    col1, col2 = st.columns(2)
    col1.metric("Temperature Variance", f"{variance_temp:.2f} °C²")
    if pd.notna(average_humidity):
        col2.metric("Average Relative Humidity", f"{average_humidity:.2f}%")
    else:
        col2.metric("Average Relative Humidity", "Not available")

    st.subheader("Historical Probability of Rain")
    st.write(
        f"Rain was recorded on {rainy_days} days out of "
        f"{total_days} observed days."
    )
    st.latex(
        r"P(R)=\frac{\text{Number of rainy days}}{\text{Total observed days}}"
    )
    st.write(f"Estimated historical probability: {rain_probability:.2f}%")
    st.caption(
        f"Dry days in this period: {dry_days}. "
        "This is an observed historical frequency, not a guaranteed forecast."
    )

    # --------------------------------------------------------
    # 7E. TEMPERATURE-RAINFALL CORRELATION
    # --------------------------------------------------------
    st.subheader("Temperature–Rainfall Correlation")
    correlation_data = df[["Mean Temperature", "Rainfall"]].dropna()

    if (
        len(correlation_data) >= 2
        and correlation_data["Mean Temperature"].nunique() > 1
        and correlation_data["Rainfall"].nunique() > 1
    ):
        correlation = correlation_data["Mean Temperature"].corr(
            correlation_data["Rainfall"]
        )
        st.metric("Pearson Correlation Coefficient", f"{correlation:.3f}")
        if correlation > 0:
            interpretation = "The observed linear relationship is positive."
        elif correlation < 0:
            interpretation = "The observed linear relationship is negative."
        else:
            interpretation = "No linear relationship was measured."
        st.write(interpretation)
        st.caption(
            "Correlation measures linear association; it does not prove causation."
        )
    else:
        st.info("There is insufficient variation to calculate correlation.")

    # --------------------------------------------------------
    # 7F. CONDITIONAL PROBABILITY
    # --------------------------------------------------------
    st.header("Conditional Probability of Rain")
    st.write(
        "Explore the historical probability of rain on days when "
        "relative humidity is at least 70%."
    )
    humidity_df = df.dropna(subset=["Humidity", "Rainfall"])
    humid_days = humidity_df[humidity_df["Humidity"] >= 70]

    if not humid_days.empty:
        rainy_humid_days = humid_days[humid_days["Rainfall"] > 0]
        conditional_probability = len(rainy_humid_days) / len(humid_days) * 100

        col1, col2 = st.columns(2)
        col1.metric("Rain Probability on Humid Days", f"{conditional_probability:.2f}%")
        col2.metric("Number of Humid Days", str(len(humid_days)))
        st.write(
            f"Rain occurred on {len(rainy_humid_days)} of the "
            f"{len(humid_days)} humid days."
        )
        st.latex(r"P(R\mid H)=\frac{P(R\cap H)}{P(H)}")
        st.caption(
            "This is an observed historical probability, not a forecast of future rainfall."
        )
    else:
        st.info("No usable humidity observations are available for this calculation.")

    # --------------------------------------------------------
    # 7G. LOGISTIC REGRESSION PREDICTION DEMONSTRATION
    # --------------------------------------------------------
    st.divider()
    st.header("Weather Prediction Engine")
    st.write(
        "This demonstration estimates next-day rain probability using "
        "previous-day observations and Logistic Regression."
    )

    feature_columns = ["Mean Temperature", "Humidity", "Wind Speed", "Rainfall"]
    model_data = df[["Date"] + feature_columns].copy()
    # The target is rain on the row's date. Inputs come from the preceding
    # calendar day, so gaps in the archive are not mistakenly treated as one day.
    model_data["Rain Tomorrow"] = (df["Rainfall"] > 0).astype(int)
    model_data["Previous Observation Date"] = df["Date"].shift(1)
    for column in feature_columns:
        model_data[column] = df[column].shift(1)

    consecutive_day = (
        model_data["Date"] - model_data["Previous Observation Date"]
    ).dt.days.eq(1)
    model_data = model_data.loc[consecutive_day].dropna(
        subset=["Date"] + feature_columns + ["Rain Tomorrow"]
    ).copy()
    # Remove non-finite values as well as NaNs before fitting sklearn.
    finite_rows = np.isfinite(model_data[feature_columns].to_numpy(dtype=float)).all(axis=1)
    model_data = model_data.loc[finite_rows].copy()
    model_data["Rain Tomorrow"] = model_data["Rain Tomorrow"].astype(int)

    X = model_data[feature_columns]
    y = model_data["Rain Tomorrow"]

    if len(model_data) < 100 or y.nunique() < 2:
        st.warning(
            "There is insufficient data or only one rain class to train the model."
        )
    else:
        split_index = int(len(model_data) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        st.subheader("Model Evaluation Setup")
        st.write(
            f"Usable examples: **{len(model_data):,}** | "
            f"Training examples: **{len(X_train):,}** | "
            f"Testing examples: **{len(X_test):,}**"
        )
        st.caption(
            f"Training target dates: {model_data['Date'].iloc[0].date()} to "
            f"{model_data['Date'].iloc[split_index - 1].date()} | "
            f"Testing target dates: {model_data['Date'].iloc[split_index].date()} to "
            f"{model_data['Date'].iloc[-1].date()}"
        )
        st.write(
            f"Rainy-day share — training: **{y_train.mean():.1%}**, "
            f"testing: **{y_test.mean():.1%}**."
        )

        if y_train.nunique() < 2:
            st.warning(
                "The training period contains only one rain class, so Logistic Regression "
                "cannot be trained reliably. More varied historical data is required."
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_probability = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            # ROC-AUC is undefined if the test set contains only one class.
            auc = roc_auc_score(y_test, y_probability) if y_test.nunique() == 2 else np.nan

            st.subheader("Model Performance on Later Dates")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            col2.metric("Precision", f"{precision * 100:.2f}%")
            col3.metric("Recall", f"{recall * 100:.2f}%")
            col4.metric("F1 Score", f"{f1 * 100:.2f}%")

            col1, col2 = st.columns(2)
            col1.metric("Balanced Accuracy", f"{balanced_accuracy * 100:.2f}%")
            col2.metric("ROC-AUC", f"{auc:.3f}" if pd.notna(auc) else "Unavailable (one test class)")

            st.caption(
                "The split is chronological: older dates train the model and later dates test it. "
                "Class weighting helps address imbalance, but does not guarantee high performance."
            )

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: No Rain", "Actual: Rain"],
                columns=["Predicted: No Rain", "Predicted: Rain"],
            )
            st.subheader("Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)

            # Latest complete observation is used as the next-day input.
            valid_live_rows = df.dropna(subset=feature_columns).copy()
            valid_live_rows = valid_live_rows.loc[
                np.isfinite(valid_live_rows[feature_columns].to_numpy(dtype=float)).all(axis=1)
            ]

            st.divider()
            st.subheader("Next-Day Rain Probability Estimate")
            if valid_live_rows.empty:
                st.warning(
                    "A next-day estimate cannot be generated because the weather data "
                    "does not contain a complete, finite set of model features."
                )
            else:
                latest_row = valid_live_rows.sort_values("Date").iloc[[-1]]
                latest_features = latest_row[feature_columns]
                next_day_probability = model.predict_proba(latest_features)[0, 1]
                latest_observation_date = latest_row["Date"].iloc[0]
                prediction_date = latest_observation_date + pd.Timedelta(days=1)
                st.write(
                    f"Latest complete observation used: **{latest_observation_date.strftime('%d %b %Y')}**"
                )
                col1, col2 = st.columns(2)
                col1.metric("Estimated Rain Probability", f"{next_day_probability * 100:.2f}%")
                col2.metric("Prediction Date", prediction_date.strftime("%d %b %Y"))
                st.progress(float(next_day_probability))
                st.caption(
                    "This model-based probability uses the latest available complete API observation. "
                    "The archive may lag behind today, so the prediction date is the day after the "
                    "observation used. This is not an official operational weather forecast."
                )

    # 7H. STATISTICAL SUMMARY AND API RECORDS
    # --------------------------------------------------------
    st.header("Statistical Summary")
    summary_columns = [
        "Mean Temperature",
        "Maximum Temperature",
        "Minimum Temperature",
        "Rainfall",
        "Wind Speed",
        "Humidity",
    ]
    summary = df[summary_columns].describe().round(2).T
    summary.index.name = "Weather Variable"
    st.dataframe(summary, use_container_width=True)

    with st.expander("View Historical API Weather Records"):
        st.dataframe(df, use_container_width=True)

    st.subheader("Download API Weather Data")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download API Weather Data (CSV)",
        data=csv_data,
        file_name=f"{city.lower()}_weather.csv",
        mime="text/csv",
    )

except requests.RequestException as error:
    st.error(
        "Unable to retrieve weather data from Open-Meteo. "
        "Check your internet connection and try again."
    )
    st.caption(f"Technical details: {error}")
except (KeyError, ValueError, TypeError) as error:
    st.error("The API weather data could not be processed.")
    st.caption(f"Technical details: {error}")
except Exception as error:
    st.error("An unexpected error occurred while loading the API weather data.")
    st.caption(f"Technical details: {error}")

# ============================================================
# 8. DOWNLOADED DATASET ANALYSIS
# ============================================================
st.divider()
st.header("Downloaded Dataset Analysis")
st.write(
    "This section analyzes the three CSV files in the project's data folder. "
    "These datasets are kept separate from the Open-Meteo API data because "
    "their time periods and formats differ."
)

try:
    daily_data, annual_data, state_data = load_all_datasets()
    st.success("All three CSV datasets loaded successfully.")

    # Dataset overview
    overview = pd.DataFrame(
        {
            "Dataset": [
                "India Daily Weather",
                "Historical Rainfall",
                "State-wise IMD Rainfall",
            ],
            "Records": [len(daily_data), len(annual_data), len(state_data)],
            "Columns": [
                len(daily_data.columns),
                len(annual_data.columns),
                len(state_data.columns),
            ],
        }
    )
    st.subheader("Dataset Overview")
    st.dataframe(overview, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # 8A. INDIA DAILY WEATHER CSV
    # --------------------------------------------------------
    st.subheader("Historical Temperature and Rainfall by City")

    required_daily = {
        "city",
        "date",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "rain_sum",
    }
    if required_daily.issubset(daily_data.columns):
        available_cities = sorted(daily_data["city"].dropna().unique())
        selected_csv_city = st.selectbox(
            "Choose a city from the daily weather dataset",
            available_cities,
            key="csv_city",
        )
        city_df = daily_data[daily_data["city"] == selected_csv_city].copy()
        city_df["date"] = pd.to_datetime(city_df["date"], errors="coerce")
        for col in [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
        ]:
            city_df[col] = pd.to_numeric(city_df[col], errors="coerce")

        city_df = city_df.dropna(
            subset=["date", "temperature_2m_max", "rain_sum"]
        ).sort_values("date")

        if not city_df.empty:
            temp = city_df["temperature_2m_max"]
            rain = city_df["rain_sum"]
            total_observations = len(city_df)
            rainy_observations = int((rain > 0).sum())
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Maximum Temperature", f"{temp.mean():.2f} °C")
            col2.metric("Temperature Variance", f"{temp.var():.2f} °C²")
            col3.metric("Temperature Std. Deviation", f"{temp.std():.2f} °C")
            col4.metric(
                "Historical Rainy-Day Probability",
                f"{rainy_observations / total_observations * 100:.2f}%",
            )

            fig = px.line(
                city_df,
                x="date",
                y=["temperature_2m_max", "temperature_2m_min"],
                title=f"Daily Maximum and Minimum Temperature: {selected_csv_city}",
                labels={
                    "date": "Date",
                    "value": "Temperature (°C)",
                    "variable": "Temperature Type",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            monthly_city_rain = (
                city_df.set_index("date")["rain_sum"]
                .resample("MS")
                .sum()
                .reset_index()
            )
            fig = px.bar(
                monthly_city_rain,
                x="date",
                y="rain_sum",
                title=f"Monthly Rainfall: {selected_csv_city}",
                labels={"date": "Month", "rain_sum": "Rainfall (mm)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No usable records were found for the selected city.")
    else:
        st.warning(
            "The daily weather CSV does not contain all expected columns. "
            "Check the column names in the file."
        )

    # --------------------------------------------------------
    # 8B. HISTORICAL ANNUAL RAINFALL CSV
    # --------------------------------------------------------
    st.subheader("Long-Term Annual and Monthly Rainfall")

    required_annual = {"SUBDIVISION", "YEAR", "ANNUAL"}
    month_columns = [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    ]
    if required_annual.issubset(annual_data.columns):
        subdivisions = sorted(annual_data["SUBDIVISION"].dropna().unique())
        selected_subdivision = st.selectbox(
            "Choose a rainfall subdivision",
            subdivisions,
            key="rain_subdivision",
        )
        subdivision_df = annual_data[
            annual_data["SUBDIVISION"] == selected_subdivision
        ].copy()
        subdivision_df["YEAR"] = pd.to_numeric(
            subdivision_df["YEAR"], errors="coerce"
        )
        subdivision_df["ANNUAL"] = pd.to_numeric(
            subdivision_df["ANNUAL"], errors="coerce"
        )
        subdivision_df = subdivision_df.dropna(
            subset=["YEAR", "ANNUAL"]
        ).sort_values("YEAR")

        if not subdivision_df.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Average Annual Rainfall",
                f"{subdivision_df['ANNUAL'].mean():.2f} mm",
            )
            col2.metric(
                "Minimum Annual Rainfall",
                f"{subdivision_df['ANNUAL'].min():.2f} mm",
            )
            col3.metric(
                "Maximum Annual Rainfall",
                f"{subdivision_df['ANNUAL'].max():.2f} mm",
            )

            fig = px.line(
                subdivision_df,
                x="YEAR",
                y="ANNUAL",
                markers=True,
                title=f"Annual Rainfall: {selected_subdivision}",
                labels={"YEAR": "Year", "ANNUAL": "Rainfall (mm)"},
            )
            st.plotly_chart(fig, use_container_width=True)

            available_months = [
                col for col in month_columns if col in subdivision_df.columns
            ]
            if available_months:
                for col in available_months:
                    subdivision_df[col] = pd.to_numeric(
                        subdivision_df[col], errors="coerce"
                    )
                monthly_average = (
                    subdivision_df[available_months]
                    .mean()
                    .rename_axis("Month")
                    .reset_index(name="Average Rainfall")
                )
                fig = px.bar(
                    monthly_average,
                    x="Month",
                    y="Average Rainfall",
                    title=f"Average Monthly Rainfall: {selected_subdivision}",
                    labels={"Average Rainfall": "Rainfall (mm)"},
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usable annual rainfall records were found.")
    else:
        st.warning(
            "The historical rainfall CSV does not contain SUBDIVISION, YEAR, and ANNUAL."
        )

    # --------------------------------------------------------
    # 8C. STATE-WISE IMD DAILY RAINFALL CSV
    # --------------------------------------------------------
    st.subheader("State-wise Actual vs Normal Rainfall")

    required_state = {"State", "Date", "Daily Actual", "Daily Normal"}
    if required_state.issubset(state_data.columns):
        state_data["State"] = state_data["State"].astype(str).str.strip()
        available_states = sorted(state_data["State"].dropna().unique())
        selected_state = st.selectbox(
            "Choose a state or union territory",
            available_states,
            key="imd_state",
        )
        state_df = state_data[state_data["State"] == selected_state].copy()
        state_df["Date"] = pd.to_datetime(state_df["Date"], errors="coerce")
        state_df["Daily Actual"] = pd.to_numeric(
            state_df["Daily Actual"], errors="coerce"
        )
        state_df["Daily Normal"] = pd.to_numeric(
            state_df["Daily Normal"], errors="coerce"
        )
        state_df = state_df.dropna(
            subset=["Date", "Daily Actual", "Daily Normal"]
        ).sort_values("Date")

        if not state_df.empty:
            actual_mean = state_df["Daily Actual"].mean()
            normal_mean = state_df["Daily Normal"].mean()
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Daily Actual Rainfall", f"{actual_mean:.2f} mm")
            col2.metric("Mean Daily Normal Rainfall", f"{normal_mean:.2f} mm")
            difference = actual_mean - normal_mean
            col3.metric("Actual − Normal", f"{difference:.2f} mm")

            fig = px.line(
                state_df,
                x="Date",
                y=["Daily Actual", "Daily Normal"],
                title=f"Daily Actual vs Normal Rainfall: {selected_state}",
                labels={
                    "Date": "Date",
                    "value": "Rainfall (mm)",
                    "variable": "Rainfall Type",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Available records for this selection: "
                f"{state_df['Date'].min().date()} to "
                f"{state_df['Date'].max().date()}. "
                "This chart describes the dates in the downloaded file."
            )
        else:
            st.info("No usable rainfall records were found for this state.")
    else:
        st.warning(
            "The state-wise rainfall CSV does not contain all expected columns."
        )

except FileNotFoundError as error:
    st.error("A dataset file could not be found.")
    st.caption(str(error))
except (pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error("A CSV file could not be read. Check its format and encoding.")
    st.caption(str(error))
except Exception as error:
    st.error("The downloaded datasets could not be analyzed.")
    st.caption(f"Technical details: {error}")

# ============================================================
# 9. ABOUT THE PROJECT
# ============================================================
st.divider()
st.header("About ClimateStat")
st.write(
    """
    ClimateStat is a regional weather analysis dashboard developed as a
    Probability and Statistics project.

    **Statistical concepts used:**
    - Arithmetic mean and median
    - Variance and standard deviation
    - Empirical probability and conditional probability
    - Pearson correlation coefficient
    - Monthly and annual rainfall aggregation
    - Descriptive statistical analysis
    - Logistic Regression and model evaluation

    The dashboard analyzes historical weather observations to identify
    patterns and relationships. Historical probabilities and model estimates
    are not guaranteed forecasts.
    """
)
st.caption(
    "Sources: Open-Meteo Historical Weather API and the three user-downloaded CSV datasets. "
    "Temperature: °C | Rainfall: mm | Wind speed: km/h | Relative humidity: %"
)
