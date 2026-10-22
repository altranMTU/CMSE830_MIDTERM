"""Streamlit dashboard for exploratory Formula 1 pit-stop strategy analysis.

The app fetches data from OpenF1, uses helper functions from data_pipeline.py
for cleaning and comparison, and displays interactive exploratory charts.

Run with: streamlit run app.py
Note: data_pipeline.py must be in the same project folder (or Python import path).
"""
# Matplotlib creates and controls the figures shown inside Streamlit.
# Pandas handles tables; Requests contacts OpenF1; Seaborn draws statistical plots.
# Streamlit creates the web dashboard and interactive controls.
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

# These project-specific functions live in data_pipeline.py:
# prepare_laps: joins lap records to tire stints and flags pit-affected laps.
# clean_pace_laps: excludes laps unsuitable for normal racing-pace analysis.
# compare_pit_windows: compares clean laps before and after recorded pit visits.
from data_pipeline import clean_pace_laps, compare_pit_windows, prepare_laps

# Configure the browser title and use the available page width for charts.
st.set_page_config(page_title="F1 Pit Strategy Explorer", layout="wide")
# Use a white grid so differences in lap times are easier to see.
sns.set_theme(style="whitegrid")
st.title("Formula 1: Pit Stops, Tire Age & Lap Times")
st.write(
    "**Research question:** How do tire age, tire compound, and the timing of pit stops relate "
    "to lap-time performance?"
)

# Dropdown options: display text is mapped to a year and circuit search term.
RACES = {
    "2024 Spanish Grand Prix (Catalunya)": (2024, "Catalunya"),
    "2024 Italian Grand Prix (Monza)": (2024, "Monza"),
}


# Store API results for 24 hours (86,400 seconds) to avoid downloading
# the same information every time a user changes a filter or reruns the app.
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def openf1(endpoint, params):
    """Request one OpenF1 endpoint and return the JSON records as a DataFrame."""
    # Example: endpoint="laps", params={"session_key": 12345}.
    # A 90-second timeout prevents a request from hanging indefinitely.
    response = requests.get(f"https://api.openf1.org/v1/{endpoint}", params=params, timeout=90)
    # Raise an exception for HTTP errors 
    response.raise_for_status()
    result = response.json()
    # OpenF1 normally returns a list of records. Reject unexpected formats.
    if not isinstance(result, list):
        raise ValueError(f"Unexpected API response for {endpoint}")
    return pd.DataFrame(result)


# Reuse already-prepared race data until the cache expires or inputs change.
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_race(year, circuit):
    """Find a race session, download its tables, and prepare lap/tire records."""
    # Query race sessions for the chosen year (ignore practice/qualifying).
    sessions = openf1("sessions", {"year": year, "session_name": "Race"})
    if sessions.empty or "circuit_short_name" not in sessions:
        raise ValueError("No race sessions or circuit names returned by OpenF1")
    # Keep only sessions whose circuit name contains the selected search term.
    # case=False means capitalization does not matter; na=False skips missing names.
    match = sessions[sessions["circuit_short_name"].astype(str).str.contains(circuit, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Could not find {year} {circuit} race session")
    # Every OpenF1 session has a unique session_key used to retrieve its data.
    session_key = int(match.iloc[0]["session_key"])
    # Download lap timing, tire stint, and pit-visit records for this race.
    laps = openf1("laps", {"session_key": session_key})
    stints = openf1("stints", {"session_key": session_key})
    pits = openf1("pit", {"session_key": session_key})
    # The helper merges laps with the correct tire stint, prepares tire ages,
    # and returns a data-quality summary (its implementation is in data_pipeline.py).
    cleaned, pits, quality = prepare_laps(laps, stints, pits)
    return cleaned, pits, quality, session_key


# The sidebar controls which race and cleaning settings are used.
with st.sidebar:
    st.header("Explore a race")
    race_label = st.selectbox("Grand Prix", list(RACES), index=0)
    # Remove unusually slow laps in clean-pace charts. For example, 1.15
    # excludes a lap that is more than 15% slower than that driver's median.
    upper_ratio = st.slider("Slow-lap filtering threshold (× driver's median)",
                            1.05, 1.50, 1.15, 0.05)
# Loading can fail due to internet errors or unmatched session names.
# Catch the error and show a helpful message instead of a full crash.
try:
    with st.spinner("Downloading OpenF1 race data and preparing tire stints..."):
        laps, pits, quality, key = load_race(*RACES[race_label])
except Exception as exc:
    st.error(f"Unable to load OpenF1 historical race data: {exc}")
    st.info("Check internet access and OpenF1 availability, then reload the app. The project README explains the data sources.")
    st.stop()

# Create filters using compounds and drivers actually found in the race data.
with st.sidebar:
    compounds = sorted(laps["compound"].dropna().astype(str).unique())
    drivers = sorted(laps["driver_number"].dropna().astype(int).unique())
    compound_selection = st.multiselect("Tire compounds", compounds, default=compounds)
    driver_selection = st.multiselect("Drivers", drivers, default=drivers)

# All graphs require at least one driver and one compound.
if not compound_selection or not driver_selection:
    st.warning("Select at least one tire compound and driver to see visualizations.")
    st.stop()

# First remove pit-affected/abnormally slow laps for normal-pace analysis.
# Keep laps and pits separately because the pit timeline and stop analyses
# should still show the events that are excluded from clean pace statistics.
clean_all = clean_pace_laps(laps, upper_ratio)
# Apply the user's selections to the cleaned laps using boolean masks.
clean = clean_all[clean_all["compound"].isin(compound_selection) & clean_all["driver_number"].isin(driver_selection)]
# Keep pit visits only for the selected drivers; use an empty DataFrame if
# no driver_number column is available in the downloaded pit data.
selected_pits = pits[pits["driver_number"].isin(driver_selection)] if "driver_number" in pits else pd.DataFrame()
# Display four headline numbers above the visualizations.
metrics = st.columns(4)
metrics[0].metric("Total race laps in data", len(laps))
metrics[1].metric("Clean laps (selection)", len(clean))
metrics[2].metric("Recorded pit visits (selection)", len(selected_pits))
metrics[3].metric("Session key", key)

# Avoid drawing graphs with no rows after filtering.
if clean.empty:
    st.warning("The current filters yield no usable pace observations. Select more drivers or compounds.")
    st.stop()

# Separate the visualizations into four topics so the dashboard is readable.
t1, t2, t3, t4 = st.tabs(["Tire degradation", "Tire compounds", "Driver race pace", "Pit stop effects"])

# Tab 1 — Question: Do older tires tend to be associated with slower laps?
with t1:
    st.subheader("Tire age and lap time")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Each dot is one clean driver-lap; color indicates tire compound.
    # alpha=0.45 makes overlapping observations easier to see.
    sns.scatterplot(data=clean, x="tyre_age", y="lap_duration", hue="compound", alpha=0.45, ax=ax)
    ax.set(xlabel="Estimated tire age (laps)", ylabel="Lap time (seconds)")
    # Render in Streamlit, then close the figure to free memory.
    st.pyplot(fig); plt.close(fig)
    st.caption("Question: Do older tires coincide with slower laps? Fuel burn and race conditions can produce an opposing trend.")
    # Combine each compound/tire-age combination:
    # median_lap represents typical observed pace; count is its sample size.
    age = clean.groupby(["compound", "tyre_age"], as_index=False).agg(
        median_lap=("lap_duration", "median"), count=("lap_duration", "size"))
    # Put pace trends and their supporting sample sizes side by side.

# Tab 2 — Question: How do observed lap times differ between compounds?
with t2:
    st.subheader("Lap-time distribution by tire compound")
    fig, ax = plt.subplots(figsize=(10, 5))
    # Boxes summarize the middle 50% of laps, the median, and potential outliers.
    sns.boxplot(data=clean, x="compound", y="lap_duration", ax=ax)
    ax.set(xlabel="Tire compound", ylabel="Lap time (seconds)")
    st.pyplot(fig); plt.close(fig)
    # Create a numerical table to support what is visible in the box plot.
    # std measures how much lap times vary around their average.
    summary = clean.groupby("compound")["lap_duration"].agg(["count", "mean", "median", "std"]).round(2)
    st.dataframe(summary, use_container_width=True)
    # Find the compound with the smallest observed median time.
    quickest = summary.loc[summary["median"].idxmin()]
    st.write(f"**Lowest observed median among selected compounds:** {summary['median'].idxmin()} "
             f"at {quickest['median']:.2f} seconds across {int(quickest['count'])} laps.")

# Tab 3 — Question: How does a driver's pace change around pit visits?
with t3:
    driver = st.selectbox("Driver for lap-by-lap timeline", driver_selection)
    # Use all recorded laps, not just clean ones, to keep pit-stop spikes visible.
    driver_laps = laps[laps["driver_number"] == driver].sort_values("lap_number")
    fig, ax = plt.subplots(figsize=(11, 5))
    # Plot each lap's duration by lap number, colored by tire compound.
    sns.scatterplot(data=driver_laps, x="lap_number", y="lap_duration", hue="compound", ax=ax)
    # Overlay a dashed vertical line at each recorded pit-visit lap.
    # The next lap may also be slow (pit-out effects depend on timing boundaries).
    if "lap_number" in selected_pits:
        for pit_lap in selected_pits.loc[selected_pits["driver_number"] == driver, "lap_number"].dropna():
            # axvline(x) draws a vertical marker at the given race-lap number.
            ax.axvline(pit_lap, color="black", linestyle="--", alpha=0.4)
    ax.set(title=f"Driver {driver}: lap times and pit visits", xlabel="Race lap", ylabel="Lap time (seconds)")
    st.pyplot(fig); plt.close(fig)
    st.caption("Dashed lines are recorded pit visits. Large spikes may be due to pits, traffic or race interruptions.")

# Tab 4 — Questions: How much time is spent in the pits, and how does
# the driver's observed pace differ before versus after stopping?
with t4:
    st.subheader("Pit-lane time distribution")
    # Use the first available, nonempty duration column in preference order.
    duration_field = next((c for c in ("lane_duration", "pit_duration", "stop_duration")
                           if c in selected_pits and selected_pits[c].notna().any()), None)
    if duration_field:
        # Exclude missing, zero, and negative durations before plotting.
        duration = selected_pits.loc[selected_pits[duration_field] > 0, duration_field]
        if not duration.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            # Histogram: x-axis is duration; bar height is the number of pit visits.
            sns.histplot(duration, bins=min(16, max(4, len(duration) // 2)), ax=ax)
            ax.set(xlabel=f"{duration_field} (seconds)", ylabel="Pit visits")
            st.pyplot(fig); plt.close(fig)
            st.write(f"**Median recorded {duration_field}:** {duration.median():.2f} seconds (n={len(duration)}).")
        else:
            st.info("There are no positive durations in the selected pit records.")
    else:
        st.info("OpenF1 did not supply pit durations for this selection.")
    st.caption("Time spent in the pit lane is not the *net* time lost relative to a normal racing lap.")
    st.subheader("Lap times before vs. after recorded pits")
    # The helper compares the median of nearby clean laps on each side of
    # a stop and returns a "Change (s)" column = AFTER minus BEFORE.
    compare = compare_pit_windows(clean_all, selected_pits, driver_selection)
    if compare.empty:
        st.info("Not enough clean laps around pit events under the current filtering threshold.")
    else:
        st.dataframe(compare.round(2), hide_index=True, use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        # Negative changes indicate faster *observed* laps after the stop;
        # positive changes indicate slower observed laps afterward.
        sns.histplot(compare["Change (s)"], bins=min(12, max(4, len(compare) // 2)), ax=ax)
        # Zero is the reference line separating faster and slower outcomes.
        ax.axvline(0, color="black", linestyle="--")
        ax.set(xlabel="After minus before median lap time (seconds)", ylabel="Pit events")
        st.pyplot(fig); plt.close(fig)
        st.write(f"**Median observed change across {len(compare)} comparable pit events:** "
                 f"{compare['Change (s)'].median():+.2f} seconds; negative values indicate faster laps afterward.")

# End with the limits of the analysis, rather than claiming that the
# observed patterns directly identify an optimal pit-stop lap.
st.divider()
st.subheader("Limitations and next work")
st.write("Fuel consumption, tire allocation, different drivers, overtaking, safety cars, weather and small compound samples "
         "can influence the patterns. A future model should estimate net pit loss, build compound-specific pace curves "
         "and validate predicted pit windows on races not used for fitting.")
