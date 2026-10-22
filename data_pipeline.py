"""
Functions for the F1 streamlit app.
"""

import numpy as np
import pandas as pd


def prepare_laps(laps_raw, stints_raw, pits_raw):
    """
    Clean OpenF1 data and attach tire information to each driver's lap.

    A row represents one driver completing one lap. Each tire stint covers a
    range of laps, so we match laps by driver AND lap range. If the starting
    tire age is unknown, we keep the age missing instead of guessing zero.
    """

    # Make copies so we do not accidentally change the original API data.
    laps, stints, pits = (frame.copy() for frame in (laps_raw, stints_raw, pits_raw))

    # A valid lap must identify the driver, the lap number, and its duration.
    # Convert these fields to numbers; invalid text becomes NaN (missing).
    for field in ("driver_number", "lap_number", "lap_duration"):
        if field not in laps:
            raise ValueError(f"OpenF1 laps missing required field: {field}")
        laps[field] = pd.to_numeric(laps[field], errors="coerce")

    # Convert stint columns only if OpenF1 actually provided them.
    # lap_start/lap_end identify which race laps used each tire set.
    for field in ("driver_number", "lap_start", "lap_end", "tyre_age_at_start", "stint_number"):
        if field in stints:
            stints[field] = pd.to_numeric(stints[field], errors="coerce")

    # Convert available pit-stop fields to numbers for later analysis.
    # lane_duration and stop_duration describe different parts of a stop.
    for field in ("driver_number", "lap_number", "lane_duration", "stop_duration", "pit_duration"):
        if field in pits:
            pits[field] = pd.to_numeric(pits[field], errors="coerce")

    # Count original rows and repeated driver/lap combinations for the quality report.
    original = len(laps)
    duplicates = int(laps.dropna(subset=["driver_number", "lap_number"]).duplicated(["driver_number", "lap_number"]).sum())

    # Remove records without essential values, impossible nonpositive lap times,
    # and duplicated driver/lap records (keeping the first occurrence).
    laps = laps.dropna(subset=["driver_number", "lap_number", "lap_duration"])
    laps = laps[laps["lap_duration"] > 0].drop_duplicates(["driver_number", "lap_number"]).copy()

    # Record how many rows were discarded so the app can explain its cleaning.
    report = {"raw_laps": original, "removed_invalid_or_duplicate_laps": original - len(laps), "duplicate_driver_lap_keys": duplicates}
    if laps.empty:
        raise ValueError("No positive, complete lap rows were returned")

    # Ensure the fields needed to match laps to tire stints exist, even if the
    # API returned an empty stint table. Missing values will be removed below.
    for col in ("driver_number", "lap_start", "lap_end"):
        if col not in stints:
            stints[col] = pd.Series(dtype=float)
    stints = stints.dropna(subset=["driver_number", "lap_start", "lap_end"])

    # Process one driver at a time so another driver's tire stints cannot match.
    chunks = []
    overlap_count = 0
    for driver, frame in laps.groupby("driver_number"):
        frame = frame.copy()

        # Start with unknown compound/age/stint until we find a matching stint.
        frame["compound"] = pd.NA
        frame["tyre_age"] = np.nan
        frame["stint_number"] = np.nan
        frame["stint_matched"] = False

        # A stint applies to every lap between its start and end, inclusive.
        for _, stint in stints[stints["driver_number"] == driver].iterrows():
            mask = frame["lap_number"].between(stint["lap_start"], stint["lap_end"])

            # Count inconsistencies if two stint ranges cover the same lap.
            overlap_count += int((mask & frame["stint_matched"]).sum())

            # When stint ranges overlap, keep the FIRST match and don't overwrite it.
            mask &= ~frame["stint_matched"]
            frame.loc[mask, "compound"] = stint.get("compound", pd.NA)
            frame.loc[mask, "stint_number"] = stint.get("stint_number", np.nan)
            age_at_start = stint.get("tyre_age_at_start", np.nan)

            # Tire age = laps since this stint began + tire age at stint start.
            # A used set of tires might already have laps on it before the stint.
            if pd.notna(age_at_start):
                frame.loc[mask, "tyre_age"] = frame.loc[mask, "lap_number"] - stint["lap_start"] + age_at_start
            frame.loc[mask, "stint_matched"] = True
        chunks.append(frame)

    # Recombine every driver's laps into one table for the dashboard.
    merged = pd.concat(chunks, ignore_index=True)
    report["unmatched_stint_laps"] = int((~merged["stint_matched"]).sum())
    report["missing_tyre_age_laps"] = int(merged["tyre_age"].isna().sum())
    report["overlapping_stint_matches"] = overlap_count

    # Identify laps that might include pit-lane activity. The recorded pit lap
    # and the immediately following lap are BOTH flagged conservatively.
    # This is an analysis rule, not a guarantee about where timing lines fall.
    affected = set()
    if {"driver_number", "lap_number"}.issubset(pits.columns):
        for _, pit in pits.dropna(subset=["driver_number", "lap_number"]).iterrows():
            driver, lap = int(pit["driver_number"]), int(pit["lap_number"])
            affected.update(((driver, lap), (driver, lap + 1)))

    # True means we should not treat this as an ordinary racing-pace lap.
    merged["pit_affected"] = [(int(d), int(l)) in affected for d, l in zip(merged["driver_number"], merged["lap_number"])]

    # Also respect OpenF1's explicit pit-out indicator if it is present.
    if "is_pit_out_lap" in merged.columns:
        merged["pit_affected"] |= merged["is_pit_out_lap"].fillna(False).astype(bool)
    report["pit_affected_laps"] = int(merged["pit_affected"].sum())

    # Send the combined data, original pit records, and cleaning counts to app.py.
    return merged, pits, report


def clean_pace_laps(data, upper_ratio=1.15):
    """
    Keep laps suitable for examining normal tire/race pace.

    Remove pit-affected laps, unusually slow laps, and laps without known tire
    details.
    """

    # Give each driver an individual upper limit, based on typical lap pace.
    cutoffs = data.groupby("driver_number")["lap_duration"].transform("median") * upper_ratio

    # All conditions must be true for a lap to be considered a clean pace lap.
    return data.loc[
        (~data["pit_affected"])        # Not part of the pit-in/pit-out window.
        & (data["lap_duration"] <= cutoffs)  # Not unusually slow for this driver.
        & data["compound"].notna()      # Tire compound is known.
        & data["tyre_age"].notna()      # Tire age is known.
        & (data["tyre_age"] >= 0)       # Tire age is not negative.
    ].copy()


def compare_pit_windows(clean, pits, drivers=None):
    """
    Compare each driver's typical lap times just before and after a stop.

    For each recorded pit event, calculate median lap times using:
        Before: up to the three laps immediately before the recorded pit lap.
        After: laps +2 through +4, avoiding the flagged next lap.
    Require at least two clean laps on each side to make a comparison.
    A negative change means observed laps were faster AFTER the pit stop.
    """

    # Standard column names make the output easy to display in Streamlit.
    columns = ["Driver", "Pit lap", "Before (s)", "After (s)", "Change (s)"]

    # If a pit table cannot identify driver and lap, return an empty result.
    if not {"driver_number", "lap_number"}.issubset(pits.columns):
        return pd.DataFrame(columns=columns)

    # Skip pit events with missing driver/lap values and optionally filter drivers.
    candidate_pits = pits.dropna(subset=["driver_number", "lap_number"])
    if drivers is not None:
        candidate_pits = candidate_pits[candidate_pits["driver_number"].isin(drivers)]
    records = []

    # Examine each pit event independently.
    for _, pit in candidate_pits.iterrows():
        driver, lap = pit["driver_number"], pit["lap_number"]
        driver_clean = clean.loc[clean["driver_number"] == driver]

        # Extract normal-pace laps on each side, leaving out pit-affected laps.
        before = driver_clean.loc[driver_clean["lap_number"].between(lap - 3, lap - 1), "lap_duration"]
        after = driver_clean.loc[driver_clean["lap_number"].between(lap + 2, lap + 4), "lap_duration"]

        # A minimum of two clean laps per window prevents one-lap comparisons.
        if len(before) >= 2 and len(after) >= 2:
            records.append({"Driver": int(driver), "Pit lap": int(lap),
                            "Before (s)": float(before.median()), "After (s)": float(after.median()),
                            "Change (s)": float(after.median() - before.median())})

    # Return a row per comparable pit stop, with the specified column order.
    return pd.DataFrame(records, columns=columns)
