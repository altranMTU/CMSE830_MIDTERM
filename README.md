F1 Pit Stop Strategy Analysis

The goal of this project is to find optimal times to use pit stops along with the effects of pit stops have.

This project uses [OpenF1](https://openf1.org/) race data to explore how tire age, tire compound, and pit stops relate to Formula 1 lap times.

Features
- Compare lap times across tire ages and compounds.
- View driver lap times and pit-stop laps.
- Compare lap times before and after pit stops.
- Explore pit-lane durations using interactive Streamlit filters.

Run the App
Place app.py and data_pipeline.py in the same folder.

start Streamlit:
streamlit run app.py

The app downloads data directly from OpenF1; no CSV file is required.

Data Processing
The code matches laps to tire stints, removes invalid records, and excludes pit-affected or unusually slow laps from normal-pace analysis.

Limitations
Traffic, fuel load, and race conditions also affect lap times. This project explores pit-stop effects but does not yet calculate the optimal pit lap.
