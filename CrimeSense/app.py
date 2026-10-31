from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from src.data import guess_column, clean_crime_data
from src.analysis import dbscan_hotspots, kmeans_hotspots, nearby_events, grid_density
from src.modeling import train_density_model
from src.reports import submit_report, get_reports, update_status

st.set_page_config(page_title="CrimeSense | Crime analysis", page_icon="🛡️", layout="wide")
st.title("🛡️ CrimeSense")
st.caption("Crime hotspot analysis · descriptive density modelling · proximity awareness")
st.warning("Educational prototype only. Hotspots and alerts are not verified threats or individual predictions. Do not use this tool for policing, profiling, or emergency decisions.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Crime records (CSV)", type=["csv"])
    st.caption("Use anonymised records. Latitude and longitude are required.")

if uploaded is None:
    st.info("Upload a CSV file to begin. No real crime records are bundled with this project.")
    st.subheader("Expected data")
    st.code("latitude,longitude,date,crime_type\n12.97,79.13,2024-01-10,Theft\n12.98,79.14,2024-01-11,Assault", language="csv")
    st.markdown("See the README for setup instructions and supported column names.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()
if raw.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

with st.sidebar:
    st.header("2. Map your columns")
    cols = list(raw.columns)
    lat_guess, lon_guess = guess_column(cols, "latitude"), guess_column(cols, "longitude")
    def idx(value): return cols.index(value) if value in cols else 0
    lat_col = st.selectbox("Latitude column", cols, index=idx(lat_guess))
    lon_col = st.selectbox("Longitude column", cols, index=idx(lon_guess))
    date_guess, type_guess = guess_column(cols, "date"), guess_column(cols, "crime_type")
    date_options = ["(not available)"] + cols
    date_col = st.selectbox("Date column", date_options, index=(date_options.index(date_guess) if date_guess in date_options else 0))
    type_col = st.selectbox("Crime category column", ["(not available)"] + cols,
                            index=(["(not available)"] + cols).index(type_guess) if type_guess in cols else 0)

try:
    data, cleaning = clean_crime_data(raw, lat_col, lon_col,
        None if date_col == "(not available)" else date_col,
        None if type_col == "(not available)" else type_col)
except Exception as exc:
    st.error(str(exc)); st.stop()
if data.empty:
    st.error("No valid coordinate rows remain after cleaning."); st.stop()

with st.sidebar:
    st.header("3. Hotspot settings")
    radius = st.slider("DBSCAN neighbourhood radius (metres)", 100, 2000, 500, 50)
    min_samples = st.slider("DBSCAN minimum points", 3, 30, 8)
    k = st.slider("K-Means cluster count", 2, min(12, max(2, len(data))), min(5, max(2, len(data))))

# Cache-independent calculations are intentionally recomputed for clarity.
db = dbscan_hotspots(data, radius, min_samples)
km = kmeans_hotspots(data, k)
cells = grid_density(data)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Uploaded rows", f"{len(raw):,}")
m2.metric("Valid records", f"{len(data):,}")
m3.metric("DBSCAN clusters", int(db.loc[db.dbscan_cluster >= 0, "dbscan_cluster"].nunique()))
m4.metric("DBSCAN noise", f"{(db.dbscan_cluster == -1).mean():.1%}")
with st.expander("Data-cleaning summary"):
    st.json(cleaning)
    st.dataframe(data.head(20), use_container_width=True)

tabs = st.tabs(["Overview & EDA", "Hotspot analysis", "Risk model", "Map & safety radius", "Community reports", "Admin review"])
with tabs[0]:
    st.subheader("Crime data overview")
    left, right = st.columns(2)
    with left:
        counts = data.crime_type.value_counts().head(15).rename_axis("Crime category").reset_index(name="Records")
        st.plotly_chart(px.bar(counts, x="Records", y="Crime category", orientation="h", title="Most frequent categories"), use_container_width=True)
    with right:
        dated = data.dropna(subset=["date"]).copy()
        if not dated.empty:
            monthly = dated.assign(month=dated.date.dt.to_period("M").astype(str)).groupby("month").size().reset_index(name="Records")
            st.plotly_chart(px.line(monthly, x="month", y="Records", markers=True, title="Monthly reported events"), use_container_width=True)
        else:
            st.info("No usable date column was selected, so a time trend cannot be plotted.")
    st.plotly_chart(px.scatter(data.sample(min(len(data), 10000), random_state=42), x="longitude", y="latitude",
                               color="crime_type", title="Geographic distribution (up to 10,000 sampled rows)"), use_container_width=True)

with tabs[1]:
    st.subheader("Spatial clustering")
    a, b = st.columns(2)
    with a:
        st.markdown("**DBSCAN**")
        st.write("Groups nearby events using a distance radius and minimum number of points. Label -1 represents noise, not a low-risk score.")
        st.dataframe(db.dbscan_cluster.value_counts().rename_axis("Cluster").reset_index(name="Records"), use_container_width=True)
    with b:
        st.markdown("**K-Means**")
        st.write("Partitions the event coordinates into the selected number of clusters. Cluster IDs are arbitrary labels.")
        st.dataframe(km.kmeans_cluster.value_counts().rename_axis("Cluster").reset_index(name="Records"), use_container_width=True)
    st.plotly_chart(px.scatter(db.sample(min(len(db), 12000), random_state=42), x="longitude", y="latitude",
                               color=db.dbscan_cluster.astype(str), title="DBSCAN hotspot groups"), use_container_width=True)

with tabs[2]:
    st.subheader("Grid-density classification")
    st.caption("This classifies relative event density in grid cells from the uploaded history. It is not a forecast of an individual crime.")
    model_name = st.radio("Model", ["Random Forest", "XGBoost"], horizontal=True)
    if st.button("Train and evaluate model"):
        with st.spinner("Training model on the uploaded dataset..."):
            model, metrics, labelled = train_density_model(cells, model_name)
        st.session_state["model_metrics"] = metrics
        st.session_state["model_cells"] = labelled
        st.session_state["model_name"] = model_name
    metrics = st.session_state.get("model_metrics")
    if metrics:
        if not metrics["available"]:
            st.warning(metrics["reason"])
        else:
            x, y, z = st.columns(3)
            x.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            y.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
            z.metric("Test cells", metrics["test_rows"])
            st.write(f"Model: {metrics['model']} · Training cells: {metrics['train_rows']} · Relative density threshold: {metrics['threshold']:.2f} events/cell")
            st.json(metrics["report"])
            st.dataframe(st.session_state["model_cells"].head(30), use_container_width=True)
    else:
        st.info("Choose a model and click Train and evaluate. At least 12 grid cells with usable variation are required.")

with tabs[3]:
    st.subheader("Interactive event map")
    color_data = db.copy()
    color_data["cluster_label"] = color_data.dbscan_cluster.apply(lambda v: "Noise" if v == -1 else f"Cluster {v}")
    st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", data=color_data.sample(min(len(color_data), 12000), random_state=42),
        get_position="[longitude, latitude]", get_radius=70, get_fill_color=[220, 70, 70, 150], pickable=True)],
        initial_view_state=pdk.ViewState(latitude=float(data.latitude.mean()), longitude=float(data.longitude.mean()), zoom=10, pitch=0),
        tooltip={"text": "{crime_type}\n{cluster_label}"}))
    st.markdown("**Check a user-selected location**")
    c1, c2, c3 = st.columns(3)
    alert_lat = c1.number_input("Latitude", value=float(data.latitude.mean()), format="%.6f")
    alert_lon = c2.number_input("Longitude", value=float(data.longitude.mean()), format="%.6f")
    alert_radius = c3.slider("Search radius (km)", 0.1, 10.0, 1.0, 0.1)
    nearby = nearby_events(data, alert_lat, alert_lon, alert_radius)
    st.metric("Historical records within radius", len(nearby))
    if len(nearby):
        st.dataframe(nearby[["date", "crime_type", "latitude", "longitude", "distance_km"]].head(100), use_container_width=True)
    st.caption("This is a count of historical uploaded records, not a live threat alert. Check the source data date range and coverage.")

with tabs[4]:
    st.subheader("Submit a community report")
    st.info("Demo only: do not submit emergencies, names, contact details, or identifying information. Reports are stored locally on this server and are not automatically verified.")
    with st.form("community_report"):
        category = st.selectbox("Category", ["Theft", "Assault", "Vandalism", "Harassment", "Other"])
        description = st.text_area("Brief description (avoid personal details)", max_chars=2000)
        r1, r2 = st.columns(2)
        report_lat = r1.number_input("Incident latitude", value=float(data.latitude.mean()), format="%.6f", key="report_lat")
        report_lon = r2.number_input("Incident longitude", value=float(data.longitude.mean()), format="%.6f", key="report_lon")
        submitted = st.form_submit_button("Submit report")
        if submitted:
            if not description.strip(): st.error("Please enter a short description.")
            elif not (-90 <= report_lat <= 90 and -180 <= report_lon <= 180): st.error("Coordinates are outside valid ranges.")
            else:
                submit_report(category, description.strip(), report_lat, report_lon)
                st.success("Report saved locally with Pending status.")

with tabs[5]:
    st.subheader("Admin report review")
    st.caption("Local demonstration moderation. Set ADMIN_PASSWORD before deployment; this default is intentionally not secure.")
    password = st.text_input("Admin password", type="password")
    expected = os.getenv("ADMIN_PASSWORD", "change-me")
    if password and password == expected:
        reports = get_reports()
        if not reports:
            st.info("No community reports have been submitted yet.")
        else:
            report_df = pd.DataFrame(reports, columns=["ID", "Created at", "Category", "Description", "Latitude", "Longitude", "Status"])
            st.dataframe(report_df, use_container_width=True)
            report_ids = report_df.ID.tolist()
            selected_id = st.selectbox("Report ID", report_ids)
            new_status = st.selectbox("Set status", ["Pending", "Verified", "Rejected"])
            if st.button("Update report status"):
                update_status(selected_id, new_status)
                st.success("Status updated. Refresh the page to see the latest list.")
    elif password:
        st.error("Incorrect admin password.")
