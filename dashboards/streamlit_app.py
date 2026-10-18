"""Local FabSense command centre reading measured project-owned outputs."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import streamlit as st

from fabstack.agent import AnalyticsAgent

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "runs/photo_grid.json"
UTILITIES = ROOT / "runs/utilities/results.json"
MSA = ROOT / "runs/msa/results.json"
DB = ROOT / "runs/utilities/fabsense.duckdb"
CMAPSS = ROOT / "evidence/cmapss_fd001.json"

st.set_page_config(page_title="FabSense local command centre", layout="wide")
st.title("FabSense")
st.caption("Seeded manufacturing/facilities data and separate simulated-engine evidence · local view")

if not all(path.exists() for path in (PHOTO, UTILITIES, MSA, DB)):
    st.error("Measured artifacts are missing. Run the project Makefile targets first.")
    st.stop()

photo = json.loads(PHOTO.read_text())
utilities = json.loads(UTILITIES.read_text())
msa = json.loads(MSA.read_text())
photo_tab, utilities_tab, msa_tab, engine_tab, agent_tab = st.tabs(
    ["Photo bay", "Sustainability", "Workstation MSA", "Engine remaining life", "Ask the data"])

with photo_tab:
    st.header("Scanner train-size scenarios")
    st.image(str(ROOT / "figures/photo_tradeoff.png"),
             caption="20 independent seeded replications per point; source: photo_grid.json")
    left, right = st.columns(2)
    mix = left.selectbox("Recipe mix", ["balanced", "skewed"])
    rate = right.selectbox("Arrival rate (lots/min)", [0.05, 0.065, 0.08])
    scenarios = [row for row in photo["scenarios"] if row["mix"] == mix and
                 row["arrival_rate_per_minute"] == rate]
    st.dataframe([{"max_train_size": row["max_train_size"],
                   "throughput_lots_per_day": row["summary"]["throughput_lots_per_day"]["mean"],
                   "p90_cycle_minutes": row["summary"]["p90_cycle_minutes"]["mean"],
                   "setup_time_share": row["summary"]["setup_time_share"]["mean"],
                   "backlog": row["summary"]["backlog"]["mean"]}
                  for row in scenarios], width="stretch")
    recommendation = next(item for item in photo["recommendations"]
                          if item["mix"] == mix and item["arrival_rate_per_minute"] == rate)
    st.info(f"Example surrogate utility selects train size "
            f"{recommendation['selected']['train_size']}. This model is stylised, not fab-calibrated.")

with utilities_tab:
    st.header("Weather-adjusted meter exceptions")
    metrics = utilities["anomaly"]
    first, second, third = st.columns(3)
    first.metric("Injected anomaly rows", metrics["injected_anomaly_rows"])
    second.metric("Precision", f"{metrics['precision']:.1%}")
    third.metric("Recall", f"{metrics['recall']:.1%}")
    building = st.selectbox("Building", [f"B{i:02d}" for i in range(20)])
    meter = st.selectbox("Meter", ["electricity", "chilled_water", "water"])
    with duckdb.connect(str(DB), read_only=True) as connection:
        series = connection.execute(
            "select hour_index, observed, predicted, flagged, row_id from fact_meter_readings "
            "where building_id=? and metric=? and predicted is not null order by hour_index",
            [building, meter]
        ).df()
    st.line_chart(series.set_index("hour_index")[["observed", "predicted"]])
    st.caption("Query ID: meter_series_by_building_v1. Hovered values map to fact_meter_readings.row_id.")
    with st.expander("Flagged source rows"):
        st.dataframe(series[series["flagged"]], width="stretch")
    insights = json.loads((ROOT / "runs/utilities/insights.json").read_text())
    for insight in insights:
        if insight["building"] == building and insight["metric"] == meter:
            st.info(insight["sentence"])
            st.caption(f"{insight['query_id']} · source row IDs {insight['row_ids']}")

with msa_tab:
    st.header("Old versus new workstation")
    old, new = msa["workstations"]["old"], msa["workstations"]["new"]
    a, b, c = st.columns(3)
    a.metric("Old %GRR", f"{old['anova']['percent_grr']:.2f}%")
    b.metric("New %GRR", f"{new['anova']['percent_grr']:.2f}%")
    c.metric("New − old bias", f"{msa['workstations']['agreement']['new_minus_old_mean']:.3f}")
    st.image(str(ROOT / "figures/msa_agreement.png"),
             caption="Source: runs/msa/measurements.csv and results.json")
    st.json(msa["spc"])

with engine_tab:
    st.header("Simulated engine remaining life")
    st.caption("NASA C-MAPSS FD001 · simulated turbofans, not factory or operating-fleet data")
    if not CMAPSS.exists():
        st.info("FD001 evidence is unavailable. Run `make cmapss` to regenerate it.")
    else:
        engine = json.loads(CMAPSS.read_text())
        test = engine["official_test"]
        interval = engine["official_test_ci_95"]["rmse_95"]
        a, b, c = st.columns(3)
        a.metric("Official test engines", test["n"])
        b.metric("RMSE (cycles)", f"{test['rmse']:.2f}")
        c.metric("MAE (cycles)", f"{test['mae']:.2f}")
        st.caption(
            f"Selected {engine['selected_model']} on {engine['validation_endpoint_count']} "
            f"engine-separated development endpoints; test RMSE bootstrap interval "
            f"[{interval[0]:.2f}, {interval[1]:.2f}] cycles."
        )
        st.subheader("Development selection")
        st.dataframe([
            {"candidate": name, "validation RMSE (cycles)": values["rmse"]}
            for name, values in engine["validation"].items()
        ], width="stretch")
        st.subheader("Error by true remaining life")
        st.dataframe([
            {"slice": name, "engines": values["n"], "RMSE (cycles)": values["rmse"]}
            for name, values in engine["official_test_rul_slices"].items()
        ], width="stretch")
        st.caption("Source: evidence/cmapss_fd001.json. One endpoint per test engine; no prospective validation.")

with agent_tab:
    st.header("Ask the local data")
    st.caption("Deterministic, allow-listed SQL tools. No generative model is used.")
    question = st.text_input("Question", "Which building had the highest electricity excess?")
    if st.button("Run question"):
        response = AnalyticsAgent(DB).answer(question)
        st.write(response["answer"])
        st.json(response["citations"])
