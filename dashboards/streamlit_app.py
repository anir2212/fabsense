"""Local FabSense command centre reading measured project-owned outputs.

Each tab loads only its own evidence. A missing or malformed file produces a
message naming the file and its regeneration command instead of numbers.
"""

from __future__ import annotations

import duckdb
import streamlit as st

from fabstack import dashboard_evidence as ev
from fabstack.agent import AnalyticsAgent

ROOT = ev.artifact_root()

st.set_page_config(page_title="FabSense local command centre", layout="wide")
st.title("FabSense")
st.caption("Seeded manufacturing/facilities data and separate simulated-engine evidence · local view")
if ROOT != ev.PROJECT_ROOT:
    # Captions fade everything inside them, so inline code would fail contrast here.
    st.caption(f"Artifact root override from {ev.ROOT_ENV}: {ROOT}")


def explain(problem: ev.EvidenceProblem) -> None:
    if problem.kind == "missing":
        st.warning(problem.message)
    else:
        st.error(problem.message)


photo, photo_problem = ev.load(ROOT, "photo", ev.photo_view)
photo_figure, photo_figure_problem = ev.figure(ROOT, "photo_figure")
utilities, utilities_problem = ev.load(ROOT, "utilities", ev.utilities_view)
insights, insights_problem = ev.load(ROOT, "insights", ev.insights_view)
database, database_problem = ev.load_database(ROOT)
msa, msa_problem = ev.load(ROOT, "msa", ev.msa_view)
msa_figure, msa_figure_problem = ev.figure(ROOT, "msa_figure")
engine, engine_problem = ev.load(ROOT, "cmapss", ev.engine_view)

problems = [problem for problem in (photo_problem, photo_figure_problem, utilities_problem,
                                    insights_problem, database_problem, msa_problem,
                                    msa_figure_problem, engine_problem) if problem]
if problems:
    st.warning(f"{len(problems)} evidence file(s) are missing or malformed: "
               + ", ".join(f"`{problem.artifact.path}`" for problem in problems)
               + ". The affected tabs say which command regenerates each file.")

photo_tab, utilities_tab, msa_tab, engine_tab, agent_tab = st.tabs(
    ["Photo bay", "Sustainability", "Workstation MSA", "Engine remaining life", "Ask the data"])

with photo_tab:
    st.header("Scanner train-size scenarios")
    if photo_figure_problem:
        explain(photo_figure_problem)
    else:
        st.image(str(photo_figure),
                 caption="20 independent seeded replications per point; source: photo_grid.json")
    if photo_problem:
        explain(photo_problem)
    else:
        left, right = st.columns(2)
        mix = left.selectbox("Recipe mix", photo["mixes"])
        rate = right.selectbox("Arrival rate (lots/min)", photo["rates"])
        st.dataframe([{name: row[name] for name in ("max_train_size", *ev.PHOTO_COLUMNS)}
                      for row in photo["scenarios"]
                      if row["mix"] == mix and row["arrival_rate_per_minute"] == rate],
                     width="stretch")
        selected = photo["recommendations"].get((mix, rate))
        if selected is None:
            st.warning("No surrogate recommendation is recorded for this scenario in "
                       "`runs/photo_grid.json`.")
        else:
            st.info(f"Example surrogate utility selects train size {selected}. "
                    "This model is stylised, not fab-calibrated.")

with utilities_tab:
    st.header("Weather-adjusted meter exceptions")
    if utilities_problem:
        explain(utilities_problem)
    else:
        first, second, third = st.columns(3)
        first.metric("Injected anomaly rows", utilities["injected_anomaly_rows"])
        second.metric("Precision", f"{utilities['precision']:.1%}")
        third.metric("Recall", f"{utilities['recall']:.1%}")
    if database_problem:
        explain(database_problem)
    else:
        building = st.selectbox("Building", database["buildings"])
        meter = st.selectbox("Meter", database["metrics"])
        try:
            with duckdb.connect(str(database["path"]), read_only=True) as connection:
                series = connection.execute(
                    "select hour_index, observed, predicted, flagged, row_id "
                    "from fact_meter_readings "
                    "where building_id=? and metric=? and predicted is not null order by hour_index",
                    [building, meter]
                ).df()
        except duckdb.Error as error:
            explain(ev.EvidenceProblem("database", "malformed",
                                       f"meter series query failed: {type(error).__name__}"))
        else:
            st.line_chart(series.set_index("hour_index")[["observed", "predicted"]])
            st.caption("Query ID: meter_series_by_building_v1. "
                       "Hovered values map to fact_meter_readings.row_id.")
            with st.expander("Flagged source rows"):
                st.dataframe(series[series["flagged"]], width="stretch")
        if insights_problem:
            explain(insights_problem)
        else:
            for insight in insights:
                if insight["building"] == building and insight["metric"] == meter:
                    st.info(insight["sentence"])
                    st.caption(f"{insight['query_id']} · source row IDs {insight['row_ids']}")

with msa_tab:
    st.header("Old versus new workstation")
    if msa_problem:
        explain(msa_problem)
    else:
        a, b, c = st.columns(3)
        a.metric("Old %GRR", f"{msa['old_percent_grr']:.2f}%")
        b.metric("New %GRR", f"{msa['new_percent_grr']:.2f}%")
        c.metric("New − old bias", f"{msa['new_minus_old_mean']:.3f}")
    if msa_figure_problem:
        explain(msa_figure_problem)
    else:
        st.image(str(msa_figure), caption="Source: runs/msa/measurements.csv and results.json")
    if not msa_problem:
        st.json(msa["spc"])

with engine_tab:
    st.header("Simulated engine remaining life")
    st.caption("NASA C-MAPSS FD001 · simulated turbofans, not factory or operating-fleet data")
    if engine_problem:
        explain(engine_problem)
        st.caption("Running make cmapss needs requirements-secom.txt and downloads NASA's pinned "
                   "archive into the ignored data/raw/ folder if it is not already there.")
    else:
        interval = engine["interval"]
        a, b, c = st.columns(3)
        a.metric("Official test engines", engine["n"])
        b.metric("RMSE (cycles)", f"{engine['rmse']:.2f}")
        c.metric("MAE (cycles)", f"{engine['mae']:.2f}")
        st.caption(
            f"Selected {engine['selected_model']} on {engine['validation_endpoint_count']} "
            f"engine-separated development endpoints; test RMSE bootstrap interval "
            f"[{interval[0]:.2f}, {interval[1]:.2f}] cycles."
        )
        # Static HTML tables without a row-number column: readable by assistive technology
        # and identical to RESULTS.md rounding.
        tables = ev.engine_tables(engine)
        st.subheader("Development selection")
        st.markdown(ev.markdown_table(tables["validation"]))
        st.subheader("Error by true remaining life")
        st.markdown(ev.markdown_table(tables["slices"]))
        st.caption("Source: evidence/cmapss_fd001.json. "
                   "One endpoint per test engine; no prospective validation.")

with agent_tab:
    st.header("Ask the local data")
    st.caption("Deterministic, allow-listed SQL tools. No generative model is used.")
    if database_problem:
        explain(database_problem)
    else:
        question = st.text_input("Question", "Which building had the highest electricity excess?")
        if st.button("Run question"):
            try:
                response = AnalyticsAgent(database["path"]).answer(question)
            except duckdb.Error as error:
                explain(ev.EvidenceProblem("database", "malformed",
                                           f"question query failed: {type(error).__name__}"))
            else:
                st.write(response["answer"])
                st.json(response["citations"])
