# pages/p_compare_export.py
# Comparison page (Run 1 vs Run 2) and Export page.
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# Local core engine imports
from modules.metrics_engine import compute_inter_metrics, compute_intra_metrics
from modules.report_gen import generate_report, draw_venn_diagram
from modules.splink_runner import (
    run_linkage, determine_cascade_order, filter_predict_by_active_rules,
    recluster_filtered, build_coverage_matrix,
)
from modules.metrics_engine import compute_intra_metrics
from utils.helpers import (
    _metric_cards, _plotly_bar, _run_analysis_and_store, render_demographic_breakdowns,
    render_demographic_comparison, compute_demographic_snapshot, cohort_from_edges,
    render_match_quality_section, split_linked_unlinked,
    render_blocking_waterfall, render_waterfall_section,
)
from utils.nav import _back_button, _go_to
import plotly.graph_objects as gobj
import streamlit.components.v1 as components


def page_comparison():
    _back_button()
    st.title("Step 6: Compare Runs")

    if st.session_state.get("run1_results") is None:
        st.warning("No Run 1 results. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    run1 = st.session_state["run1_results"]
    m1   = st.session_state["run1_metrics"]

    st.divider()

    tab_within, tab_rerun = st.tabs([
        "Within-run rule toggle analysis",
        "Full re-run with new blocking rules",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Within-run: toggle match_keys off, re-cluster the existing edges
    # This mirrors the linkage-metrics notebook approach: filter df_predict by
    # removing rows belonging to a disabled match_key, then re-cluster.
    # No new model training or prediction is needed — instant results.
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_within:
        st.write(
            "Toggle individual blocking rules on or off below. "
            "The app removes candidate pairs that were **only** captured by disabled "
            "rules, then re-clusters the remaining edges instantly — no re-training needed. "
            "This shows the marginal contribution of each rule."
        )

        # ── Coverage matrix (built once, cached in session) ───────────────────
        cov = st.session_state.get("coverage_matrix")
        if cov is None or cov.empty:
            st.info(
                "Coverage matrix not available. "
                "Run the analysis first (Step 5) to enable within-run comparison."
            )
        else:
            # ── Rule toggles ──────────────────────────────────────────────────
            st.subheader("Toggle blocking rules")
            run1_toggles = run1["run_config"]["blocking_toggles"]

            if "within_toggles" not in st.session_state:
                st.session_state["within_toggles"] = dict(run1_toggles)

            # Show ALL rules: single-field AND composite from run_config
            all_rules = dict(run1_toggles)
            composite = run1["run_config"].get("composite_rules", {})
            all_rules.update(composite)

            wt = {}
            n_cols = min(4, max(1, len(all_rules)))
            wt_cols = st.columns(n_cols)
            for i, (field, was_on) in enumerate(all_rules.items()):
                col = wt_cols[i % n_cols]
                label = field.replace("+", " + ")   # make composite rules readable
                wt[field] = col.toggle(
                    label, value=st.session_state["within_toggles"].get(field, was_on),
                    key=f"wt_{field}",
                )
            st.session_state["within_toggles"] = wt

            # ── Blocking cascade waterfall (live) ───────────────────────────────
            render_waterfall_section(
                cov, wt, run1["run_config"].get("blocking_mode", "OR"),
                key_prefix="within_waterfall", run_label="Run 1",
            )

            # ── Filter + re-cluster ───────────────────────────────────────────
            filtered_df = filter_predict_by_active_rules(
                run1["df_predict"], cov, wt
            )
            n_filt = len(filtered_df)
            n_orig = len(run1["df_predict"])

            removed = n_orig - n_filt
            st.caption(
                f"Edges after toggle: **{n_filt:,}** "
                f"(removed {removed:,} = {100*removed/max(n_orig,1):.1f}% of Run 1 edges)"
            )

            # ── Real-time demographic comparison (no re-cluster needed) ────────
            # Uses records touched by the currently active blocking rules as a
            # live proxy cohort, so this updates on every toggle instantly.
            # Works identically for dummy dataset (gender/city) and NC voter
            # registry (Tier-1 fields) since compute_demographic_snapshot is
            # fully data-driven.
            st.divider()
            try:
                _baseline_snap = compute_demographic_snapshot(run1.get("df_cluster", pd.DataFrame()))
                _live_cohort   = cohort_from_edges(
                    st.session_state.get("fakea"), st.session_state.get("fakeb"), filtered_df
                )
                _current_snap  = compute_demographic_snapshot(_live_cohort)
                render_demographic_comparison(
                    _baseline_snap, _current_snap,
                    baseline_label="Run 1", current_label="Run 1 (toggled, live)",
                    key_prefix="within_demo_cmp",
                )
            except Exception as _demo_err:
                st.warning(f"Could not compute live demographic comparison: {_demo_err}")

            # ── Match quality by demographic group (real-time) ─────────────────
            st.divider()
            try:
                _within_fields = run1["run_config"].get("selected_fields", [])
                render_match_quality_section(
                    run1["df_predict"], filtered_df,
                    _within_fields, _within_fields,
                    threshold=run1["run_config"].get("cluster_threshold", 0.8),
                    baseline_label="Run 1", current_label="Run 1 (toggled, live)",
                    key_prefix="within_match_quality",
                )
            except Exception as _mq_err:
                st.warning(f"Could not compute match-quality comparison: {_mq_err}")

            if st.button("Re-cluster with toggled rules", type="primary", key="within_recluster"):
                with st.spinner("Re-clustering…"):
                    threshold = run1["run_config"].get("cluster_threshold", 0.8)
                    new_clusters = recluster_filtered(
                        filtered_df,
                        st.session_state["fakea"],
                        st.session_state.get("fakeb"),
                        threshold=threshold,
                    )
                    st.session_state["within_clusters"] = new_clusters

            within_clusters = st.session_state.get("within_clusters")
            if within_clusters is not None and not within_clusters.empty:
                n_new_cl = within_clusters["cluster_id"].nunique()
                n_orig_cl = m1["n_clusters"]

                wk1, wk2, wk3 = st.columns(3)
                wk1.metric("Edges (toggled)", f"{n_filt:,}",
                           delta=f"{n_filt - n_orig:+,}")
                wk2.metric("Clusters (re-clustered)", f"{n_new_cl:,}",
                           delta=f"{n_new_cl - n_orig_cl:+,}")
                wk3.metric("Edges removed", f"{removed:,}")

                # Set-difference metrics (from the linkage-metrics notebook pattern)
                import duckdb as _ddb
                _con = _ddb.connect()
                _con.register("orig_edges", run1["df_predict"][
                    ["unique_id_l","unique_id_r","source_dataset_l","source_dataset_r"]
                ])
                _con.register("filt_edges", filtered_df[
                    ["unique_id_l","unique_id_r","source_dataset_l","source_dataset_r"]
                ] if not filtered_df.empty else run1["df_predict"].iloc[0:0][
                    ["unique_id_l","unique_id_r","source_dataset_l","source_dataset_r"]
                ])

                shared_n = _con.sql("""
                    SELECT COUNT(*) FROM orig_edges o
                    INNER JOIN filt_edges f
                    USING (unique_id_l, unique_id_r, source_dataset_l, source_dataset_r)
                """).fetchone()[0]
                removed_n = n_orig - shared_n
                _con.close()

                st.write("**Set-difference edge metrics**")
                st.dataframe(pd.DataFrame([
                    {"Metric": "Edges in original run", "Count": n_orig},
                    {"Metric": "Edges retained after toggle", "Count": shared_n},
                    {"Metric": "Edges removed by disabling rules", "Count": removed_n},
                    {"Metric": "Clusters (original)", "Count": n_orig_cl},
                    {"Metric": "Clusters (after toggle)", "Count": n_new_cl},
                    {"Metric": "Cluster delta", "Count": n_new_cl - n_orig_cl},
                ]), use_container_width=True, hide_index=True)

                st.caption(
                    "Interpretation: 'Edges removed' shows how many candidate pairs "
                    "were contributed exclusively by the disabled rule(s). "
                    "A large number means that rule was capturing many unique pairs."
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Full re-run with new blocking rules
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_rerun:
        st.write(
            "Run a completely new linkage model with different blocking rules. "
            "This re-trains (for probabilistic) or re-predicts (deterministic) "
            "from scratch, giving a fully independent second result to compare."
        )
        st.subheader("Run 1 summary")
        active1 = [f for f, v in run1["run_config"]["blocking_toggles"].items() if v]
        st.caption(f"Blocking rules: {', '.join(active1)}")
        _metric_cards([
            ("Run 1: Edges",    f"{m1['n_edges']:,}"),
            ("Run 1: Clusters", f"{m1['n_clusters']:,}"),
            ("Run 1: Mean match prob",
             str(m1["match_prob_stats"]["mean_match_prob"].iloc[0])
             if not m1["match_prob_stats"].empty else "N/A"),
        ])

        st.divider()
        st.subheader("Modify blocking rules for Run 2")

        if st.session_state.get("run2_blocking_toggles") is None:
            st.session_state["run2_blocking_toggles"] = dict(
                run1["run_config"]["blocking_toggles"]
            )

        r2_toggles = {}
        tc = st.columns(3)
        for i, field in enumerate(st.session_state["selected_fields"]):
            col = tc[i % 3]
            enabled = col.toggle(
                field,
                value=st.session_state["run2_blocking_toggles"].get(field, True),
                key=f"r2_{field}",
            )
            r2_toggles[field] = enabled

        # ── Composite blocking rules for Run 2 ────────────────────────────────
        with st.expander("Composite blocking rules for Run 2 (optional)", expanded=False):
            st.write("Combine two or three fields into a single AND rule.")
            if "r2_composite_rules" not in st.session_state:
                st.session_state["r2_composite_rules"] = {}
            sel_fields = st.session_state.get("selected_fields", [])
            if len(sel_fields) >= 2:
                rc1, rc2, rc3, rc4 = st.columns([2, 2, 2, 1])
                rf1 = rc1.selectbox("Field 1", sel_fields, key="r2_cb_f1")
                rf2_opts = [f for f in sel_fields if f != rf1]
                rf2 = rc2.selectbox("Field 2", rf2_opts, key="r2_cb_f2")
                rf3_opts = ["(none)"] + [f for f in sel_fields if f not in (rf1, rf2)]
                rf3_sel = rc3.selectbox("Field 3 (optional)", rf3_opts, key="r2_cb_f3")
                rf3 = None if rf3_sel == "(none)" else rf3_sel
                if rc4.button("Add rule", key="r2_cb_add"):
                    rkey = f"{rf1}+{rf2}" + (f"+{rf3}" if rf3 else "")
                    st.session_state["r2_composite_rules"][rkey] = True
            for rkey in list(st.session_state.get("r2_composite_rules", {}).keys()):
                rparts = rkey.split("+")
                rsql = " AND ".join(f'l."{p}" = r."{p}"' for p in rparts)
                rcr1, rcr2 = st.columns([4, 1])
                rcr1.code(rsql)
                if rcr2.button("Remove", key=f"r2_rm_{rkey}"):
                    del st.session_state["r2_composite_rules"][rkey]

        has_single_r2    = any(r2_toggles.values())
        has_composite_r2 = bool(st.session_state.get("r2_composite_rules"))

        # ── FIX: always persist toggles so the next rerun sees current values ─
        # Previously this was inside the else: block which meant the FIRST click
        # saved state but the run didn't fire until the SECOND click.
        st.session_state["run2_blocking_toggles"] = r2_toggles

        if not has_single_r2 and not has_composite_r2:
            st.error("At least one blocking rule (single-field or composite) must be defined for Run 2.")
        else:
            if st.button("Run full analysis with updated blocking rules", type="primary"):
                fakea_r2 = st.session_state.get("fakea")
                if fakea_r2 is not None:
                    n_a = len(fakea_r2)
                    fakeb_r2 = st.session_state.get("fakeb")
                    n_b = len(fakeb_r2) if fakeb_r2 is not None else n_a
                    _blocked = False
                    for _key, _en in r2_toggles.items():
                        if not _en or "+" in _key or _key not in fakea_r2.columns:
                            continue
                        _n_u = fakea_r2[_key].nunique()
                        if _n_u == 0:
                            continue
                        _est = int((n_a / _n_u) * (n_b / _n_u) * _n_u)
                        if _est > 5_000_000:
                            st.error(
                                f"Blocking on `{_key}` would generate ~{_est:,} pairs "
                                f"({_n_u} unique values). Disable this rule."
                            )
                            _blocked = True
                    if not _blocked:
                        with st.spinner("Running Run 2…"):
                            try:
                                run2 = run_linkage(
                                    fakea=st.session_state["fakea"],
                                    fakeb=st.session_state["fakeb"],
                                    selected_fields=st.session_state["selected_fields"],
                                    blocking_toggles=r2_toggles,
                                    operation_mode=st.session_state["operation_mode"],
                                    linkage_type=st.session_state["linkage_type"],
                                    hyperparams=st.session_state.get("hyperparams", {}),
                                    composite_rules=st.session_state.get("r2_composite_rules", {}),
                                )
                                m2 = compute_intra_metrics(run2["df_predict"], run2["df_cluster"])

                                from modules.metrics_engine import (
                                    compute_confusion_matrix as _cm2_fn,
                                    compute_truth_space      as _ts2_fn,
                                    compute_crl_score        as _crl2_fn,
                                )
                                from modules.splink_runner import build_coverage_matrix as _bcov2_fn

                                _op2 = run2["run_config"]["operation_mode"]
                                _lt2 = run2["run_config"]["linkage_type"]
                                _fa2 = st.session_state["fakea"]
                                _fb2 = st.session_state.get("fakeb")
                                _cm2  = _cm2_fn(run2["df_predict"], _fa2, _fb2, _op2)
                                _cov2 = _bcov2_fn(run2["df_predict"],
                                                  run2["run_config"]["selected_fields"])

                                if _lt2 == "probabilistic":
                                    _ts2  = _ts2_fn(run2["df_predict"], _fa2, _fb2, _op2)
                                    _crl2 = _crl2_fn(_ts2)
                                else:
                                    _ts2, _crl2 = None, {}

                                st.session_state["run2_results"] = run2
                                st.session_state["run2_metrics"] = m2
                                st.session_state["run2_cm"]      = _cm2
                                st.session_state["run2_cov"]     = _cov2
                                st.session_state["run2_ts"]      = _ts2
                                st.session_state["run2_crl"]     = _crl2
                                st.success("Run 2 complete.")
                            except Exception as e:
                                st.error(f"Run 2 failed: {e}")

        # ── Run 2 results — full metrics + comparison ─────────────────────────
        if st.session_state.get("run2_results") is not None:
            run2 = st.session_state["run2_results"]
            m2   = st.session_state["run2_metrics"]
            lt2  = run2["run_config"].get("linkage_type", "deterministic")

            st.divider()

            r2_tab_metrics, r2_tab_compare = st.tabs([
                "Run 2 — Full Metrics",
                "Run 1 vs Run 2 Comparison",
            ])

            # ── Run 2 Full Metrics (mirrors Run 1 analysis page) ──────────────
            with r2_tab_metrics:
                st.subheader("Run 2 Results")
                _metric_cards([
                    ("Edges",         f"{m2['n_edges']:,}"),
                    ("Clusters",      f"{m2['n_clusters']:,}"),
                    ("Unique IDs",    f"{m2['n_unique_ids']:,}"),
                    ("Cross-dataset", f"{m2['n_cross_dataset']:,}"),
                ])

                (r2_sub_edge, r2_sub_cluster, r2_sub_demo,
                 r2_sub_explorer, r2_sub_studio, r2_sub_cm, r2_sub_data) = st.tabs([
                    "Edge Metrics", "Cluster Metrics", "Demographics",
                    "Blocking Explorer", "Cluster Studio", "Confusion Matrix", "Raw Data",
                ])

                with r2_sub_edge:
                    prob_stats2 = m2.get("match_prob_stats", pd.DataFrame())
                    if not prob_stats2.empty:
                        st.write("**Match Probability Statistics**")
                        st.dataframe(prob_stats2, use_container_width=True)
                    prob_dist2 = m2.get("prob_dist", pd.DataFrame())
                    if not prob_dist2.empty and len(prob_dist2) > 1:
                        st.plotly_chart(
                            _plotly_bar(prob_dist2, "prob_bin", "n_edges",
                                        "Match Probability Distribution"),
                            use_container_width=True,
                        )
                    wd2 = m2.get("weight_dist", pd.DataFrame())
                    if not wd2.empty and len(wd2) > 1:
                        st.plotly_chart(
                            _plotly_bar(wd2, "weight_bin", "n_edges",
                                        "Match Weight Histogram", "#E55C30"),
                            use_container_width=True,
                        )
                    g2 = m2.get("gamma_means", pd.DataFrame())
                    if not g2.empty and lt2 == "probabilistic":
                        g2_long = g2.T.reset_index()
                        g2_long.columns = ["field", "mean_gamma"]
                        g2_long["field"] = g2_long["field"].str.replace("gamma_", "", regex=False)
                        st.plotly_chart(
                            _plotly_bar(g2_long, "field", "mean_gamma",
                                        "Mean Gamma Score per Field", "#2ECC71"),
                            use_container_width=True,
                        )

                with r2_sub_cluster:
                    c2_1, c2_2 = st.columns(2)
                    c2_1.metric("Total clusters", f"{m2['n_clusters']:,}")
                    c2_2.metric("Cross-dataset clusters", f"{m2['n_cross_dataset']:,}")
                    s2 = m2.get("singleton_stats", pd.DataFrame())
                    if not s2.empty:
                        st.write("**Singleton vs Multi-record Clusters**")
                        _s2c1, _s2c2 = st.columns([1, 1])
                        with _s2c1:
                            st.dataframe(s2, use_container_width=True, hide_index=True)
                        with _s2c2:
                            if "cluster_type" in s2.columns and "n_clusters" in s2.columns:
                                _s2fig = px.bar(
                                    s2, x="cluster_type", y="n_clusters",
                                    color="cluster_type",
                                    color_discrete_sequence=["#1E6EC4", "#E55C30"],
                                    title="Singleton vs Multi-record",
                                    template="simple_white",
                                    labels={"cluster_type": "", "n_clusters": "N clusters"},
                                )
                                _s2fig.update_layout(height=260, showlegend=False,
                                                     margin=dict(l=10,r=10,t=40,b=10))
                                st.plotly_chart(_s2fig, use_container_width=True)
                    cs2 = m2.get("cluster_sizes", pd.DataFrame())
                    if not cs2.empty:
                        st.plotly_chart(
                            _plotly_bar(cs2, "n_nodes", "n_clusters",
                                        "Cluster Size Distribution"),
                            use_container_width=True,
                        )
                    venn2 = m2.get("venn", {})
                    op2   = run2["run_config"]["operation_mode"]
                    if op2 != "dedupe" and any(venn2.values()):
                        a2, b2, ab2 = venn2.get("a_only",0), venn2.get("b_only",0), venn2.get("both_ab",0)
                        st.dataframe(pd.DataFrame([
                            {"Category":"Dataset A only","N Clusters":a2},
                            {"Category":"Both A and B",  "N Clusters":ab2},
                            {"Category":"Dataset B only","N Clusters":b2},
                        ]), use_container_width=True, hide_index=True)
                        _vf2 = draw_venn_diagram(
                            a2, ab2, b2,
                            figsize=(3, 2.1), title_fontsize=8, number_fontsize=10, label_fontsize=6,
                            title="Venn Diagram — Cluster_id Set Membership",
                        )
                        _vcol2_l, _vcol2_mid, _vcol2_r = st.columns([1, 1, 1])
                        with _vcol2_mid:
                            st.pyplot(_vf2, use_container_width=True)
                        import matplotlib.pyplot as _plt2
                        _plt2.close(_vf2)

                with r2_sub_demo:
                    g2d = m2.get("gender_dist", pd.DataFrame())
                    c2d = m2.get("city_dist",   pd.DataFrame())
                    nc2d = m2.get("nc_demographics", {})
                    d2_1, d2_2 = st.columns(2)
                    if not g2d.empty:
                        with d2_1:
                            st.plotly_chart(
                                px.pie(g2d, values="n_records", names="gender",
                                       title="Gender Distribution (Run 2)",
                                       template="simple_white",
                                       color_discrete_sequence=px.colors.qualitative.Set2),
                                use_container_width=True,
                            )
                    if not c2d.empty:
                        with d2_2:
                            st.plotly_chart(
                                _plotly_bar(c2d.head(10), "city", "n_records",
                                            "Top 10 Locations (Run 2)", "#9B59B6"),
                                use_container_width=True,
                            )
                    if nc2d:
                        st.divider()
                        render_demographic_breakdowns(nc2d, key_prefix="run2_demo")
                    if g2d.empty and c2d.empty and not nc2d:
                        st.info("No demographic columns found in Run 2 results.")

                    st.divider()
                    st.write("**Linked vs. Unlinked — Demographic Comparison (Run 2)**")
                    try:
                        _r2_linked_df, _r2_unlinked_df = split_linked_unlinked(
                            run2.get("df_cluster", pd.DataFrame())
                        )
                        _r2_linked_snap   = compute_demographic_snapshot(_r2_linked_df)
                        _r2_unlinked_snap = compute_demographic_snapshot(_r2_unlinked_df)
                        render_demographic_comparison(
                            _r2_linked_snap, _r2_unlinked_snap,
                            baseline_label="Linked", current_label="Unlinked",
                            key_prefix="r2_linked_unlinked",
                        )
                        st.caption(
                            f"Linked: {len(_r2_linked_df):,} records. "
                            f"Unlinked: {len(_r2_unlinked_df):,} records."
                        )
                    except Exception as _r2_lu_err:
                        st.warning(f"Could not compute linked vs unlinked comparison: {_r2_lu_err}")

                # ── Blocking Explorer (Run 2) ──────────────────────────────────
                with r2_sub_explorer:
                    st.subheader("Interactive Blocking Explorer — Run 2")
                    st.write(
                        "Toggle Run 2 blocking rules on or off. "
                        "Pairs captured only by disabled rules are removed; "
                        "the remaining edges can be re-clustered instantly."
                    )
                    _fpar2, _rcf2, _bcm2 = filter_predict_by_active_rules, recluster_filtered, build_coverage_matrix
                    _cov2_disp = st.session_state.get("run2_cov")
                    if _cov2_disp is None or _cov2_disp.empty:
                        st.info("Re-run Run 2 to populate the blocking explorer.")
                    else:
                        _r2_toggles_src = run2["run_config"]["blocking_toggles"]
                        if "r2_exp_toggles" not in st.session_state:
                            st.session_state["r2_exp_toggles"] = dict(_r2_toggles_src)

                        # ── Blocking cascade waterfall (live) ───────────────────────
                        render_waterfall_section(
                            _cov2_disp, st.session_state["r2_exp_toggles"],
                            run2["run_config"].get("blocking_mode", "OR"),
                            key_prefix="r2_waterfall", run_label="Run 2",
                        )

                        _ecol_rules, _ecol_table = st.columns([1, 2.5], gap="large")
                        with _ecol_rules:
                            st.markdown("**Blocking Rules**")
                            _esa, _eca = st.columns(2)
                            if _esa.button("Select All", key="r2_exp_all"):
                                for _f in st.session_state["r2_exp_toggles"]:
                                    st.session_state["r2_exp_toggles"][_f] = True
                                st.rerun()
                            if _eca.button("Clear All", key="r2_exp_none"):
                                for _f in st.session_state["r2_exp_toggles"]:
                                    st.session_state["r2_exp_toggles"][_f] = False
                                st.rerun()
                            _r2_count_map = {
                                r["rule_sql"]: r["n"]
                                for r in run2.get("blocking_counts", [])
                            }
                            _new_r2_exp = {}
                            for _ef, _ev in st.session_state["r2_exp_toggles"].items():
                                with st.container(border=True):
                                    _etc, _eic = st.columns([1, 3])
                                    _nv = _etc.toggle("", value=_ev, key=f"r2_exp_tog_{_ef}")
                                    _new_r2_exp[_ef] = _nv
                                    _esql = f'l."{_ef}" = r."{_ef}"'
                                    _en   = _r2_count_map.get(_esql, 0)
                                    _eic.markdown(f"**{_ef}**")
                                    _eic.code(_esql, language="sql")
                                    _ebadge = "ACTIVE" if _nv else "INACTIVE"
                                    _ecolor = "green" if _nv else "grey"
                                    _eic.markdown(
                                        f'<span style="color:{_ecolor};font-weight:bold;font-size:11px">'
                                        f'{_ebadge}</span>&nbsp;&nbsp;<span style="font-size:11px">'
                                        f'{_en:,} pairs</span>',
                                        unsafe_allow_html=True,
                                    )
                            if _new_r2_exp != st.session_state["r2_exp_toggles"]:
                                st.session_state["r2_exp_toggles"] = _new_r2_exp

                        with _ecol_table:
                            _r2_filt = _fpar2(run2["df_predict"], _cov2_disp,
                                              st.session_state["r2_exp_toggles"])
                            _r2_norig = len(run2["df_predict"])
                            _r2_nfilt = len(_r2_filt)
                            _r2_red   = (1 - _r2_nfilt / _r2_norig) * 100 if _r2_norig > 0 else 0
                            _r2_nact  = sum(1 for v in st.session_state["r2_exp_toggles"].values() if v)
                            hs1, hs2, hs3, hs4 = st.columns(4)
                            hs1.metric("Candidate Pairs", f"{_r2_nfilt:,}")
                            hs2.metric("Rules Enabled",   f"{_r2_nact}/{len(st.session_state['r2_exp_toggles'])}")
                            hs3.metric("Reduction Ratio", f"{_r2_red:.1f}%")
                            hs4.metric("Original Pairs",  f"{_r2_norig:,}")
                            st.write("**Pairwise Edge Table**")
                            if _r2_filt.empty:
                                st.warning("No pairs covered by the current active rules.")
                            else:
                                _id_c   = [c for c in ["unique_id_l","unique_id_r",
                                            "source_dataset_l","source_dataset_r"] if c in _r2_filt.columns]
                                _rl_c   = ["effective_rule"] if "effective_rule" in _r2_filt.columns else []
                                _sc_c   = [c for c in ["match_probability","match_weight"] if c in _r2_filt.columns]
                                _gm_c   = [c for c in _r2_filt.columns if c.startswith("gamma_")][:4]
                                _disp_c = _id_c + _rl_c + _sc_c + _gm_c
                                _disp   = _r2_filt[_disp_c].head(200).copy()
                                if "match_probability" in _disp.columns:
                                    st.dataframe(
                                        _disp.style.background_gradient(
                                            subset=["match_probability"], cmap="RdYlGn", vmin=0, vmax=1
                                        ),
                                        use_container_width=True, height=360,
                                    )
                                else:
                                    st.dataframe(_disp, use_container_width=True, height=360)
                                st.caption(f"Showing up to 200 of {_r2_nfilt:,} filtered pairs.")

                        # ── Real-time Run 1 vs Run 2 demographic comparison ─────────
                        # Live proxy cohort from records touched by the currently
                        # active Run 2 rules — updates instantly on every toggle,
                        # no re-clustering required. Same helper used in the
                        # within-run tab above, so behaviour is identical across
                        # dummy dataset, NC voter registry, Upload, and Advanced.
                        st.divider()
                        try:
                            _r2_baseline_snap = compute_demographic_snapshot(run1.get("df_cluster", pd.DataFrame()))
                            _r2_live_cohort   = cohort_from_edges(
                                st.session_state.get("fakea"), st.session_state.get("fakeb"), _r2_filt
                            )
                            _r2_current_snap  = compute_demographic_snapshot(_r2_live_cohort)
                            render_demographic_comparison(
                                _r2_baseline_snap, _r2_current_snap,
                                baseline_label="Run 1", current_label="Run 2 (toggled, live)",
                                key_prefix="r2_demo_cmp",
                            )
                        except Exception as _r2_demo_err:
                            st.warning(f"Could not compute live demographic comparison: {_r2_demo_err}")

                        st.divider()
                        _r2_exp_thresh = st.slider(
                            "Cluster threshold (explorer)", 0.5, 0.99,
                            st.session_state.get("r2_exp_threshold", 0.8), 0.01,
                            key="r2_exp_thresh_slider",
                        )
                        st.session_state["r2_exp_threshold"] = _r2_exp_thresh

                        # ── Match quality by demographic group (real-time) ─────────
                        st.divider()
                        try:
                            render_match_quality_section(
                                run1["df_predict"], _r2_filt,
                                run1["run_config"].get("selected_fields", []),
                                run2["run_config"].get("selected_fields", []),
                                threshold=_r2_exp_thresh,
                                baseline_label="Run 1", current_label="Run 2 (toggled, live)",
                                key_prefix="r2_match_quality",
                            )
                        except Exception as _r2_mq_err:
                            st.warning(f"Could not compute match-quality comparison: {_r2_mq_err}")

                        # ── Match quality by demographic group (real-time) ─────────
                        st.divider()
                        try:
                            render_match_quality_section(
                                run1["df_predict"], _r2_filt,
                                run1["run_config"].get("selected_fields", []),
                                run2["run_config"].get("selected_fields", []),
                                threshold=_r2_exp_thresh,
                                baseline_label="Run 1", current_label="Run 2 (toggled, live)",
                                key_prefix="r2_match_quality",
                            )
                        except Exception as _r2_mq_err:
                            st.warning(f"Could not compute match-quality comparison: {_r2_mq_err}")
                        if st.button("Re-cluster with active rules (Run 2)", type="primary",
                                     key="r2_exp_recluster"):
                            if _r2_filt.empty:
                                st.warning("No pairs to cluster.")
                            else:
                                with st.spinner("Re-clustering…"):
                                    try:
                                        _r2_new_cl = _rcf2(
                                            _r2_filt,
                                            st.session_state["fakea"],
                                            st.session_state.get("fakeb"),
                                            threshold=_r2_exp_thresh,
                                        )
                                        if not _r2_new_cl.empty:
                                            _r2_nnc = _r2_new_cl["cluster_id"].nunique()
                                            st.success(f"Re-clustered: {_r2_nnc:,} clusters.")
                                            _rcc1, _rcc2 = st.columns(2)
                                            _rcc1.metric("Clusters (Run 2 original)", f"{m2['n_clusters']:,}")
                                            _rcc2.metric("Clusters (explorer)", f"{_r2_nnc:,}",
                                                         delta=f"{_r2_nnc - m2['n_clusters']:+,}")
                                        else:
                                            st.info("Re-clustering returned no clusters.")
                                    except Exception as _re:
                                        st.error(f"Re-clustering failed: {_re}")

                # ── Cluster Studio (Run 2) ─────────────────────────────────────
                with r2_sub_studio:
                    st.subheader("Splink Cluster Studio — Run 2")
                    st.write(
                        "Interactive visualisation of Run 2 entity clusters. "
                        "Each node is a record; edges are predicted matches."
                    )
                    _r2_html = run2.get("cluster_html", "")
                    _r2_n_edges = m2.get("n_edges", 0)
                    if not _r2_html:
                        st.info("Cluster studio HTML could not be generated for Run 2.")
                    elif _r2_n_edges == 0:
                        st.warning("No predicted edges — the cluster studio has nothing to visualise.")
                    else:
                        try:
                            components.html(_r2_html, height=650, scrolling=True)
                        except Exception as _sce:
                            st.warning(f"Cluster studio render error: {_sce}")

                # ── Confusion Matrix (Run 2) ───────────────────────────────────
                with r2_sub_cm:
                    st.subheader("Confusion Matrix — Run 2")
                    st.write(
                        "Ground truth: the 'cluster' column in the original datasets. "
                        "Records sharing the same cluster value are true matches."
                    )
                    _cm2_disp = st.session_state.get("run2_cm", {})
                    _ts2_disp = st.session_state.get("run2_ts")
                    _crl2_disp = st.session_state.get("run2_crl", {})

                    if not _cm2_disp:
                        st.info("Run 2 not yet complete. Click 'Run full analysis' above.")
                    elif _cm2_disp.get("unavailable"):
                        st.info(_cm2_disp.get("unavailable_reason", "Confusion matrix not available."))
                    elif "error" in _cm2_disp:
                        st.info(f"Confusion matrix error: {_cm2_disp['error']}")
                    else:
                        kc1, kc2, kc3, kc4 = st.columns(4)
                        kc1.metric("True Positives (TP)",  f"{_cm2_disp.get('tp',0):,}")
                        kc2.metric("False Positives (FP)", f"{_cm2_disp.get('fp',0):,}")
                        kc3.metric("False Negatives (FN)", f"{_cm2_disp.get('fn',0):,}")
                        kc4.metric("Ground truth pairs",   f"{_cm2_disp.get('n_gt_edges',0):,}")
                        st.divider()
                        _mc1, _mc2 = st.columns(2)
                        with _mc1:
                            st.write("**Derived Metrics**")
                            st.dataframe(pd.DataFrame([
                                {"Metric":"Precision","Value":f"{_cm2_disp.get('precision',0):.4f}","Meaning":"TP/(TP+FP)"},
                                {"Metric":"Recall",   "Value":f"{_cm2_disp.get('recall',0):.4f}",   "Meaning":"TP/(TP+FN)"},
                                {"Metric":"F1 Score", "Value":f"{_cm2_disp.get('f1',0):.4f}",       "Meaning":"Harmonic mean"},
                                {"Metric":"F* Score", "Value":f"{_cm2_disp.get('fstar',0):.4f}",    "Meaning":"TP/(TP+FP+FN)"},
                                {"Metric":"FDR",      "Value":f"{_cm2_disp.get('fdr',0):.4f}",      "Meaning":"False Discovery Rate"},
                                {"Metric":"FNR",      "Value":f"{_cm2_disp.get('fnr',0):.4f}",      "Meaning":"False Negative Rate"},
                            ]), use_container_width=True, hide_index=True)
                        with _mc2:
                            st.write("**Confusion Matrix**")
                            _z2   = [[_cm2_disp.get("tp",0), _cm2_disp.get("fp",0)],
                                     [_cm2_disp.get("fn",0), 0]]
                            _txt2 = [[f"TP<br>{_cm2_disp.get('tp',0):,}", f"FP<br>{_cm2_disp.get('fp',0):,}"],
                                     [f"FN<br>{_cm2_disp.get('fn',0):,}", "TN<br>(omitted)"]]
                            _fcm2 = gobj.Figure(data=gobj.Heatmap(
                                z=_z2, text=_txt2, texttemplate="%{text}",
                                colorscale=[[0,"#B85050"],[0.5,"#CCCCCC"],[1,"#1d8a50"]],
                                showscale=False,
                            ))
                            _fcm2.update_layout(
                                xaxis=dict(tickvals=[0,1], ticktext=["Predicted Match","Predicted Non-Match"]),
                                yaxis=dict(tickvals=[0,1], ticktext=["True Non-Match","True Match"],
                                           autorange="reversed"),
                                height=260, margin=dict(l=10,r=10,t=30,b=10),
                                title="Pairwise Confusion Matrix (Run 2)",
                            )
                            st.plotly_chart(_fcm2, use_container_width=True)

                    if _ts2_disp is not None and not _ts2_disp.empty:
                        st.divider()
                        st.subheader("Precision-Recall Curve — Run 2")
                        _p1, _p2 = st.columns(2)
                        _ts2_pr = _ts2_disp.dropna(subset=["precision_val","recall_val"])
                        if not _ts2_pr.empty:
                            with _p1:
                                _fpr2 = px.line(_ts2_pr, x="recall_val", y="precision_val",
                                                title="Precision-Recall Curve",
                                                template="simple_white",
                                                color_discrete_sequence=["#1E6EC4"])
                                _fpr2.update_layout(height=300, xaxis_range=[0,1], yaxis_range=[0,1.05])
                                st.plotly_chart(_fpr2, use_container_width=True)
                        _ts2_fs = _ts2_disp.dropna(subset=["fstar","match_probability"])
                        if not _ts2_fs.empty:
                            with _p2:
                                _ffs2 = px.line(_ts2_fs, x="match_probability", y="fstar",
                                                title="F* Score vs Threshold",
                                                template="simple_white",
                                                color_discrete_sequence=["#28A060"])
                                _ffs2.update_layout(height=300, xaxis_range=[0,1], yaxis_range=[0,1.05])
                                st.plotly_chart(_ffs2, use_container_width=True)
                        if _crl2_disp.get("crl_score") is not None:
                            _cr1,_cr2,_cr3,_cr4 = st.columns(4)
                            _cr1.metric("CRL Score", f"{_crl2_disp.get('crl_score',0):.6f}")
                            _cr2.metric("t_upper",   str(_crl2_disp.get("t_upper","N/A")))
                            _cr3.metric("t_lower",   str(_crl2_disp.get("t_lower","N/A")))
                            _cr4.metric("epsilon_z", str(_crl2_disp.get("epsilon_z","N/A")))

                # ── Raw Data (Run 2) ───────────────────────────────────────────
                with r2_sub_data:
                    st.subheader("Raw Tables — Run 2")
                    st.write("**df_predict (first 100 rows)**")
                    st.dataframe(run2["df_predict"].head(100), use_container_width=True)
                    st.caption(
                        "Each row is a candidate record pair. "
                        "gamma_ columns show field-level agreement. "
                        "match_key indicates which blocking rule generated this pair."
                    )
                    st.write("**df_cluster (first 100 rows)**")
                    st.dataframe(run2["df_cluster"].head(100), use_container_width=True)

            # ── Run 1 vs Run 2 Comparison ─────────────────────────────────────
            with r2_tab_compare:
                inter = compute_inter_metrics(
                    run1["df_predict"], run2["df_predict"],
                    run1["df_cluster"], run2["df_cluster"],
                )

                st.subheader("Comparison: Run 1 vs Run 2")
                mp1 = (m1["match_prob_stats"]["mean_match_prob"].iloc[0]
                       if not m1["match_prob_stats"].empty else 0)
                mp2 = (m2["match_prob_stats"]["mean_match_prob"].iloc[0]
                       if not m2["match_prob_stats"].empty else 0)

                kc1, kc2, kc3 = st.columns(3)
                kc1.metric("Edges", f"{m2['n_edges']:,}",
                           delta=f"{m2['n_edges'] - m1['n_edges']:+,}")
                kc2.metric("Clusters", f"{m2['n_clusters']:,}",
                           delta=f"{m2['n_clusters'] - m1['n_clusters']:+,}")
                kc3.metric("Mean match prob", f"{mp2:.4f}",
                           delta=f"{mp2 - mp1:+.4f}")

                st.divider()
                ed = inter.get("edge_diff", pd.DataFrame())
                if not ed.empty:
                    st.write("**Edge Changes Between Runs**")
                    ed_d = ed.set_index("category")["n"].to_dict()
                    st.dataframe(pd.DataFrame([
                        {"Metric": "Shared edges (both runs)",    "Count": ed_d.get("shared", 0)},
                        {"Metric": "Edges added in Run 2",        "Count": ed_d.get("added", 0)},
                        {"Metric": "Edges removed in Run 2",      "Count": ed_d.get("removed", 0)},
                        {"Metric": "Exact matching clusters",     "Count": inter.get("n_exact_matching_clusters", 0)},
                        {"Metric": "Partially matching clusters", "Count": inter.get("n_partial_matching_clusters", 0)},
                    ]), use_container_width=True, hide_index=True)

                pd1 = inter.get("prob_dist_run1", pd.DataFrame())
                pd2 = inter.get("prob_dist_run2", pd.DataFrame())
                if not pd1.empty and not pd2.empty:
                    pd1["run"] = "Run 1"; pd2["run"] = "Run 2"
                    fig = px.bar(pd.concat([pd1, pd2]), x="prob_bin", y="n_edges",
                                 color="run", barmode="group",
                                 title="Match Probability Distribution — Run 1 vs Run 2",
                                 template="simple_white",
                                 color_discrete_sequence=["#1E6EC4", "#E55C30"])
                    fig.update_layout(height=340)
                    st.plotly_chart(fig, use_container_width=True)

                cs1 = inter.get("cluster_sizes_run1", pd.DataFrame())
                cs2c = inter.get("cluster_sizes_run2", pd.DataFrame())
                if not cs1.empty and not cs2c.empty:
                    cs1["run"] = "Run 1"; cs2c["run"] = "Run 2"
                    fig2 = px.bar(pd.concat([cs1, cs2c]), x="n_nodes", y="n_clusters",
                                  color="run", barmode="group",
                                  title="Cluster Size Distribution — Run 1 vs Run 2",
                                  template="simple_white",
                                  color_discrete_sequence=["#1E6EC4", "#E55C30"])
                    fig2.update_layout(height=340)
                    st.plotly_chart(fig2, use_container_width=True)

                st.divider()
                if st.button("Generate PDF report for Run 2"):
                    with st.spinner("Generating…"):
                        try:
                            pdf2 = generate_report(
                                run_label="Run 2", run_config=run2["run_config"], metrics=m2,
                                n_input_records=run2["n_input_records"],
                                model_params=run2.get("model_params", {}),
                                missingness_a=run2.get("missingness_a", {}),
                                missingness_b=run2.get("missingness_b", {}),
                                blocking_counts=run2.get("blocking_counts", []),
                                unlinkables=run2.get("unlinkables", {}),
                                settings_used=run2.get("settings_used", {}),
                            )
                            st.download_button("Download Run 2 PDF", data=pdf2,
                                               file_name=f"linkage_report_run2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                               mime="application/pdf")
                        except Exception as e:
                            st.error(f"PDF failed: {e}")

    st.divider()
    if st.button("Continue to export", type="primary"):
        _go_to(6)


def page_export():
    _back_button()
    st.title("Step 7: Export Cohort")

    if st.session_state.get("run1_results") is None:
        st.warning("No analysis results available. Please complete the analysis first.")
        if st.button("Go to analysis"):
            _go_to(4)
        return

    st.write(
        "Download the final cohort as a CSV. The output contains all original "
        "record fields plus a cluster_id column. Records sharing the same "
        "cluster_id are predicted to represent the same real-world individual."
    )
    st.divider()

    st.subheader("Select which run to export")
    run_opts = ["Run 1"]
    if st.session_state.get("run2_results") is not None:
        run_opts.append("Run 2")
    else:
        st.caption("Run 2 is not available. Complete a comparison run to add it as an option.")

    selected_run = st.radio(
        "Export cluster assignments from:", run_opts, horizontal=True
    )
    chosen = (
        st.session_state["run1_results"]
        if selected_run == "Run 1"
        else st.session_state["run2_results"]
    )

    with st.expander("🧹 Free up memory (optional)", expanded=False):
        st.write(
            "The raw pairs table (df_predict) is the largest object kept in "
            "memory and isn't needed for export, the PDF report, or "
            "demographic views (only df_cluster is). Clearing it here frees "
            "that memory but disables the Blocking Explorer for that run "
            "until you re-run the analysis."
        )
        fc1, fc2 = st.columns(2)
        _r1 = st.session_state.get("run1_results")
        _r2 = st.session_state.get("run2_results")
        with fc1:
            if _r1 is not None and _r1.get("df_predict") is not None:
                if st.button("Free Run 1's raw pairs table"):
                    st.session_state["run1_results"]["df_predict"] = None
                    st.success("Run 1's df_predict cleared.")
            else:
                st.caption("Run 1: already cleared or unavailable.")
        with fc2:
            if _r2 is not None and _r2.get("df_predict") is not None:
                if st.button("Free Run 2's raw pairs table"):
                    st.session_state["run2_results"]["df_predict"] = None
                    st.success("Run 2's df_predict cleared.")
            else:
                st.caption("Run 2: not available or already cleared.")

    st.divider()

    df_cluster = chosen["df_cluster"]
    op_mode    = chosen["run_config"]["operation_mode"]

    if op_mode == "dedupe":
        raw = st.session_state["fakea"].copy()
    else:
        raw = pd.concat(
            [st.session_state["fakea"], st.session_state["fakeb"]],
            ignore_index=True,
        )

    merge_keys = (
        ["unique_id", "source_dataset"]
        if "source_dataset" in df_cluster.columns
        else ["unique_id"]
    )
    cohort = raw.merge(
        df_cluster[merge_keys + ["cluster_id"]],
        on=merge_keys, how="left",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records",          f"{len(cohort):,}")
    c2.metric("Distinct cluster IDs",   f"{cohort['cluster_id'].nunique():,}")
    c3.metric("Records with cluster",   f"{cohort['cluster_id'].notna().sum():,}")

    st.subheader("Cohort preview (first 50 rows, sorted by cluster_id)")
    st.dataframe(cohort.sort_values("cluster_id").head(50),
                 use_container_width=True)

    st.divider()
    csv_bytes = cohort.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download cohort CSV ({selected_run})",
        data=csv_bytes,
        file_name=f"cohort_{selected_run.lower().replace(' ','_')}.csv",
        mime="text/csv",
    )

    st.info(
        "Cohort CSV downloaded. Future versions will support direct databank integration."
    )