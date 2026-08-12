# pages/p_analysis.py
# Analysis page — shared by standard, upload, and advanced flows.
# Shows data summary, run button, tabbed results (edge metrics, cluster metrics,
# demographics, blocking explorer, cluster studio, confusion matrix, raw data).
# Also provides the model JSON and PDF download buttons.

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as gobj
import streamlit as st
import streamlit.components.v1 as components

from modules.splink_runner import (
    filter_predict_by_active_rules, recluster_filtered, reconstruct_model_json,
    determine_cascade_order,
)
from modules.report_gen import generate_report, draw_venn_diagram
from utils.helpers import (
    _metric_cards, _plotly_bar, _run_analysis_and_store, render_demographic_breakdowns,
    render_demographic_comparison, compute_demographic_snapshot, cohort_from_edges,
    render_match_quality_section, split_linked_unlinked,
    render_blocking_waterfall, render_waterfall_section,
)
from utils.nav import _back_button, _go_to


def page_analysis():
    _back_button()
    flow = st.session_state.get("flow", "standard")

    # ── Guard: ensure required state is present ───────────────────────────────
    if flow == "standard":
        if not st.session_state["dataset_ready"]:
            st.warning("No dataset loaded. Go back to Step 1.")
            if st.button("Go to Step 1"):
                _go_to(0)
            return
        if st.session_state.get("operation_mode") is None:
            st.warning("Operation mode not set. Go back to Step 3.")
            if st.button("Go to Step 3"):
                _go_to(2)
            return
        if st.session_state.get("linkage_type") is None:
            st.warning("Linkage type not set. Go back to Step 4.")
            if st.button("Go to Step 4"):
                _go_to(3)
            return

    if flow == "advanced":
        st.title("Analysis (Advanced Flow)")
    elif flow == "upload":
        st.title("Step 5: Run Analysis (Upload Data)")
    else:
        st.title("Step 5: Run Analysis")

    # ── Dataset-specific warnings ─────────────────────────────────────────────
    _linkage_type = st.session_state.get("linkage_type")
    _is_nc = (st.session_state.get("nc_field_types") is not None
              and st.session_state.get("flow") == "standard")

    if _is_nc:
        st.warning(
            "⚠️ **North Carolina voter dataset detected.** "
            "This dataset is large and may cause high memory usage during analysis. "
            "Streamlit has memory limits and the app may crash on very large runs. "
            "Recommendation: use a small sample, select fewer comparison fields "
            "to reduce memory pressure."
        )

    if _linkage_type == "deterministic":
        st.info(
            "ℹ️ **Deterministic mode** generates all candidate pairs matching any "
            "blocking rule and assigns each pair match_probability = 1.0. "
            "With low-cardinality blocking fields (e.g. gender, county) this can "
            "produce a very large number of pairs and **may take significantly longer** "
            "than probabilistic mode. Consider using high-selectivity fields "
            "(e.g. voter_reg_num, ncid, last_name) as blocking rules."
        )

    # ══════════════════════════════════════════════════════════════════════
    # UPLOAD FLOW: show what dataset / fields are actually loaded and
    # validate them against the real data before allowing a run.
    # ══════════════════════════════════════════════════════════════════════
    if flow == "upload":
        _uf_fakea = st.session_state.get("fakea")
        _uf_sel   = st.session_state.get("selected_fields", [])
        _uf_block = st.session_state.get("blocking_toggles", {})

        if _uf_fakea is None:
            st.error(
                "No uploaded dataset is loaded. "
                "Complete the upload flow (upload file → EDA → configure fields) first."
            )
            if st.button("Go to Upload", key="uf_goto_upload"):
                _go_to("upload_setup")
            return

        # Compute which selected fields actually exist in the data
        _uf_actual   = set(_uf_fakea.columns)
        _uf_valid    = [f for f in _uf_sel if f in _uf_actual]
        _uf_missing  = [f for f in _uf_sel if f not in _uf_actual]

        if not _uf_valid:
            # No valid fields at all — redirect to configure
            st.error(
                "None of the configured comparison fields exist in the uploaded dataset. "
                "Go back to Configure Fields and select fields that match your uploaded data."
            )
            c_err1, c_err2 = st.columns(2)
            c_err1.write(f"**Configured fields:** {', '.join(_uf_sel) or 'none'}")
            c_err2.write(f"**Dataset columns:** {', '.join(sorted(_uf_actual)[:10])}{'...' if len(_uf_actual)>10 else ''}")
            if st.button("Go to Configure Fields", key="uf_goto_cfg"):
                _go_to("upload_configure")
            return

        # Data summary box — always visible so user can see what's loaded
        with st.container(border=True):
            st.markdown("**Uploaded dataset loaded**")
            ds1, ds2, ds3, ds4 = st.columns(4)
            ds1.metric("Rows (Dataset A)",   f"{len(_uf_fakea):,}")
            ds2.metric("Columns",            f"{_uf_fakea.shape[1]:,}")
            ds3.metric("Fields for linkage", f"{len(_uf_valid)}")
            _uf_fb = st.session_state.get("fakeb")
            ds4.metric("Dataset B rows",
                       f"{len(_uf_fb):,}" if _uf_fb is not None else "None")

            st.caption(
                f"**Comparison fields:** {', '.join(_uf_valid)}"
            )
            if _uf_missing:
                st.warning(
                    f"These configured fields are NOT in the dataset and will be skipped: "
                    f"{', '.join(_uf_missing)}"
                )
            _uf_active_block = [f for f, v in _uf_block.items() if v and f in _uf_actual]
            st.caption(f"**Active blocking rules:** {', '.join(_uf_active_block) or 'none'}")

            if not _uf_active_block:
                st.error(
                    "No active blocking rules match columns in the dataset. "
                    "Go back to Configure Fields."
                )
                if st.button("Go to Configure Fields", key="uf_cfg_block"):
                    _go_to("upload_configure")
                return

        # If cached results exist from a different run (e.g. standard flow),
        # make it very clear the user needs to click Run to get upload results
        _cached = st.session_state.get("run1_results")
        if _cached:
            _prev_fields = _cached.get("run_config", {}).get("selected_fields", [])
            if set(_prev_fields) != set(_uf_valid):
                st.info(
                    "The results shown below are from a **previous run** with different "
                    "fields or data. Click **Run analysis** to run with your uploaded dataset."
                )

    # ── Configuration summary (previous run) ─────────────────────────────────
    run_results = st.session_state.get("run1_results")
    if run_results and flow != "upload":
        rc = run_results["run_config"]
        with st.expander("Run configuration", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Operation:** {rc.get('operation_mode','').replace('_',' ').title()}")
            c2.write(f"**Linkage:** {rc.get('linkage_type','').title()}")
            c3.write(f"**Fields:** {', '.join(rc.get('selected_fields',[]))}")
            if rc.get("from_json"):
                st.info("Results produced from uploaded model JSON.")

    # ── Run / re-run button ───────────────────────────────────────────────────
    if flow in ("standard", "upload"):
        run_label = (
            "Run analysis"
            if run_results is None
            else "Re-run analysis with current settings"
        )
        # For upload flow always label it clearly
        if flow == "upload":
            run_label = f"Run analysis on uploaded dataset ({len(st.session_state['fakea']):,} rows)"

        if st.button(run_label, type="primary"):
            with st.spinner(
                "Running model. Probabilistic training may take 1-2 minutes..."
            ):
                ok = _run_analysis_and_store(
                    fakea=st.session_state["fakea"],
                    fakeb=st.session_state["fakeb"],
                    selected_fields=st.session_state["selected_fields"],
                    blocking_toggles=st.session_state["blocking_toggles"],
                    operation_mode=st.session_state["operation_mode"],
                    linkage_type=st.session_state["linkage_type"],
                    hyperparams=st.session_state.get("hyperparams", {}),
                    composite_rules=st.session_state.get("composite_rules", {}),
                )
                if ok:
                    st.success("Analysis complete.")

    if st.session_state.get("run1_results") is None:
        return

    results = st.session_state["run1_results"]
    metrics = st.session_state.get("run1_metrics", {})

    if results.get("n_edges", 0) == 0:
        _diag = results.get("zero_edge_diagnostic") or []
        st.error(
            "This run produced **0 candidate pairs** — every record ended up in "
            "its own singleton cluster. Below is exactly why, field by field."
        )
        if _diag:
            st.dataframe(pd.DataFrame(_diag), use_container_width=True, hide_index=True)
            _bad = [d["field"] for d in _diag if d.get("issue")]
            if _bad:
                st.caption(
                    f"Field(s) with no repeated values at all: {', '.join(_bad)}. "
                    "Disable these as blocking rules, or check the Dataset Profile "
                    "step to confirm they weren't corrupted during cleaning."
                )
        st.divider()

    # ── KPI headline row ──────────────────────────────────────────────────────
    st.subheader("Summary")
    _metric_cards([
        ("Records processed",         f"{results['n_input_records']:,}"),
        ("Predicted edges (matches)", f"{metrics['n_edges']:,}"),
        ("Distinct entity clusters",  f"{metrics['n_clusters']:,}"),
        ("Unique IDs with a match",   f"{metrics['n_unique_ids']:,}"),
    ])

    st.divider()

    # ── Tabbed results ────────────────────────────────────────────────────────
    (tab_edges, tab_clusters, tab_demo,
     tab_explorer, tab_studio, tab_cm, tab_data) = st.tabs([
        "Edge Metrics",
        "Cluster Metrics",
        "Demographics",
        "Blocking Explorer",    # NEW interactive explorer tab
        "Cluster Studio",
        "Confusion Matrix",
        "Raw Data",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Edge Metrics
    # ═══════════════════════════════════════════════════════════════════════
    with tab_edges:
        st.subheader("Edge Metrics")
        lt = results["run_config"]["linkage_type"]

        prob_stats = metrics.get("match_prob_stats", pd.DataFrame())
        if not prob_stats.empty:
            st.write("**Match Probability Statistics**")
            st.dataframe(prob_stats, use_container_width=True)

        prob_dist = metrics.get("prob_dist", pd.DataFrame())
        if not prob_dist.empty and len(prob_dist) > 1:
            st.plotly_chart(
                _plotly_bar(prob_dist, "prob_bin", "n_edges",
                            "Match Probability Distribution"),
                use_container_width=True,
            )
            st.caption(
                "Bars near 1.0 indicate confident predictions. "
                "Bars spread across mid-range indicate uncertain predictions."
            )

        weight_dist = metrics.get("weight_dist", pd.DataFrame())
        if not weight_dist.empty and len(weight_dist) > 1:
            st.plotly_chart(
                _plotly_bar(weight_dist, "weight_bin", "n_edges",
                            "Match Weight Histogram", "#E55C30"),
                use_container_width=True,
            )
            st.caption(
                "Match weight = log2(m/u). Positive values = more likely a match. "
                "Higher values = greater confidence."
            )

        gamma_df = metrics.get("gamma_means", pd.DataFrame())
        if not gamma_df.empty and lt == "probabilistic":
            g_long = gamma_df.T.reset_index()
            g_long.columns = ["field", "mean_gamma"]
            g_long["field"] = g_long["field"].str.replace("gamma_", "", regex=False)
            st.plotly_chart(
                _plotly_bar(g_long, "field", "mean_gamma",
                            "Mean Gamma Score per Field", "#2ECC71"),
                use_container_width=True,
            )
            st.caption(
                "Gamma = 1: exact agreement. Gamma = 0: total disagreement. "
                "High mean gamma means matched pairs agree on this field."
            )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Cluster Metrics
    # ═══════════════════════════════════════════════════════════════════════
    with tab_clusters:
        st.subheader("Cluster Metrics")
        c1, c2 = st.columns(2)
        c1.metric("Total clusters", f"{metrics['n_clusters']:,}")
        c2.metric("Cross-dataset clusters", f"{metrics['n_cross_dataset']:,}")

        s = metrics.get("singleton_stats", pd.DataFrame())
        if not s.empty:
            st.write("**Singleton vs Multi-record Clusters**")
            _sc1, _sc2 = st.columns([1, 1])
            with _sc1:
                st.dataframe(s, use_container_width=True, hide_index=True)
                st.caption(
                    "High singleton count = many records could not be linked. "
                    "Multi-record clusters = found duplicates / cross-dataset matches."
                )
            with _sc2:
                if "cluster_type" in s.columns and "n_clusters" in s.columns:
                    _s_fig = px.bar(
                        s, x="cluster_type", y="n_clusters",
                        color="cluster_type",
                        color_discrete_sequence=["#1E6EC4", "#E55C30"],
                        title="Singleton vs Multi-record",
                        template="simple_white",
                        labels={"cluster_type": "", "n_clusters": "Number of clusters"},
                    )
                    _s_fig.update_layout(
                        height=280, showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(_s_fig, use_container_width=True)

        cs = metrics.get("cluster_sizes", pd.DataFrame())
        if not cs.empty:
            st.plotly_chart(
                _plotly_bar(cs, "n_nodes", "n_clusters", "Cluster Size Distribution"),
                use_container_width=True,
            )
            st.caption(
                "A J-shaped curve (many size-1, few large clusters) is typical. "
                "Very large clusters may indicate over-linking."
            )

        venn = metrics.get("venn", {})
        op   = results["run_config"]["operation_mode"]
        if op != "dedupe" and any(venn.values()):
            st.write("**Dataset Overlap in Clusters**")
            a_only  = venn.get("a_only",  0)
            b_only  = venn.get("b_only",  0)
            both_ab = venn.get("both_ab", 0)
            vdf = pd.DataFrame([
                {"Category": "Dataset A only", "N Clusters": a_only},
                {"Category": "Both A and B",   "N Clusters": both_ab},
                {"Category": "Dataset B only", "N Clusters": b_only},
            ])
            st.dataframe(vdf, use_container_width=True, hide_index=True)

            # ── Venn diagram (shared drawing function — see modules/report_gen.py) ──
            _vfig = draw_venn_diagram(
                a_only, both_ab, b_only,
                figsize=(3, 2.1), title_fontsize=8, number_fontsize=10, label_fontsize=6,
                title="Venn Diagram — Cluster_id Set Membership",
            )
            _vcol_l, _vcol_mid, _vcol_r = st.columns([1, 1, 1])
            with _vcol_mid:
                st.pyplot(_vfig, use_container_width=False)
            import matplotlib.pyplot as _plt
            _plt.close(_vfig)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Demographics
    # ═══════════════════════════════════════════════════════════════════════
    with tab_demo:
        st.subheader("Demographic Breakdown")
        g = metrics.get("gender_dist", pd.DataFrame())
        c = metrics.get("city_dist",   pd.DataFrame())
        d1, d2 = st.columns(2)
        if not g.empty:
            with d1:
                st.plotly_chart(
                    px.pie(g, values="n_records", names="gender",
                           title="Gender Distribution in Clusters",
                           template="simple_white",
                           color_discrete_sequence=px.colors.qualitative.Set2),
                    use_container_width=True,
                )
        if not c.empty:
            with d2:
                st.plotly_chart(
                    _plotly_bar(c.head(10), "city", "n_records",
                                "Top 10 Cities in Clusters", "#9B59B6"),
                    use_container_width=True,
                )

        if metrics.get("nc_demographics"):
            st.divider()
            render_demographic_breakdowns(metrics["nc_demographics"], key_prefix="run1_demo")

        # ── Linked vs. Unlinked demographic comparison ──────────────────────
        # Does linkage outcome differ systematically by demographic group?
        # Complements (doesn't replace) the population/match-quality views
        # from the last two rounds — this answers a distinct question:
        # who ends up WITHOUT a match, and how does that group differ.
        st.divider()
        st.write("**Linked vs. Unlinked — Demographic Comparison**")
        try:
            _linked_df, _unlinked_df = split_linked_unlinked(results.get("df_cluster", pd.DataFrame()))
            _linked_snap   = compute_demographic_snapshot(_linked_df)
            _unlinked_snap = compute_demographic_snapshot(_unlinked_df)
            render_demographic_comparison(
                _linked_snap, _unlinked_snap,
                baseline_label="Linked", current_label="Unlinked",
                key_prefix="linked_unlinked",
            )
            st.caption(
                f"Linked: {len(_linked_df):,} records in clusters of size > 1. "
                f"Unlinked: {len(_unlinked_df):,} records in singleton clusters."
            )
        except Exception as _lu_err:
            st.warning(f"Could not compute linked vs unlinked comparison: {_lu_err}")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Interactive Blocking Explorer
    # Mirrors the design from the screen.png mockup:
    #   Left panel  – toggleable rule cards with pair counts
    #   Right panel – live df_predict table + headline stats
    # Toggling a rule updates the table in real time (Streamlit rerun).
    # "Re-cluster" button recomputes entity clusters from the filtered edges.
    # ═══════════════════════════════════════════════════════════════════════
    with tab_explorer:
        st.subheader("Interactive Blocking Explorer")
        st.write(
            "Toggle blocking rules on or off. The pairwise edge table updates "
            "to show only pairs covered by at least one active rule. "
            "If a pair is covered by multiple rules, it is kept and the "
            "'effective rule' column reflects the first active rule covering it. "
            "Click 'Re-cluster' to see how the cluster assignments change."
        )

        cov_matrix = st.session_state.get("coverage_matrix")
        if cov_matrix is None or cov_matrix.empty:
            st.info("Run an analysis first to enable the interactive explorer.")
        else:
            # Initialise explorer toggles from run config if not yet set
            run_toggles = results["run_config"].get("blocking_toggles", {})
            if not st.session_state.get("explorer_toggles"):
                st.session_state["explorer_toggles"] = dict(run_toggles)

            # ── Blocking cascade waterfall (live) ───────────────────────────────
            render_waterfall_section(
                cov_matrix, st.session_state["explorer_toggles"],
                results["run_config"].get("blocking_mode", "OR"),
                key_prefix="exp_waterfall",
            )

            # ── Two-column layout ─────────────────────────────────────────────
            col_rules, col_table = st.columns([1, 2.5], gap="large")

            with col_rules:
                st.markdown("**Blocking Rules**")

                # Select All / Clear All buttons
                sa, ca = st.columns(2)
                if sa.button("Select All", key="exp_all"):
                    for f in st.session_state["explorer_toggles"]:
                        st.session_state["explorer_toggles"][f] = True
                    st.rerun()
                if ca.button("Clear All", key="exp_none"):
                    for f in st.session_state["explorer_toggles"]:
                        st.session_state["explorer_toggles"][f] = False
                    st.rerun()

                # Count map: pairs originally generated by each rule
                count_map = {
                    r["rule_sql"]: r["n"]
                    for r in results.get("blocking_counts", [])
                }

                # Rule cards
                new_toggles = {}
                for field, currently_on in st.session_state["explorer_toggles"].items():
                    with st.container(border=True):
                        tc, ic = st.columns([1, 3])
                        new_val = tc.toggle(
                            "", value=currently_on, key=f"exp_tog_{field}"
                        )
                        new_toggles[field] = new_val
                        sql = f'l."{field}" = r."{field}"'
                        n   = count_map.get(sql, 0)
                        ic.markdown(f"**{field}**")
                        ic.code(sql, language="sql")
                        # ACTIVE / INACTIVE badge
                        badge = "ACTIVE" if new_val else "INACTIVE"
                        color = "green" if new_val else "grey"
                        ic.markdown(
                            f'<span style="color:{color};font-weight:bold;'
                            f'font-size:11px">{badge}</span>'
                            f'&nbsp;&nbsp;<span style="font-size:11px">'
                            f'{n:,} pairs</span>',
                            unsafe_allow_html=True,
                        )

                # Update explorer toggles if anything changed
                if new_toggles != st.session_state["explorer_toggles"]:
                    st.session_state["explorer_toggles"] = new_toggles

            with col_table:
                # ── Filter df_predict by active explorer rules ─────────────────
                filtered_df = filter_predict_by_active_rules(
                    results["df_predict"],
                    cov_matrix,
                    st.session_state["explorer_toggles"],
                )

                n_orig     = len(results["df_predict"])
                n_filtered = len(filtered_df)
                n_active   = sum(1 for v in st.session_state["explorer_toggles"].values() if v)
                reduction  = (1 - n_filtered / n_orig) * 100 if n_orig > 0 else 0

                # ── Headline stats ─────────────────────────────────────────────
                hs1, hs2, hs3, hs4 = st.columns(4)
                hs1.metric("Candidate Pairs",  f"{n_filtered:,}")
                hs2.metric("Rules Enabled",    f"{n_active}/{len(st.session_state['explorer_toggles'])}")
                hs3.metric("Reduction Ratio",  f"{reduction:.1f}%")
                hs4.metric("Original Pairs",   f"{n_orig:,}")

                # ── Pair table ─────────────────────────────────────────────────
                st.write("**Pairwise Edge Table**")
                if filtered_df.empty:
                    st.warning("No pairs covered by the current active rules.")
                else:
                    # Select display columns: IDs, effective rule, scores, key gammas
                    id_cols   = [c for c in ["unique_id_l","unique_id_r",
                                              "source_dataset_l","source_dataset_r"]
                                 if c in filtered_df.columns]
                    rule_cols = ["effective_rule"] if "effective_rule" in filtered_df.columns else []
                    score_cols= [c for c in ["match_probability","match_weight"]
                                 if c in filtered_df.columns]
                    gamma_cols= [c for c in filtered_df.columns
                                 if c.startswith("gamma_")][:4]   # show first 4 gammas max

                    display_cols = id_cols + rule_cols + score_cols + gamma_cols
                    display_df   = filtered_df[display_cols].head(200).copy()

                    # Colour-code match_probability: show as bar chart column
                    if "match_probability" in display_df.columns:
                        st.dataframe(
                            display_df.style.background_gradient(
                                subset=["match_probability"],
                                cmap="RdYlGn",
                                vmin=0, vmax=1,
                            ),
                            use_container_width=True,
                            height=360,
                        )
                    else:
                        st.dataframe(display_df, use_container_width=True, height=360)

                    st.caption(
                        f"Showing up to 200 of {n_filtered:,} filtered pairs. "
                        "match_probability is colour-coded: red = low confidence, "
                        "green = high confidence."
                    )

            # ── Match quality by demographic group (real-time) ─────────────────
            # Edge-level view — reacts immediately to rule toggles, unlike the
            # population-count charts, since disabling a rule directly removes
            # the edges it contributed.
            st.divider()
            try:
                _exp_fields = results["run_config"].get("selected_fields", [])
                render_match_quality_section(
                    results["df_predict"], filtered_df,
                    _exp_fields, _exp_fields,
                    threshold=exp_thresh if "exp_thresh" in dir() else st.session_state.get("explorer_threshold", 0.8),
                    baseline_label="Original rules", current_label="Toggled rules (live)",
                    key_prefix="exp_match_quality",
                )
            except Exception as _mq_err:
                st.warning(f"Could not compute match-quality comparison: {_mq_err}")

            # ── Re-cluster button ─────────────────────────────────────────────
            st.divider()
            exp_thresh = st.slider(
                "Cluster threshold for explorer",
                0.5, 0.99,
                st.session_state.get("explorer_threshold", 0.8),
                0.01,
                key="exp_thresh_slider",
            )
            st.session_state["explorer_threshold"] = exp_thresh

            if st.button("Re-cluster with active rules", type="primary"):
                if filtered_df.empty:
                    st.warning("No pairs to cluster.")
                else:
                    with st.spinner("Re-clustering..."):
                        try:
                            new_clusters = recluster_filtered(
                                df_predict_filtered=filtered_df,
                                fakea=st.session_state["fakea"],
                                fakeb=st.session_state.get("fakeb"),
                                threshold=exp_thresh,
                            )
                            if not new_clusters.empty:
                                new_n_clusters = new_clusters["cluster_id"].nunique()
                                st.success(
                                    f"Re-clustered: {new_n_clusters:,} clusters "
                                    f"from {n_filtered:,} filtered edges."
                                )
                                # Side-by-side comparison
                                rc1, rc2 = st.columns(2)
                                rc1.metric(
                                    "Clusters (original rules)",
                                    f"{metrics['n_clusters']:,}",
                                )
                                rc2.metric(
                                    "Clusters (explorer rules)",
                                    f"{new_n_clusters:,}",
                                    delta=f"{new_n_clusters - metrics['n_clusters']:+,}",
                                )
                            else:
                                st.info(
                                    "Re-clustering returned no clusters. "
                                    "Try lowering the threshold or enabling more rules."
                                )
                        except Exception as e:
                            st.error(f"Re-clustering failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Cluster Studio
    # ═══════════════════════════════════════════════════════════════════════
    with tab_studio:
        st.subheader("Splink Cluster Studio")
        st.write(
            "Interactive visualisation of entity clusters. Each node is a record; "
            "edges are predicted matches. Use this to visually inspect linkage quality."
        )
        html = results.get("cluster_html", "")
        if html:
            components.html(html, height=650, scrolling=True)
        else:
            st.info("Cluster studio HTML could not be generated for this run.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Confusion Matrix
    # ═══════════════════════════════════════════════════════════════════════
    with tab_cm:
        st.subheader("Confusion Matrix and Model Accuracy")
        st.write(
            "Ground truth: the 'cluster' column in the original datasets. "
            "Records sharing the same cluster value are true matches."
        )
        cm  = st.session_state.get("run1_cm", {})
        ts  = st.session_state.get("run1_ts")
        crl = st.session_state.get("run1_crl", {})

        if not cm:
            st.info("Confusion matrix not yet available. Run analysis first.")
        elif cm.get("unavailable"):
            # Dataset does not have a 'cluster' column — show a clear explanation
            st.info(cm.get("unavailable_reason", "Confusion matrix not available."))
            st.caption(
                "To use the confusion matrix, ensure your dataset has a 'cluster' column "
                "containing integer group labels identifying which records refer to the "
                "same real-world entity. The built-in fake1000 dummy dataset has this column."
            )
        elif "error" in cm:
            st.info(f"Confusion matrix error: {cm['error']}")
        else:
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("True Positives (TP)",  f"{cm.get('tp',0):,}")
            kc2.metric("False Positives (FP)", f"{cm.get('fp',0):,}")
            kc3.metric("False Negatives (FN)", f"{cm.get('fn',0):,}")
            kc4.metric("Ground truth pairs",   f"{cm.get('n_gt_edges',0):,}")

            st.divider()
            mc1, mc2 = st.columns(2)
            with mc1:
                st.write("**Derived Metrics**")
                mdf = pd.DataFrame([
                    {"Metric":"Precision", "Value":f"{cm.get('precision',0):.4f}",
                     "Meaning":"TP / (TP+FP)"},
                    {"Metric":"Recall",    "Value":f"{cm.get('recall',0):.4f}",
                     "Meaning":"TP / (TP+FN)"},
                    {"Metric":"F1 Score",  "Value":f"{cm.get('f1',0):.4f}",
                     "Meaning":"Harmonic mean"},
                    {"Metric":"F* Score",  "Value":f"{cm.get('fstar',0):.4f}",
                     "Meaning":"TP / (TP+FP+FN)"},
                    {"Metric":"FDR",       "Value":f"{cm.get('fdr',0):.4f}",
                     "Meaning":"False Discovery Rate"},
                    {"Metric":"FNR",       "Value":f"{cm.get('fnr',0):.4f}",
                     "Meaning":"False Negative Rate"},
                ])
                st.dataframe(mdf, use_container_width=True, hide_index=True)

            with mc2:
                st.write("**Confusion Matrix**")
                z    = [[cm.get("tp",0), cm.get("fp",0)],
                        [cm.get("fn",0), 0]]
                text = [[f"TP<br>{cm.get('tp',0):,}",  f"FP<br>{cm.get('fp',0):,}"],
                        [f"FN<br>{cm.get('fn',0):,}",  "TN<br>(omitted)"]]
                fig_cm = gobj.Figure(data=gobj.Heatmap(
                    z=z, text=text, texttemplate="%{text}",
                    colorscale=[[0,"#B85050"],[0.5,"#CCCCCC"],[1,"#1d8a50"]],
                    showscale=False,
                ))
                fig_cm.update_layout(
                    xaxis=dict(tickvals=[0,1],
                               ticktext=["Predicted Match","Predicted Non-Match"]),
                    yaxis=dict(tickvals=[0,1],
                               ticktext=["True Non-Match","True Match"],
                               autorange="reversed"),
                    height=260, margin=dict(l=10,r=10,t=30,b=10),
                    title="Pairwise Confusion Matrix",
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # Precision-Recall curve (probabilistic only)
        if ts is not None and not ts.empty:
            st.divider()
            st.subheader("Precision-Recall Curve and CRL Score")
            p1, p2 = st.columns(2)
            ts_pr = ts.dropna(subset=["precision_val","recall_val"])
            if not ts_pr.empty:
                with p1:
                    fig_pr = px.line(ts_pr, x="recall_val", y="precision_val",
                                     title="Precision-Recall Curve",
                                     template="simple_white",
                                     color_discrete_sequence=["#1E6EC4"])
                    fig_pr.update_layout(height=300,
                                         xaxis_range=[0,1], yaxis_range=[0,1.05])
                    st.plotly_chart(fig_pr, use_container_width=True)
            ts_fs = ts.dropna(subset=["fstar","match_probability"])
            if not ts_fs.empty:
                with p2:
                    fig_fs = px.line(ts_fs, x="match_probability", y="fstar",
                                     title="F* Score vs Threshold",
                                     template="simple_white",
                                     color_discrete_sequence=["#28A060"])
                    fig_fs.update_layout(height=300,
                                         xaxis_range=[0,1], yaxis_range=[0,1.05])
                    st.plotly_chart(fig_fs, use_container_width=True)
            if crl.get("crl_score") is not None:
                cr1,cr2,cr3,cr4 = st.columns(4)
                cr1.metric("CRL Score", f"{crl.get('crl_score',0):.6f}")
                cr2.metric("t_upper",   str(crl.get("t_upper","N/A")))
                cr3.metric("t_lower",   str(crl.get("t_lower","N/A")))
                cr4.metric("epsilon_z", str(crl.get("epsilon_z","N/A")))

    # ═══════════════════════════════════════════════════════════════════════
    # TAB: Raw Data
    # ═══════════════════════════════════════════════════════════════════════
    with tab_data:
        st.subheader("Raw Tables")
        st.write("**df_predict (first 100 rows)**")
        st.dataframe(results["df_predict"].head(100), use_container_width=True)
        st.caption(
            "Each row is a candidate record pair. "
            "gamma_ columns show field-level agreement (1=exact, 0=disagree). "
            "match_key indicates which blocking rule generated this pair."
        )
        st.write("**df_cluster (first 100 rows)**")
        st.dataframe(results["df_cluster"].head(100), use_container_width=True)

    # ── PDF download ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Save trained model as JSON")
    st.write(
        "Download the trained model as a JSON file. You can upload this file "
        "later using Advanced Mode to skip training and go straight to prediction."
    )
    if st.button("Generate model JSON for download"):
        r = st.session_state.get("run1_results", {})
        mp = r.get("model_params", {})
        su = r.get("settings_used", {})
        if not su:
            st.warning("No settings available. Run the analysis first.")
        else:
            try:
                model_json = reconstruct_model_json(
                    su, mp,
                    linkage_type=r.get("run_config", {}).get("linkage_type", "probabilistic"),
                )
                json_bytes = json.dumps(model_json, indent=2).encode("utf-8")
                st.download_button(
                    "Download model JSON",
                    data=json_bytes,
                    file_name=f"splink_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
                if mp.get("training_complete"):
                    st.success("Model JSON includes trained m/u probabilities. "
                               "Upload this in Advanced Mode to skip training.")
                else:
                    st.info("Model JSON contains settings only (deterministic run). "
                            "Uploading it in Advanced Mode will run prediction without EM training.")
            except Exception as e:
                st.error(f"Failed to generate model JSON: {e}")

    st.divider()
    st.subheader("Download SeRP-style PDF Report")
    if st.button("Generate PDF report"):
        with st.spinner("Generating report..."):
            try:
                ts_for_pdf = (
                    st.session_state["run1_ts"]
                    if st.session_state.get("run1_ts") is not None
                    else pd.DataFrame()
                )
                pdf_bytes = generate_report(
                    run_label="Run 1",
                    run_config=results["run_config"],
                    metrics=metrics,
                    n_input_records=results["n_input_records"],
                    model_params=results.get("model_params", {}),
                    missingness_a=results.get("missingness_a", {}),
                    missingness_b=results.get("missingness_b", {}),
                    blocking_counts=results.get("blocking_counts", []),
                    unlinkables=results.get("unlinkables", {}),
                    settings_used=results.get("settings_used", {}),
                    confusion_matrix=st.session_state.get("run1_cm", {}),
                    truth_space_df=ts_for_pdf,
                    crl_score=st.session_state.get("run1_crl", {}),
                )
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"linkage_report_run1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.divider()
    if st.button("Continue to compare runs", type="primary"):
        _go_to(5)


# =============================================================================
# ── PAGE 5: COMPARISON ────────────────────────────────────────────────────────
# =============================================================================

