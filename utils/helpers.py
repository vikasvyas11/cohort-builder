# utils/helpers.py
# Reusable UI components and the core analysis runner.
# _run_analysis_and_store() is the single point through which ALL three flows
# (standard, upload, advanced) trigger a Splink linkage run.

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.splink_runner import run_linkage, build_coverage_matrix, determine_cascade_order
from modules.metrics_engine import (
    compute_intra_metrics, compute_confusion_matrix,
    compute_truth_space, compute_crl_score, compute_nc_demographic_breakdowns,
    compute_edge_demographic_quality, DEMOGRAPHIC_REGISTRY_FIELDS, NC_DEMOGRAPHIC_FIELD_CONFIG,
)


def _metric_cards(metrics: list) -> None:
    """Render a row of st.metric cards from [(label, value), ...] tuples."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label=label, value=value)


def _plotly_bar(df: pd.DataFrame, x: str, y: str, title: str,
                colour: str = "#1E6EC4") -> "go.Figure":
    """Return a clean Plotly bar chart."""
    fig = px.bar(df, x=x, y=y, title=title,
                 color_discrete_sequence=[colour], template="simple_white")
    fig.update_layout(
        title_font_size=14,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        margin=dict(l=40, r=20, t=50, b=40), height=320,
    )
    return fig


def render_demographic_breakdowns(nc_demo: dict, key_prefix: str = "demo") -> None:
    """Render one chart per Tier-1 NC demographic field found in nc_demo
    (the dict returned by compute_nc_demographic_breakdowns). Pie charts for
    low-cardinality codes, bar charts for birth_state and the binned
    birth_year/age_at_year_end brackets.

    No-op if nc_demo is empty (dummy dataset runs, generic uploads without
    these columns) — this is purely additive to the existing gender/city
    charts, which are unchanged. key_prefix must be unique per call site on
    a page (Run 1 tab, after-toggle panel, Run 2 tab, etc.) to avoid
    Streamlit duplicate-element-id errors.
    """
    if not nc_demo:
        return

    st.write("**NC Voter Demographic Breakdown**")
    fields = list(nc_demo.items())
    for i in range(0, len(fields), 2):
        cols = st.columns(2)
        for col, (field, info) in zip(cols, fields[i:i + 2]):
            df, label, kind = info["df"], info["label"], info["kind"]
            with col:
                if kind == "pie":
                    fig = px.pie(
                        df, values="n_records", names="label",
                        title=f"{label} Distribution",
                        template="simple_white",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
                else:
                    x_col = "label" if kind == "bar_binned" else "value"
                    show_df = df if kind == "bar_binned" else df.head(10)
                    title = label if kind == "bar_binned" else f"Top {min(10, len(df))} {label} values"
                    fig = px.bar(
                        show_df, x=x_col, y="n_records",
                        title=title, template="simple_white",
                        color_discrete_sequence=["#9B59B6"],
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320,
                                       xaxis_title="", yaxis_title="Records")
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{field}")
    st.caption(
        "Demographic breakdown reflects records currently in the clustered "
        "output. Apply a cohort filter on the landing page before running "
        "the analysis to see how a specific cohort's demographics differ."
    )


@st.cache_data(show_spinner=False)
def compute_demographic_snapshot(df: pd.DataFrame) -> dict:
    """Unified demographic breakdown for ANY record-level dataframe — dummy
    dataset's gender/city, or NC voter's Tier-1 fields — in the SAME
    {field: {label, kind, df}} shape that render_demographic_breakdowns()
    and render_demographic_comparison() expect. Wraps
    compute_nc_demographic_breakdowns() and additionally covers gender/city
    with pure pandas (no DuckDB dependency), so it can run on any live
    record subset — e.g. the records touched by the currently active
    blocking rules — without needing a full re-cluster first.
    """
    snapshot = compute_nc_demographic_breakdowns(df)
    if df is None or df.empty:
        return snapshot

    def _simple_breakdown(series: pd.Series) -> pd.DataFrame:
        series = series.dropna()
        series = series[series.astype(str).str.strip() != ""]
        if series.empty:
            return pd.DataFrame()
        counts = series.value_counts()
        out = pd.DataFrame({"value": counts.index.astype(str), "n_records": counts.values})
        out["label"] = out["value"]
        total = out["n_records"].sum()
        out["pct"] = (100.0 * out["n_records"] / total).round(1) if total else 0.0
        return out.sort_values("n_records", ascending=False).reset_index(drop=True)

    if "gender" in df.columns and "gender" not in snapshot:
        gdf = _simple_breakdown(df["gender"])
        if not gdf.empty:
            snapshot["gender"] = {"label": "Gender", "kind": "pie", "df": gdf}

    if "city" in df.columns and "city" not in snapshot:
        cdf = _simple_breakdown(df["city"])
        if not cdf.empty:
            snapshot["city"] = {"label": "City", "kind": "bar", "df": cdf}

    return snapshot


def cohort_from_edges(fakea: pd.DataFrame, fakeb: "pd.DataFrame | None",
                       edges_df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of records (from fakea + fakeb) that appear on
    either side of at least one surviving edge — a live proxy for 'the
    cohort currently reachable under the active blocking rules', without
    needing a full re-cluster. Fully vectorised (merge-based, no per-row
    iteration) so it stays fast on large datasets like NC voter registry.
    """
    if edges_df is None or edges_df.empty:
        return pd.DataFrame()

    id_frames = []
    if {"unique_id_l", "source_dataset_l"}.issubset(edges_df.columns):
        id_frames.append(edges_df[["unique_id_l", "source_dataset_l"]].rename(
            columns={"unique_id_l": "unique_id", "source_dataset_l": "source_dataset"}))
    if {"unique_id_r", "source_dataset_r"}.issubset(edges_df.columns):
        id_frames.append(edges_df[["unique_id_r", "source_dataset_r"]].rename(
            columns={"unique_id_r": "unique_id", "source_dataset_r": "source_dataset"}))
    if not id_frames:
        return pd.DataFrame()
    active_ids = pd.concat(id_frames, ignore_index=True).drop_duplicates()

    frames = []
    if fakea is not None and not fakea.empty:
        a = fakea.copy()
        if "source_dataset" not in a.columns:
            a["source_dataset"] = "A"
        frames.append(a)
    if fakeb is not None and not fakeb.empty:
        b = fakeb.copy()
        if "source_dataset" not in b.columns:
            b["source_dataset"] = "B"
        frames.append(b)
    if not frames:
        return pd.DataFrame()
    all_records = pd.concat(frames, ignore_index=True)

    return (all_records.merge(active_ids, on=["unique_id", "source_dataset"], how="inner")
            .drop_duplicates(subset=["unique_id", "source_dataset"]))


def split_linked_unlinked(df_cluster: pd.DataFrame) -> tuple:
    """Split df_cluster into (linked, unlinked) subsets using cluster size
    as the single definition across dedupe and link modes: cluster size > 1
    = linked (a match was found), size == 1 = unlinked (singleton — no
    match found for this record under the current rules/threshold)."""
    if df_cluster is None or df_cluster.empty or "cluster_id" not in df_cluster.columns:
        return pd.DataFrame(), pd.DataFrame()
    sizes = df_cluster.groupby("cluster_id")["cluster_id"].transform("size")
    linked   = df_cluster[sizes > 1].copy()
    unlinked = df_cluster[sizes == 1].copy()
    return linked, unlinked


def render_demographic_comparison(
    baseline_snapshot: dict,
    current_snapshot:  dict,
    baseline_label:    str = "Run 1",
    current_label:     str = "Run 2 (real-time)",
    key_prefix:        str = "demo_cmp",
    max_fields:        int = 2,
) -> None:
    """Render a side-by-side baseline vs current demographic comparison: up
    to `max_fields` demographic fields, each shown as a
    [baseline chart | current chart] pair in the SAME row — 2 fields = 4
    charts in one row — followed by a plain-language summary of how the
    cohort's demographic composition shifted.

    Works for both dummy (gender/city) and NC voter (Tier-1) fields with no
    dataset-specific logic, since both snapshot dicts come from
    compute_demographic_snapshot(). Safe to reuse across Standard, Upload,
    and Advanced flows — page_comparison is shared across all three.
    """
    common_fields = [f for f in baseline_snapshot if f in current_snapshot]
    common_fields = sorted(
        common_fields, key=lambda f: (0 if f in ("gender", "gender_code") else 1, f)
    )[:max_fields]

    if not common_fields:
        st.info(
            "No comparable demographic fields between the baseline and the "
            "current toggle state — nothing to compare yet."
        )
        return

    st.write(f"**{baseline_label} vs {current_label} — Demographic Comparison**")
    cols = st.columns(2 * len(common_fields))
    summary_lines = []

    for i, field in enumerate(common_fields):
        base_info = baseline_snapshot[field]
        curr_info = current_snapshot[field]
        label, kind = base_info["label"], base_info["kind"]

        for j, (state_label, info) in enumerate(
            [(baseline_label, base_info), (current_label, curr_info)]
        ):
            col = cols[i * 2 + j]
            df = info["df"]
            with col:
                if df.empty:
                    st.caption(f"{label} ({state_label}): no data")
                    continue
                if kind == "pie":
                    fig = px.pie(
                        df, values="n_records", names="label",
                        title=f"{label} — {state_label}",
                        template="simple_white",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(margin=dict(l=5, r=5, t=40, b=5), height=280,
                                       showlegend=True, legend=dict(font=dict(size=9)))
                else:
                    x_col = "label" if kind == "bar_binned" else "value"
                    show_df = df if kind == "bar_binned" else df.head(8)
                    fig = px.bar(
                        show_df, x=x_col, y="n_records",
                        title=f"{label} — {state_label}",
                        template="simple_white",
                        color_discrete_sequence=["#9B59B6" if j == 0 else "#1E6EC4"],
                    )
                    fig.update_layout(margin=dict(l=5, r=5, t=40, b=5), height=280,
                                       xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{field}_{j}")

        # ── Plain-language delta summary for this field ───────────────────────
        base_lookup = dict(zip(base_info["df"].get("label", []), base_info["df"].get("pct", [])))
        curr_lookup = dict(zip(curr_info["df"].get("label", []), curr_info["df"].get("pct", [])))
        shared_labels = set(base_lookup) & set(curr_lookup)
        biggest_shift, biggest_delta = None, 0.0
        for lbl in shared_labels:
            delta = curr_lookup[lbl] - base_lookup[lbl]
            if abs(delta) > abs(biggest_delta):
                biggest_delta, biggest_shift = delta, lbl
        new_only = set(curr_lookup) - set(base_lookup)
        dropped  = set(base_lookup) - set(curr_lookup)

        if biggest_shift is not None:
            direction = "up" if biggest_delta > 0 else "down"
            line = (
                f"**{label}**: '{biggest_shift}' moved {direction} {abs(biggest_delta):.1f} pts "
                f"({base_lookup[biggest_shift]:.1f}% → {curr_lookup[biggest_shift]:.1f}%)."
            )
            if new_only:
                line += f" New category appeared: {', '.join(sorted(new_only))}."
            if dropped:
                line += f" Dropped from view: {', '.join(sorted(dropped))}."
            summary_lines.append(line)
        elif base_lookup or curr_lookup:
            summary_lines.append(f"**{label}**: composition unchanged under the current toggle state.")

    st.divider()
    if summary_lines:
        st.markdown(
            f"{baseline_label} vs {current_label} saw the following changes to the cohort:  \n"
            + "  \n".join(f"- {line}" for line in summary_lines)
        )
    else:
        st.caption(f"{baseline_label} vs {current_label}: no comparable demographic shift detected.")


def render_blocking_waterfall(coverage_matrix: pd.DataFrame, cascade_order: list,
                               active_toggles: dict, key_prefix: str = "waterfall") -> None:
    """Side-by-side blocking-rule edge charts.

      LEFT  — "Original": replicates Splink's own
              cumulative_comparisons_to_be_scored_from_blocking_rules_chart
              methodology — a horizontal bar per rule (in cascade order)
              showing the running UNION total of comparisons if rules
              1..i were all active. A pair covered by two rules is
              attributed once, to the first rule in cascade order that
              covers it — Splink's own de-duplication approach — so
              multi-rule edges are never double-counted. Static reference,
              unaffected by toggles.

      RIGHT — "Toggled (Live)": the existing reactive stacked waterfall.
              Disabled rules show their original standalone count in red
              (what you're giving up); enabled rules show their CURRENT
              redistributed count, including any overflow caught from a
              disabled upstream rule; Total reconciles the recoverable
              total (teal) against permanently lost edges (red top-up).
              Recomputes on every toggle.

    Both charts are built from the SAME underlying attribution
    (compute_blocking_waterfall) so they always agree with each other.
    """
    from modules.splink_runner import compute_blocking_waterfall
    import plotly.graph_objects as go

    data = compute_blocking_waterfall(coverage_matrix, cascade_order, active_toggles)
    if not data["fields"]:
        st.info("No blocking rule coverage data available yet.")
        return

    fields            = data["fields"]
    all_active_count  = data["all_active_count"]
    active_only_count = data["active_only_count"]
    grand_total       = data["grand_total"]
    active_total      = data["active_total"]

    x_labels = [f"Rule {i+1}: {f}" for i, f in enumerate(fields)]

    col_left, col_right = st.columns(2)

    # ── LEFT: original waterfall — all rules active, static (no toggles) ──────
    with col_left:
        st.markdown("**Original — Run Waterfall**")
        st.caption(
            "The run's original state: every configured rule active, "
            "cascaded in order. Static — not affected by toggles."
        )
        original_values = [all_active_count.get(f, 0) for f in fields]

        fig_left = go.Figure()
        fig_left.add_trace(go.Waterfall(
            name="Original cascade",
            x=x_labels + ["Total"],
            measure=["relative"] * len(fields) + ["total"],
            y=original_values + [None],
            text=[f"{v:,}" for v in original_values] + [f"{grand_total:,}"],
            connector={"line": {"color": "rgba(150,150,150,0.4)"}},
            increasing={"marker": {"color": "#1E6EC4"}},
            totals={"marker": {"color": "#1E6EC4"}},
        ))
        fig_left.update_layout(
            template="simple_white", height=420, showlegend=False,
            yaxis_title="Candidate edge pairs", xaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_left, use_container_width=True, key=f"{key_prefix}_original")

    # ── RIGHT: live, toggle-reactive stacked waterfall ─────────────────────────
    with col_right:
        st.markdown("**Toggled (Live) — Redistribution Waterfall**")
        st.caption(
            "Disabling a rule doesn't just remove its edges — later rules "
            "may already cover the same pairs and catch them. Red = "
            "disabled rule's original share / permanently lost edges."
        )
        wf_values, wf_measures = [], []
        for f in fields:
            wf_values.append(active_only_count.get(f, 0) if active_toggles.get(f, False) else 0)
            wf_measures.append("relative")

        fig_right = go.Figure()
        fig_right.add_trace(go.Waterfall(
            name="Active cascade",
            x=x_labels + ["Total"],
            measure=wf_measures + ["total"],
            y=wf_values + [None],
            text=[f"{v:,}" if v else "" for v in wf_values] + [f"{active_total:,}"],
            connector={"line": {"color": "rgba(150,150,150,0.4)"}},
            increasing={"marker": {"color": "#1E6EC4"}},
            totals={"marker": {"color": "#1E6EC4"}},
        ))

        disabled_idx = [i for i, f in enumerate(fields) if not active_toggles.get(f, False)]
        if disabled_idx:
            fig_right.add_trace(go.Bar(
                name="Disabled (original contribution)",
                x=[x_labels[i] for i in disabled_idx],
                y=[all_active_count.get(fields[i], 0) for i in disabled_idx],
                marker_color="#C0392B",
                text=[f"{all_active_count.get(fields[i], 0):,}" for i in disabled_idx],
                textposition="inside",
            ))

        lost = max(grand_total - active_total, 0)
        if lost > 0:
            fig_right.add_trace(go.Bar(
                name="Lost (no active rule catches these)",
                x=["Total"], y=[lost], base=[active_total],
                marker_color="#C0392B",
                text=[f"{lost:,}"], textposition="inside",
            ))

        fig_right.update_layout(
            template="simple_white", height=420, showlegend=True,
            yaxis_title="Candidate edge pairs", xaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_right, use_container_width=True, key=f"{key_prefix}_toggled")

    st.caption(
        f"Theoretical maximum (all rules on): **{grand_total:,}** edges. "
        f"With the current toggle state: **{active_total:,}** edges recoverable "
        f"({'no loss' if lost == 0 else f'{lost:,} permanently lost — no active rule covers them'})."
    )


def render_waterfall_section(coverage_matrix: pd.DataFrame, toggles: dict,
                              blocking_mode: str, key_prefix: str, run_label: str = "") -> None:
    """Shared boilerplate for all three Blocking Explorer locations (Run 1's
    own explorer, the within-run toggle tab, Run 2's explorer): renders the
    cascading waterfall chart for OR-mode runs, or an explanatory note for
    AND-mode runs (a single combined rule has nothing to cascade between).

    This exact block was previously duplicated three times, differing only
    in which session-state key held the toggles and which run's config was
    being read — genuinely identical logic, now defined once.
    """
    suffix = f" ({run_label})" if run_label else ""
    if blocking_mode == "OR":
        st.write(f"**Blocking Rule Cascade — Candidate Edge Pairs{suffix}**")
        st.caption(
            "Shows how disabling a rule redistributes its edges to "
            "downstream rules (or loses them entirely) rather than "
            "simply subtracting its count."
        )
        try:
            cascade_fields = determine_cascade_order(coverage_matrix, list(toggles.keys()))
            render_blocking_waterfall(coverage_matrix, cascade_fields, toggles, key_prefix=key_prefix)
        except Exception as e:
            st.warning(f"Could not compute blocking waterfall: {e}")
        st.divider()
    else:
        st.info(
            f"This run{suffix} used **AND** blocking mode (all fields combined into "
            "one rule), so the rule-cascade waterfall — which visualises "
            "OR-style redistribution between independent rules — doesn't apply here."
        )


def demographic_comparison_fields(baseline_fields: list, current_fields: list) -> tuple:
    """Intersect the known demographic registry against fields actually
    selected as comparisons in BOTH runs — a fair match-quality baseline
    requires the field to have been compared in both.

    Returns (usable_fields, baseline_only, current_only): usable_fields is
    the safe intersection to render; baseline_only/current_only list
    registry fields selected in only one run, so callers can show a clear
    "not available — wasn't selected in [run]'s configuration" note instead
    of silently dropping them.
    """
    registry = set(DEMOGRAPHIC_REGISTRY_FIELDS)
    b = set(baseline_fields or []) & registry
    c = set(current_fields or []) & registry
    usable        = sorted(b & c)
    baseline_only = sorted(b - c)
    current_only  = sorted(c - b)
    return usable, baseline_only, current_only


def render_demographic_match_quality(
    baseline_edges: pd.DataFrame,
    current_edges:  pd.DataFrame,
    fields:         list,
    threshold:      float = 0.8,
    baseline_label: str = "Run 1",
    current_label:  str = "Run 2 (real-time)",
    key_prefix:     str = "match_quality",
) -> None:
    """For each demographic field in `fields`, render two grouped bar charts
    side by side — average match probability, and % of edges at/above the
    cluster threshold — comparing baseline vs current per category (e.g.
    each race code, each birth-decade bucket), stacked one field per row.
    Followed by a one-line summary of the biggest shift for that field.

    This is the edge-level, match-QUALITY counterpart to
    render_demographic_comparison()'s population view: it reacts
    immediately to blocking rule toggles, even when the raw touched-record
    population barely changes, because it's built directly from
    compute_edge_demographic_quality().
    """
    base_quality = compute_edge_demographic_quality(baseline_edges, fields, threshold)
    curr_quality = compute_edge_demographic_quality(current_edges, fields, threshold)
    common = [f for f in fields if f in base_quality and f in curr_quality]

    if not common:
        st.info(
            "No demographic fields have both baseline and current match data "
            "available yet — try enabling more blocking rules."
        )
        return

    st.caption(
        f"Cluster threshold: {threshold:.2f}. Category assigned from each "
        "edge's left-hand record value."
    )

    for field in common:
        bdf = base_quality[field].copy()
        cdf = curr_quality[field].copy()
        bdf["run"], cdf["run"] = baseline_label, current_label
        combo = pd.concat([bdf, cdf], ignore_index=True)

        field_label = NC_DEMOGRAPHIC_FIELD_CONFIG.get(field, {}).get(
            "label", field.replace("_", " ").title()
        )
        st.markdown(f"##### {field_label}")

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(
                combo, x="label", y="avg_match_probability", color="run",
                barmode="group", title=f"Avg. Match Probability by {field_label}",
                template="simple_white",
                color_discrete_map={baseline_label: "#9B59B6", current_label: "#1E6EC4"},
            )
            fig1.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300,
                                xaxis_title="", yaxis_title="Avg. Match Probability",
                                yaxis_range=[0, 1])
            st.plotly_chart(fig1, use_container_width=True, key=f"{key_prefix}_{field}_prob")
        with c2:
            fig2 = px.bar(
                combo, x="label", y="pct_above_threshold", color="run",
                barmode="group", title=f"% Edges ≥ {threshold:.2f} by {field_label}",
                template="simple_white",
                color_discrete_map={baseline_label: "#9B59B6", current_label: "#1E6EC4"},
            )
            fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=300,
                                xaxis_title="", yaxis_title="% Edges Above Threshold",
                                yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_{field}_pct")

        # ── Biggest shift, in plain language ───────────────────────────────
        merged = bdf.merge(cdf, on="category", suffixes=("_base", "_curr"), how="outer")
        merged["label"] = merged["label_base"].fillna(merged["label_curr"])
        merged["prob_delta"] = (
            merged["avg_match_probability_curr"] - merged["avg_match_probability_base"]
        ).fillna(0)
        top = merged.reindex(merged["prob_delta"].abs().sort_values(ascending=False).index).head(1)

        if not top.empty and abs(top["prob_delta"].iloc[0]) > 0.001:
            row = top.iloc[0]
            direction = "up" if row["prob_delta"] > 0 else "down"
            base_n = bdf.loc[bdf["category"] == row["category"], "n_edges"].sum()
            curr_n = cdf.loc[cdf["category"] == row["category"], "n_edges"].sum()
            st.caption(
                f"Biggest shift: **{row['label']}** avg. match probability moved {direction} "
                f"{abs(row['prob_delta']):.3f} "
                f"({row.get('avg_match_probability_base', float('nan')):.3f} → "
                f"{row.get('avg_match_probability_curr', float('nan')):.3f}), "
                f"n={base_n:,} → {curr_n:,} edges."
            )
        else:
            st.caption("No material match-quality shift detected for this field under the current toggle state.")
        st.divider()


def render_match_quality_section(
    baseline_edges:  pd.DataFrame,
    current_edges:   pd.DataFrame,
    baseline_fields: list,
    current_fields:  list,
    threshold:       float = 0.8,
    baseline_label:  str = "Run 1",
    current_label:   str = "Run 2 (real-time)",
    key_prefix:      str = "match_quality",
) -> None:
    """Orchestration wrapper: intersects baseline/current comparison fields
    against the demographic registry, renders the match-quality comparison
    for the safe intersection, and shows a clear note for any registry
    field that was only selected in one of the two runs. Single entry point
    used by all three toggle-driven explorer locations (Run 1's own
    explorer, the within-run toggle tab, and Run 2's explorer) so behaviour
    is identical across dummy dataset, NC voter registry, Upload, and
    Advanced flows.
    """
    usable, baseline_only, current_only = demographic_comparison_fields(
        baseline_fields, current_fields
    )

    st.write(f"**Match Quality by Demographic Group — {baseline_label} vs {current_label}**")

    if not usable:
        st.info(
            f"No demographic fields are selected as comparisons in both {baseline_label} "
            f"and {current_label}'s configuration, so match-quality-by-group can't be "
            "computed yet."
        )
    else:
        render_demographic_match_quality(
            baseline_edges, current_edges, usable, threshold,
            baseline_label, current_label, key_prefix,
        )

    notes = [
        f"`{f}` — not available: wasn't selected as a comparison field in {current_label}'s configuration."
        for f in baseline_only
    ] + [
        f"`{f}` — not available: wasn't selected as a comparison field in {baseline_label}'s configuration."
        for f in current_only
    ]
    if notes:
        with st.expander("Demographic fields not shown (configuration mismatch)"):
            for n in notes:
                st.caption(n)


def _run_analysis_and_store(
    fakea, fakeb, selected_fields, blocking_toggles,
    operation_mode, linkage_type, hyperparams, composite_rules,
) -> bool:
    """Run Splink linkage and persist ALL results to session state.

    Validates selected_fields and blocking_toggles against actual dataset
    columns before calling run_linkage.  Auto-repairs mismatches with a
    visible warning so the user always gets a result rather than a crash.

    Returns True on success, False if validation or linkage failed.
    """
    # ── Field validation: find columns common to all input datasets ────────────
    cols_a = set(fakea.columns)
    cols_b = set(fakeb.columns) if fakeb is not None else cols_a
    common  = cols_a & cols_b

    valid_fields   = [f for f in selected_fields if f in common]
    skipped_fields = [f for f in selected_fields if f not in common]

    if skipped_fields:
        st.warning(
            f"Fields not found in all datasets — skipped: **{', '.join(skipped_fields)}**"
        )
    if not valid_fields:
        st.error(
            "None of the selected fields exist in the loaded dataset. "
            "Go back to Configure Fields and select columns that are present in your data. "
            f"Available columns (Dataset A): {sorted(cols_a)}"
        )
        return False

    # ── Blocking validation: keep only rules whose column is in common set ─────
    valid_blocking = {}
    for key, enabled in blocking_toggles.items():
        if not enabled:
            valid_blocking[key] = False
        elif "+" in key:
            parts = [f.strip() for f in key.split("+")]
            valid_blocking[key] = all(p in common for p in parts)
        else:
            valid_blocking[key] = key in common

    if not any(valid_blocking.values()):
        st.error(
            "No active blocking rules match columns in the dataset. "
            f"Columns present in both datasets: {sorted(common)}"
        )
        return False

    # ── Comparison types (upload flow only; None = use default comparisons) ────
    raw_ct   = st.session_state.get("upload_comp_types") or {}
    comp_types = {f: t for f, t in raw_ct.items() if f in valid_fields} or None

    # ── Run linkage ────────────────────────────────────────────────────────────
    try:
        results = run_linkage(
            fakea=fakea, fakeb=fakeb,
            selected_fields=valid_fields,
            blocking_toggles=valid_blocking,
            operation_mode=operation_mode,
            linkage_type=linkage_type,
            hyperparams=hyperparams,
            composite_rules=composite_rules,
            comp_types=comp_types,
            blocking_mode=st.session_state.get("blocking_mode", "OR"),
        )
    except Exception as e:
        st.error(f"Linkage failed: {e}")
        return False

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = compute_intra_metrics(results["df_predict"], results["df_cluster"])
    cm      = compute_confusion_matrix(results["df_predict"], fakea, fakeb, operation_mode)

    if linkage_type == "probabilistic":
        ts  = compute_truth_space(results["df_predict"], fakea, fakeb, operation_mode)
        crl = compute_crl_score(ts)
    else:
        ts  = None
        crl = {}

    cov = build_coverage_matrix(results["df_predict"], valid_fields)

    # ── Persist ────────────────────────────────────────────────────────────────
    st.session_state.update({
        "run1_results":    results,
        "run1_metrics":    metrics,
        "run1_cm":         cm,
        "run1_ts":         ts,
        "run1_crl":        crl,
        "coverage_matrix": cov,
        "explorer_toggles": dict(valid_blocking),
    })
    return True
