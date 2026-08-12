# =============================================================================
# modules/metrics_engine.py
# PURPOSE: All linkage quality metrics implemented as DuckDB SQL queries.
#
# Implements the full suite from the linkage-metrics repo examples 0-16:
#   example_0_match_weight_counts          -> q_match_weight_distribution
#   example_1_n_links                      -> q_n_edges
#   example_2_n_unique_ids_with_edge       -> q_n_unique_ids_with_edge
#   example_3_match_probability_stats      -> q_match_probability_stats
#   example_4_match_probability_dist       -> q_match_probability_distribution
#   example_5_gamma_scores                 -> q_gamma_scores_dynamic
#   example_6_n_clusters                   -> q_n_clusters
#   example_7_node_counts_per_cluster      -> q_node_counts_per_cluster
#   example_8_cluster_size_distribution    -> q_cluster_size_distribution
#   example_9_singleton_vs_multi           -> q_singleton_vs_multi
#   example_10_source_dataset_membership   -> q_source_dataset_membership
#   example_11_cluster_id_set_membership   -> q_venn_overlap
#   example_12_value_counts_per_cluster    -> q_demographic_breakdown
#   example_13_compare_edges               -> q_edge_difference_counts (inter)
#   example_14_edge_difference_counts      -> q_edge_difference_counts (inter)
#   example_15_exact_partial_clusters      -> q_exact/partial_matching_clusters
#   example_16_CRL_score_pandas            -> compute_truth_space, compute_crl_score
#
# Plus: compute_confusion_matrix() using the `cluster` ground-truth column.
# =============================================================================

import math
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from modules.cohort_filter import (
    RACE_CODE_LABELS, ETHNIC_CODE_LABELS, PARTY_CD_LABELS, GENDER_CODE_LABELS,
)

# =============================================================================
# ── DEMOGRAPHIC BREAKDOWNS — Tier 1 NC voter fields (all flows) ──────────────
# Generalises the existing gender_dist/city_dist pattern (dummy dataset,
# unchanged) to any of the Tier-1 demographic fields defined in
# modules/cohort_filter.py. Works for Standard (NC voter), Upload, and
# Advanced flows alike: fields are auto-detected from whatever columns are
# actually present in df_cluster (Splink retains all original input columns
# per record), so nothing here depends on which fields were selected for
# comparison/blocking.
# =============================================================================

# field -> display config. kind: "pie" for low-cardinality codes, "bar" for
# wider categorical (e.g. state), "bar_binned" for numeric ranges that need
# bucketing before they're readable as a chart.
NC_DEMOGRAPHIC_FIELD_CONFIG = {
    "gender_code":     {"label": "Gender",          "value_labels": GENDER_CODE_LABELS, "kind": "pie"},
    "race_code":       {"label": "Race",             "value_labels": RACE_CODE_LABELS,   "kind": "pie"},
    "ethnic_code":     {"label": "Ethnicity",        "value_labels": ETHNIC_CODE_LABELS, "kind": "pie"},
    "party_cd":        {"label": "Registered Party", "value_labels": PARTY_CD_LABELS,    "kind": "pie"},
    "birth_state":     {"label": "Birth State",      "value_labels": None,               "kind": "bar"},
    "birth_year":      {"label": "Decade of Birth",  "value_labels": None,               "kind": "bar_binned",
                         "bins": list(range(1900, 2031, 10)),
                         "bin_labels": [f"{y}s" for y in range(1900, 2021, 10)]},
    "age_at_year_end": {"label": "Age Bracket",      "value_labels": None,               "kind": "bar_binned",
                         "bins": [0, 18, 25, 35, 45, 55, 65, 75, 85, 130],
                         "bin_labels": ["<18", "18-24", "25-34", "35-44", "45-54",
                                        "55-64", "65-74", "75-84", "85+"]},
}


@st.cache_data(show_spinner=False)
def compute_nc_demographic_breakdowns(df_cluster: pd.DataFrame) -> dict:
    """Compute a distribution DataFrame for every Tier-1 NC demographic field
    that is actually present in df_cluster.

    Returns {field: {"label": str, "kind": str, "df": DataFrame}} where df
    has columns [value, label, n_records, pct] — 'value' is the raw code,
    'label' is the human-readable version (falls back to the raw code when
    no mapping exists).
    """
    breakdowns = {}
    if df_cluster is None or df_cluster.empty:
        return breakdowns

    for field, cfg in NC_DEMOGRAPHIC_FIELD_CONFIG.items():
        if field not in df_cluster.columns:
            continue

        if cfg["kind"] == "bar_binned":
            numeric = pd.to_numeric(df_cluster[field], errors="coerce").dropna()
            if numeric.empty:
                continue
            binned = pd.cut(numeric, bins=cfg["bins"], labels=cfg["bin_labels"],
                             right=False, include_lowest=True)
            counts = binned.value_counts().reindex(cfg["bin_labels"]).fillna(0).astype(int)
            df = pd.DataFrame({
                "value":     counts.index.astype(str),
                "label":     counts.index.astype(str),
                "n_records": counts.values,
            })
            total = df["n_records"].sum()
            df["pct"] = (100.0 * df["n_records"] / total).round(1) if total else 0.0
            df = df[df["n_records"] > 0].reset_index(drop=True)
        else:
            series = df_cluster[field].dropna()
            series = series[series.astype(str).str.strip() != ""]
            if series.empty:
                continue
            counts = series.value_counts()
            value_labels = cfg.get("value_labels") or {}
            df = pd.DataFrame({
                "value":     counts.index.astype(str),
                "n_records": counts.values,
            })
            df["label"] = df["value"].map(lambda v: value_labels.get(v, v))
            total = df["n_records"].sum()
            df["pct"] = (100.0 * df["n_records"] / total).round(1) if total else 0.0
            df = df.sort_values("n_records", ascending=False).reset_index(drop=True)

        if not df.empty:
            breakdowns[field] = {"label": cfg["label"], "kind": cfg["kind"], "df": df}

    return breakdowns


# Every demographic field this app knows how to visualise — Tier-1 NC voter
# fields plus the dummy dataset's gender/city. Used to intersect against
# whatever fields were actually selected as comparisons in a given run.
DEMOGRAPHIC_REGISTRY_FIELDS = list(NC_DEMOGRAPHIC_FIELD_CONFIG.keys()) + ["gender", "city"]


def compute_edge_demographic_quality(edges_df: pd.DataFrame, fields: list,
                                      threshold: float = 0.8) -> dict:
    """Edge-level match-quality breakdown, per demographic field, per
    category. Unlike compute_nc_demographic_breakdowns() (population counts
    from df_cluster), this reads the category straight off each edge's
    left-hand value (`<field>_l`) and reports match QUALITY — it is far
    more sensitive to blocking rule toggles, since removing a rule directly
    removes the specific edges it contributed, immediately shifting the
    probability distribution of what's left, even when the raw touched-
    record population barely moves.

    Only fields with a `<field>_l` column in edges_df are included — i.e.
    fields actually selected as comparisons for that run (Splink only
    retains matching columns for compared fields), which is what keeps this
    naturally scoped to "fields selected in the configuration step".

    Returns {field: DataFrame[category, label, n_edges,
             avg_match_probability, pct_above_threshold]}.
    """
    results = {}
    if edges_df is None or edges_df.empty or "match_probability" not in edges_df.columns:
        return results

    for field in fields:
        col_l = f"{field}_l"
        if col_l not in edges_df.columns:
            continue

        cfg  = NC_DEMOGRAPHIC_FIELD_CONFIG.get(field, {})
        kind = cfg.get("kind", "pie")

        sub = edges_df[[col_l, "match_probability"]].copy()

        if kind == "bar_binned":
            numeric = pd.to_numeric(sub[col_l], errors="coerce")
            binned  = pd.cut(numeric, bins=cfg["bins"], labels=cfg["bin_labels"],
                              right=False, include_lowest=True)
            sub["_cat"] = binned.astype(str)
            sub = sub[sub["_cat"] != "nan"]
        else:
            sub["_cat"] = sub[col_l].astype(str).str.strip()
            sub = sub[(sub["_cat"] != "") & (~sub["_cat"].str.lower().isin(["nan", "none"]))]

        if sub.empty:
            continue

        sub["_above"] = sub["match_probability"] >= threshold
        grouped = sub.groupby("_cat").agg(
            n_edges=("match_probability", "count"),
            avg_match_probability=("match_probability", "mean"),
            pct_above_threshold=("_above", "mean"),
        ).reset_index().rename(columns={"_cat": "category"})
        grouped["avg_match_probability"] = grouped["avg_match_probability"].round(3)
        grouped["pct_above_threshold"]   = (grouped["pct_above_threshold"] * 100).round(1)

        value_labels = cfg.get("value_labels") or {}
        grouped["label"] = grouped["category"].map(lambda v: value_labels.get(v, v))
        grouped = grouped.sort_values("n_edges", ascending=False).reset_index(drop=True)
        results[field] = grouped

    return results


# =============================================================================
# ── INTRA-MODEL EDGE METRICS (examples 0-5) ───────────────────────────────────
# =============================================================================

def q_n_edges(table: str = "df_predict") -> str:
    """Example 1: Total count of predicted pairwise edges."""
    return f"SELECT COUNT(1) AS n_edges FROM {table}"


def q_n_unique_ids_with_edge(table: str = "df_predict") -> str:
    """Example 2: Number of distinct unique_ids appearing in at least one edge.
    Unions left and right sides so each ID is counted once."""
    return f"""
    SELECT COUNT(DISTINCT uid) AS n_unique_ids_with_edge
    FROM (
        SELECT unique_id_l AS uid FROM {table}
        UNION
        SELECT unique_id_r AS uid FROM {table}
    )"""


def q_match_probability_stats(table: str = "df_predict") -> str:
    """Example 3: Summary statistics for the match_probability column."""
    return f"""
    SELECT
        ROUND(AVG(match_probability),    4) AS mean_match_prob,
        ROUND(MEDIAN(match_probability), 4) AS median_match_prob,
        ROUND(MIN(match_probability),    4) AS min_match_prob,
        ROUND(MAX(match_probability),    4) AS max_match_prob,
        ROUND(STDDEV(match_probability), 4) AS stddev_match_prob
    FROM {table}"""


def q_match_probability_distribution(table: str = "df_predict") -> str:
    """Example 4: Histogram of match probabilities in 0.05-wide bins."""
    return f"""
    SELECT
        ROUND(FLOOR(match_probability / 0.05) * 0.05, 2) AS prob_bin,
        COUNT(1) AS n_edges
    FROM {table}
    GROUP BY prob_bin
    ORDER BY prob_bin"""


def q_match_weight_distribution(table: str = "df_predict") -> str:
    """Example 0: Histogram of match weights (rounded to 1 decimal place).
    The match_weight is log2(m/u) summed across all comparison fields."""
    return f"""
    SELECT
        ROUND(match_weight, 1) AS weight_bin,
        COUNT(1)               AS n_edges
    FROM {table}
    GROUP BY weight_bin
    ORDER BY weight_bin"""


# =============================================================================
# ── INTRA-MODEL CLUSTER METRICS (examples 6-12) ───────────────────────────────
# =============================================================================

def q_n_clusters(table: str = "df_cluster") -> str:
    """Example 6: Count of distinct entity clusters."""
    return f"SELECT COUNT(DISTINCT cluster_id) AS n_clusters FROM {table}"


def q_node_counts_per_cluster(table: str = "df_cluster") -> str:
    """Example 7: Number of records in each cluster."""
    return f"""
    SELECT cluster_id, COUNT(1) AS n_nodes
    FROM {table}
    GROUP BY cluster_id"""


def q_cluster_size_distribution(table: str = "df_cluster") -> str:
    """Example 8: How many clusters have each size (1 record, 2 records, etc.)."""
    return f"""
    SELECT n_nodes, COUNT(1) AS n_clusters
    FROM ({q_node_counts_per_cluster(table)})
    GROUP BY n_nodes
    ORDER BY n_nodes"""


def q_singleton_vs_multi(table: str = "df_cluster") -> str:
    """Example 9: Split clusters into singletons (1 record) and multi-record groups."""
    return f"""
    SELECT
        CASE WHEN n_nodes = 1
             THEN 'Singleton (1 record)'
             ELSE 'Multi-record cluster (2+ records)'
        END         AS cluster_type,
        COUNT(1)    AS n_clusters,
        SUM(n_nodes) AS total_records
    FROM ({q_node_counts_per_cluster(table)})
    GROUP BY cluster_type
    ORDER BY cluster_type"""


def q_source_dataset_membership(table: str = "df_cluster") -> str:
    """Example 10: How many clusters contain records from each source dataset."""
    return f"""
    SELECT
        source_dataset,
        COUNT(DISTINCT cluster_id) AS n_clusters_containing_dataset
    FROM {table}
    GROUP BY source_dataset
    ORDER BY source_dataset"""


def q_venn_overlap(table: str = "df_cluster") -> str:
    """Example 11: cluster_id_set_membership adapted for Venn diagram rendering.
    Returns three counts: clusters with only A records, only B records, or both.
    Used to draw the two-circle Venn diagram in the PDF report."""
    return f"""
    SELECT
        SUM(CASE WHEN has_a AND NOT has_b THEN 1 ELSE 0 END) AS a_only,
        SUM(CASE WHEN has_b AND NOT has_a THEN 1 ELSE 0 END) AS b_only,
        SUM(CASE WHEN has_a AND has_b     THEN 1 ELSE 0 END) AS both_ab
    FROM (
        SELECT
            cluster_id,
            MAX(CASE WHEN source_dataset = 'A' THEN 1 ELSE 0 END) AS has_a,
            MAX(CASE WHEN source_dataset = 'B' THEN 1 ELSE 0 END) AS has_b
        FROM {table}
        GROUP BY cluster_id
    )"""


def q_demographic_breakdown(table: str = "df_cluster", col: str = "gender") -> str:
    """Example 12: value_counts_per_cluster adapted – frequency table for a column."""
    return f"""
    SELECT
        {col},
        COUNT(1) AS n_records,
        ROUND(100.0 * COUNT(1) / SUM(COUNT(1)) OVER (), 1) AS pct
    FROM {table}
    WHERE {col} IS NOT NULL
    GROUP BY {col}
    ORDER BY n_records DESC"""


def q_cross_dataset_clusters(table: str = "df_cluster") -> str:
    """Clusters that contain records from more than one source dataset."""
    return f"""
    SELECT COUNT(DISTINCT cluster_id) AS n_cross_dataset_clusters
    FROM (
        SELECT cluster_id, COUNT(DISTINCT source_dataset) AS n_sources
        FROM {table}
        GROUP BY cluster_id
        HAVING n_sources > 1
    )"""


# =============================================================================
# ── INTER-MODEL METRICS (examples 13-15) ──────────────────────────────────────
# =============================================================================

def q_edge_difference_counts(
    table_a: str = "df_predict_run1",
    table_b: str = "df_predict_run2",
) -> str:
    """Examples 13-14: Count edges shared, added, and removed between two runs.
    Uses only unique_id_l/r to avoid BinderException when source_dataset_l/r
    are absent (e.g. deterministic linkage output)."""
    return f"""
    SELECT COUNT(*) AS n, 'shared'  AS category
    FROM {table_a} a
    INNER JOIN {table_b} b
    ON a.unique_id_l = b.unique_id_l AND a.unique_id_r = b.unique_id_r
    UNION ALL
    SELECT COUNT(*) AS n, 'added'   AS category
    FROM {table_b} b
    WHERE NOT EXISTS (
        SELECT 1 FROM {table_a} a
        WHERE a.unique_id_l = b.unique_id_l AND a.unique_id_r = b.unique_id_r
    )
    UNION ALL
    SELECT COUNT(*) AS n, 'removed' AS category
    FROM {table_a} a
    WHERE NOT EXISTS (
        SELECT 1 FROM {table_b} b
        WHERE b.unique_id_l = a.unique_id_l AND b.unique_id_r = a.unique_id_r
    )"""


def q_match_prob_comparison(
    table_a: str = "df_predict_run1",
    table_b: str = "df_predict_run2",
) -> str:
    """Compare mean match probabilities side by side between two runs."""
    return f"""
    SELECT 'Run 1' AS run,
           ROUND(AVG(match_probability), 4) AS mean_match_prob,
           ROUND(MEDIAN(match_probability), 4) AS median_match_prob,
           COUNT(1) AS n_edges
    FROM {table_a}
    UNION ALL
    SELECT 'Run 2' AS run,
           ROUND(AVG(match_probability), 4) AS mean_match_prob,
           ROUND(MEDIAN(match_probability), 4) AS median_match_prob,
           COUNT(1) AS n_edges
    FROM {table_b}"""


def q_exact_matching_clusters(
    table1: str = "df_cluster_run1",
    table2: str = "df_cluster_run2",
) -> str:
    """Example 15a: Clusters whose exact membership is identical across two runs."""
    return f"""
    SELECT a.cluster_id, a.unique_ids AS run1_unique_ids
    FROM (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table1} GROUP BY cluster_id
    ) AS a
    INNER JOIN (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table2} GROUP BY cluster_id
    ) AS b
    USING (cluster_id)
    WHERE a.unique_ids = b.unique_ids"""


def q_partial_matching_clusters(
    table1: str = "df_cluster_run1",
    table2: str = "df_cluster_run2",
) -> str:
    """Example 15b: Clusters that share some but not all members between runs."""
    return f"""
    SELECT
        a.cluster_id AS run1_cluster_id,
        b.cluster_id AS run2_cluster_id,
        a.unique_ids AS run1_unique_ids,
        b.unique_ids AS run2_unique_ids
    FROM (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table1} GROUP BY cluster_id
    ) AS a
    JOIN (
        SELECT cluster_id, LIST(unique_id ORDER BY unique_id) AS unique_ids
        FROM {table2} GROUP BY cluster_id
    ) AS b
    ON list_has_any(a.unique_ids, b.unique_ids)
       AND a.unique_ids != b.unique_ids"""


# =============================================================================
# ── TRUTH SPACE AND CRL SCORE (example 16) ────────────────────────────────────
# Based on: linkage_metrics/intra_model/edge_metrics.py
#           truth_space_table_from_labels_column_offline
#           calculate_crl_score
# =============================================================================

def _build_gt_edge_sql(gt_table: str, operation_mode: str) -> str:
    """Build the SQL to generate the ground truth edge list from the cluster column.
    For dedupe mode: all within-dataset pairs with the same cluster value.
    For link mode: all cross-dataset pairs with the same cluster value."""
    if operation_mode == "dedupe":
        # Both IDs come from dataset A; cluster column is the ground truth
        return f"""
        SELECT
            a.unique_id          AS unique_id_l,
            a.source_dataset     AS source_dataset_l,
            b.unique_id          AS unique_id_r,
            b.source_dataset     AS source_dataset_r,
            1.0                  AS clerical_match_score
        FROM {gt_table} a
        JOIN {gt_table} b
             ON a.cluster = b.cluster
        WHERE (a.source_dataset || '-' || a.unique_id)
            < (b.source_dataset || '-' || b.unique_id)"""
    else:
        # Link mode: true matches span dataset A and B
        # fakeb unique_ids end in '_B'; their cluster values are identical to fakea
        return f"""
        SELECT
            a.unique_id          AS unique_id_l,
            a.source_dataset     AS source_dataset_l,
            b.unique_id          AS unique_id_r,
            b.source_dataset     AS source_dataset_r,
            1.0                  AS clerical_match_score
        FROM {gt_table} a
        JOIN {gt_table} b
             ON a.cluster = b.cluster
        WHERE a.source_dataset != b.source_dataset
          AND (a.source_dataset || '-' || a.unique_id)
            < (b.source_dataset || '-' || b.unique_id)"""


def compute_truth_space(
    df_predict: pd.DataFrame,
    fakea: pd.DataFrame,
    fakeb: Optional[pd.DataFrame],
    operation_mode: str,
) -> pd.DataFrame:
    """Example 16: Compute the truth_space table.

    For each match_probability threshold, calculates:
      tp      : true positives above threshold
      fp      : false positives above threshold
      fn      : false negatives (missed true matches)
      precision: tp / (tp + fp)
      recall  : tp / (tp + fn)
      fdr     : 1 - precision  (false discovery rate)
      fnr     : 1 - recall     (false negative rate)
      fstar   : tp / (tp + fp + fn)  (F* = F-star score)

    Mirrors truth_space_table_from_labels_column_offline() from metrics repo.
    Returns a DataFrame sorted by match_probability descending.
    Returns empty DataFrame if computation fails.
    """
    try:
        con = duckdb.connect()

        # Build the ground truth table
        if operation_mode == "dedupe":
            gt_df = fakea[["unique_id", "source_dataset", "cluster"]].copy()
        else:
            gt_df = pd.concat(
                [
                    fakea[["unique_id", "source_dataset", "cluster"]],
                    fakeb[["unique_id", "source_dataset", "cluster"]],
                ],
                ignore_index=True,
            )

        con.register("df_predict",  df_predict)
        con.register("gt_raw",      gt_df)

        # Create ground truth edges
        gt_sql = _build_gt_edge_sql("gt_raw", operation_mode)
        con.execute(f"CREATE TEMP TABLE gt_edges AS {gt_sql}")

        # Truth-space query (direct port of metrics repo example 16)
        truth_space_sql = """
        WITH pred_gt_join AS (
            SELECT
                p.match_probability,
                p.unique_id_l  AS pred,
                g.unique_id_l  AS gt
            FROM df_predict p
            FULL JOIN gt_edges g
                USING (unique_id_l, unique_id_r, source_dataset_l, source_dataset_r)
            ORDER BY match_probability DESC
        )
        SELECT
            match_probability,
            SUM(COUNTIF(pred IS NOT NULL AND gt IS NOT NULL)) OVER w_up   AS tp,
            SUM(COUNTIF(pred IS NOT NULL AND gt IS NULL))     OVER w_up   AS fp,
            COALESCE(
                SUM(COUNTIF(pred IS NOT NULL AND gt IS NOT NULL)) OVER w_down,
                0
            )                                                              AS fn,
            CASE WHEN tp + fp > 0 THEN ROUND(tp * 1.0 / (tp + fp), 6) ELSE NULL END AS precision_val,
            CASE WHEN tp + fn > 0 THEN ROUND(tp * 1.0 / (tp + fn), 6) ELSE NULL END AS recall_val,
            CASE WHEN tp + fp > 0 THEN ROUND(1.0 - tp * 1.0 / (tp + fp), 6) ELSE NULL END AS fdr,
            CASE WHEN tp + fn > 0 THEN ROUND(1.0 - tp * 1.0 / (tp + fn), 6) ELSE NULL END AS fnr,
            CASE WHEN tp + fp + fn > 0 THEN ROUND(tp * 1.0 / (tp + fp + fn), 6) ELSE NULL END AS fstar
        FROM pred_gt_join
        GROUP BY match_probability
        WINDOW
            w_up   AS (ORDER BY match_probability DESC GROUPS UNBOUNDED PRECEDING),
            w_down AS (ORDER BY match_probability DESC GROUPS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
        ORDER BY match_probability DESC
        """
        truth_space_df = con.sql(truth_space_sql).df()
        con.close()
        return truth_space_df

    except Exception as e:
        return pd.DataFrame()


def compute_crl_score(
    truth_space_df: pd.DataFrame,
    epsilon: float = 0.1,
) -> dict:
    """Compute the CRL score from the truth space table (example 16).

    The CRL score is a composite quality metric from the metrics repo:
      CRL = AVG(f_star) * (t_upper - t_lower)
    where the window is constrained to regions where both:
      FDR (false discovery rate) <= epsilon
      FNR (false negative rate)  <= epsilon

    A higher CRL score means better linkage quality in the valid operating region.
    epsilon_z is the smallest epsilon at which a valid operating region exists.

    Returns a dict with crl_score, t_upper, t_lower, epsilon_z, and a curve DataFrame.
    """
    if truth_space_df.empty:
        return {"crl_score": None, "t_upper": None, "t_lower": None, "epsilon_z": None}

    try:
        # Evaluate CRL at the requested epsilon
        valid = truth_space_df[
            (truth_space_df["fdr"]  <= epsilon) &
            (truth_space_df["fnr"]  <= epsilon) &
            truth_space_df["fstar"].notna()
        ]
        if valid.empty:
            crl, t_up, t_low = 0.0, None, None
        else:
            t_up  = float(valid["match_probability"].max())
            t_low = float(valid["match_probability"].min())
            crl   = float(valid["fstar"].mean()) * (t_up - t_low)

        # Find epsilon_z (smallest epsilon where a valid region exists) by scanning
        epsilons   = np.linspace(0.0, 1.0, 100)
        epsilon_z  = None
        crl_curve  = []
        for eps in epsilons:
            v = truth_space_df[
                (truth_space_df["fdr"]  <= eps) &
                (truth_space_df["fnr"]  <= eps) &
                truth_space_df["fstar"].notna()
            ]
            if v.empty:
                crl_curve.append(0.0)
            else:
                score = float(v["fstar"].mean()) * (
                    float(v["match_probability"].max()) - float(v["match_probability"].min())
                )
                crl_curve.append(score)
                if epsilon_z is None and score > 0:
                    epsilon_z = float(eps)

        return {
            "crl_score":  crl,
            "t_upper":    t_up,
            "t_lower":    t_low,
            "epsilon_z":  epsilon_z,
            "epsilon":    epsilon,
            "crl_curve":  pd.DataFrame({"epsilon": epsilons, "crl_score": crl_curve}),
        }
    except Exception:
        return {"crl_score": None, "t_upper": None, "t_lower": None, "epsilon_z": None}


# =============================================================================
# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
# Uses the `cluster` column in fakea/fakeb as the ground truth label.
# A pair is a TRUE MATCH if both records share the same cluster value.
# A pair is a PREDICTED MATCH if it appears in df_predict (after threshold).
# =============================================================================

def compute_confusion_matrix(
    df_predict:     pd.DataFrame,
    fakea:          pd.DataFrame,
    fakeb:          Optional[pd.DataFrame],
    operation_mode: str,
) -> dict:
    """Compute pairwise confusion matrix using the cluster column as ground truth.

    Metrics:
      TP : predicted edge AND true match (same cluster on both sides)
      FP : predicted edge BUT NOT true match (different cluster values)
      FN : true match pair NOT found in df_predict
      TN : omitted – too large to count (all non-adjacent, non-matching pairs)
      Precision = TP / (TP + FP)
      Recall    = TP / (TP + FN)
      F1        = 2 * Precision * Recall / (Precision + Recall)
      FDR       = FP / (TP + FP)   (false discovery rate)
      FNR       = FN / (TP + FN)   (false negative rate)

    Returns a dict with all of the above, plus:
      df_predict_labelled : df_predict with an 'is_true_match' boolean column
      n_gt_edges          : total ground truth pairs (denominator for recall)
    """
    if df_predict.empty:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "fdr": 1.0, "fnr": 1.0,
                "n_gt_edges": 0, "df_predict_labelled": pd.DataFrame()}

    # Ground truth requires a 'cluster' column with entity group labels.
    # Uploaded datasets will not have this column — return a graceful message
    # rather than crashing with a KeyError.
    has_cluster_a = "cluster" in fakea.columns
    has_cluster_b = (fakeb is None) or ("cluster" in fakeb.columns)
    if not has_cluster_a:
        return {
            "tp": None, "fp": None, "fn": None,
            "n_gt_edges": None, "n_pred_edges": len(df_predict),
            "precision": None, "recall": None, "f1": None,
            "fdr": None, "fnr": None, "fstar": None,
            "df_predict_labelled": pd.DataFrame(),
            "unavailable": True,
            "unavailable_reason": (
                "The confusion matrix requires a 'cluster' column in the dataset "
                "as the ground-truth entity label. This column was not found. "
                "For uploaded datasets, ground truth is automatically assigned "
                "when Dataset B is generated via 'Generate Dataset B'. "
                "If you uploaded Dataset B separately, the confusion matrix "
                "cannot be computed without an external ground-truth cluster column."
            ),
        }

    try:
        con = duckdb.connect()

        # Build combined ground truth table
        if operation_mode == "dedupe":
            gt_df = fakea[["unique_id", "source_dataset", "cluster"]].copy()
        else:
            gt_df = pd.concat(
                [
                    fakea[["unique_id", "source_dataset", "cluster"]],
                    fakeb[["unique_id", "source_dataset", "cluster"]],
                ],
                ignore_index=True,
            )

        con.register("df_predict", df_predict)
        con.register("gt_raw",     gt_df)

        # ── Ground truth edges ─────────────────────────────────────────────────
        gt_sql = _build_gt_edge_sql("gt_raw", operation_mode)
        con.execute(f"CREATE TEMP TABLE gt_edges AS {gt_sql}")
        n_gt = con.sql("SELECT COUNT(*) AS n FROM gt_edges").fetchone()[0]

        # ── Label each predicted edge as TP or FP ──────────────────────────────
        con.execute("""
            CREATE TEMP TABLE df_predict_labelled AS
            SELECT
                p.*,
                -- Look up the cluster value for each side from the ground truth table
                gt_l.cluster AS cluster_l,
                gt_r.cluster AS cluster_r,
                -- is_true_match = True if both records share the same cluster value
                (gt_l.cluster IS NOT NULL
                 AND gt_r.cluster IS NOT NULL
                 AND gt_l.cluster = gt_r.cluster) AS is_true_match
            FROM df_predict p
            LEFT JOIN gt_raw gt_l
                ON p.unique_id_l      = gt_l.unique_id
               AND p.source_dataset_l = gt_l.source_dataset
            LEFT JOIN gt_raw gt_r
                ON p.unique_id_r      = gt_r.unique_id
               AND p.source_dataset_r = gt_r.source_dataset
        """)

        # TP: predicted match where cluster values agree
        tp = con.sql(
            "SELECT COUNT(*) AS n FROM df_predict_labelled WHERE is_true_match = TRUE"
        ).fetchone()[0]

        # FP: predicted match where cluster values disagree
        fp = con.sql(
            "SELECT COUNT(*) AS n FROM df_predict_labelled WHERE is_true_match = FALSE"
        ).fetchone()[0]

        # FN: true matches not found in df_predict
        fn_sql = """
        SELECT COUNT(*) AS n FROM gt_edges g
        WHERE NOT EXISTS (
            SELECT 1 FROM df_predict p
            WHERE p.unique_id_l      = g.unique_id_l
              AND p.unique_id_r      = g.unique_id_r
              AND p.source_dataset_l = g.source_dataset_l
              AND p.source_dataset_r = g.source_dataset_r
        )"""
        fn = con.sql(fn_sql).fetchone()[0]

        # Retrieve labelled predictions for display
        df_labelled = con.sql("SELECT * FROM df_predict_labelled").df()
        con.close()

        # ── Compute derived metrics ────────────────────────────────────────────
        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1         = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        fdr        = 1.0 - precision    # False discovery rate
        fnr        = 1.0 - recall       # False negative rate

        # F-star: tp / (tp + fp + fn)  – balanced metric without TN
        fstar      = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        return {
            "tp":                   int(tp),
            "fp":                   int(fp),
            "fn":                   int(fn),
            "n_gt_edges":           int(n_gt),
            "n_pred_edges":         len(df_predict),
            "precision":            round(precision, 4),
            "recall":               round(recall, 4),
            "f1":                   round(f1, 4),
            "fdr":                  round(fdr, 4),
            "fnr":                  round(fnr, 4),
            "fstar":                round(fstar, 4),
            "df_predict_labelled":  df_labelled,
        }
    except Exception as e:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "fdr": 1.0, "fnr": 1.0,
                "fstar": 0.0, "n_gt_edges": 0, "n_pred_edges": len(df_predict),
                "df_predict_labelled": pd.DataFrame(),
                "error": str(e)}


# =============================================================================
# ── EXECUTION HELPERS ─────────────────────────────────────────────────────────
# =============================================================================

def compute_intra_metrics(df_predict: pd.DataFrame, df_cluster: pd.DataFrame) -> dict:
    """Compute all intra-model metrics for a single run (examples 0-12).

    Returns a dict with all metric DataFrames and scalar values.
    All computations use fresh in-memory DuckDB connections.
    """
    con = duckdb.connect()
    con.register("df_predict", df_predict)
    con.register("df_cluster", df_cluster)

    results = {}

    # ── Edge counts (examples 1, 2) ───────────────────────────────────────────
    results["n_edges"]      = con.sql(q_n_edges()).fetchone()[0]
    results["n_unique_ids"] = con.sql(q_n_unique_ids_with_edge()).fetchone()[0]

    # ── Match probability stats (example 3) ──────────────────────────────────
    results["match_prob_stats"] = con.sql(q_match_probability_stats()).df()

    # ── Distributions (examples 0, 4) ────────────────────────────────────────
    results["weight_dist"] = con.sql(q_match_weight_distribution()).df()
    results["prob_dist"]   = con.sql(q_match_probability_distribution()).df()

    # ── Cluster metrics (examples 6-10) ──────────────────────────────────────
    results["n_clusters"]     = con.sql(q_n_clusters()).fetchone()[0]
    results["cluster_sizes"]  = con.sql(q_cluster_size_distribution()).df()
    results["singleton_stats"]= con.sql(q_singleton_vs_multi()).df()
    results["source_overlap"] = con.sql(q_source_dataset_membership()).df()

    # ── Cross-dataset cluster count ───────────────────────────────────────────
    try:
        results["n_cross_dataset"] = con.sql(q_cross_dataset_clusters()).fetchone()[0]
    except Exception:
        results["n_cross_dataset"] = 0

    # ── Venn overlap numbers (example 11) ────────────────────────────────────
    try:
        venn_row = con.sql(q_venn_overlap()).fetchone()
        results["venn"] = {
            "a_only":  int(venn_row[0] or 0),
            "b_only":  int(venn_row[1] or 0),
            "both_ab": int(venn_row[2] or 0),
        }
    except Exception:
        results["venn"] = {"a_only": 0, "b_only": 0, "both_ab": 0}

    # ── Gamma scores (example 5) – dynamic on available gamma_ columns ────────
    gamma_cols = [c for c in df_predict.columns if c.startswith("gamma_")]
    if gamma_cols:
        gamma_agg = ", ".join(
            [f"ROUND(AVG({c}), 4) AS {c}" for c in gamma_cols]
        )
        results["gamma_means"] = con.sql(
            f"SELECT {gamma_agg} FROM df_predict"
        ).df()
    else:
        results["gamma_means"] = pd.DataFrame()

    # ── Demographics (example 12) ─────────────────────────────────────────────
    cluster_cols = df_cluster.columns.tolist()
    results["gender_dist"] = (
        con.sql(q_demographic_breakdown("df_cluster", "gender")).df()
        if "gender" in cluster_cols else pd.DataFrame()
    )
    results["city_dist"] = (
        con.sql(q_demographic_breakdown("df_cluster", "city")).df()
        if "city" in cluster_cols else pd.DataFrame()
    )
    # NC voter Tier-1 demographic fields (no-op / empty dict for datasets
    # that don't have these columns, e.g. dummy dataset or generic uploads)
    results["nc_demographics"] = compute_nc_demographic_breakdowns(df_cluster)

    # ── Additional edge metrics (Feedback 2) ──────────────────────────────────
    # Linkage rate: % of input records that appear in at least one edge
    n_records = con.sql("SELECT COUNT(DISTINCT unique_id) FROM df_cluster").fetchone()[0]
    n_with_edge = results["n_unique_ids"]
    results["linkage_rate"] = round(100.0 * n_with_edge / n_records, 2) if n_records else 0.0

    # Match weight percentiles (P10, P25, P75, P90)
    if "match_weight" in df_predict.columns:
        try:
            pcts = con.sql("""
                SELECT
                    ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY match_weight), 4) AS p10,
                    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY match_weight), 4) AS p25,
                    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY match_weight), 4) AS p75,
                    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY match_weight), 4) AS p90
                FROM df_predict
            """).df()
            results["weight_percentiles"] = pcts
        except Exception:
            results["weight_percentiles"] = pd.DataFrame()
    else:
        results["weight_percentiles"] = pd.DataFrame()

    # Additional cluster metrics (Feedback 2)
    # Average, max, and median cluster size
    try:
        cluster_stats = con.sql("""
            WITH sizes AS (
                SELECT cluster_id, COUNT(*) AS n
                FROM df_cluster
                GROUP BY cluster_id
            )
            SELECT
                ROUND(AVG(n), 2)    AS mean_cluster_size,
                MAX(n)              AS max_cluster_size,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n) AS median_cluster_size,
                SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) AS multi_member_clusters,
                ROUND(100.0 * SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END)
                      / COUNT(*), 2) AS pct_multi_member
            FROM sizes
        """).df()
        results["cluster_stats"] = cluster_stats
    except Exception:
        results["cluster_stats"] = pd.DataFrame()

    # Reduction ratio: fraction of O(n²) space searched via blocking
    try:
        n_total_records = con.sql("SELECT COUNT(*) FROM df_cluster").fetchone()[0]
        max_possible = (n_total_records * (n_total_records - 1)) // 2
        results["reduction_ratio"] = round(
            100.0 * (1 - results["n_edges"] / max(max_possible, 1)), 2
        )
    except Exception:
        results["reduction_ratio"] = None

    con.close()
    return results


def compute_inter_metrics(
    df_predict_run1: pd.DataFrame,
    df_predict_run2: pd.DataFrame,
    df_cluster_run1: pd.DataFrame,
    df_cluster_run2: pd.DataFrame,
) -> dict:
    """Compute inter-model comparison metrics (examples 13-15)."""
    con = duckdb.connect()
    con.register("df_predict_run1", df_predict_run1)
    con.register("df_predict_run2", df_predict_run2)
    con.register("df_cluster_run1", df_cluster_run1)
    con.register("df_cluster_run2", df_cluster_run2)

    results = {}

    # Example 13-14: edge differences
    results["edge_diff"]      = con.sql(q_edge_difference_counts()).df()
    results["prob_comparison"] = con.sql(q_match_prob_comparison()).df()

    # Example 15: exact and partial cluster matching
    try:
        results["n_exact_matching_clusters"] = len(
            con.sql(q_exact_matching_clusters()).df()
        )
    except Exception:
        results["n_exact_matching_clusters"] = 0

    try:
        results["n_partial_matching_clusters"] = len(
            con.sql(q_partial_matching_clusters()).df()
        )
    except Exception:
        results["n_partial_matching_clusters"] = 0

    # Distributions for side-by-side charts
    results["prob_dist_run1"]    = con.sql(q_match_probability_distribution("df_predict_run1")).df()
    results["prob_dist_run2"]    = con.sql(q_match_probability_distribution("df_predict_run2")).df()
    results["cluster_sizes_run1"]= con.sql(q_cluster_size_distribution("df_cluster_run1")).df()
    results["cluster_sizes_run2"]= con.sql(q_cluster_size_distribution("df_cluster_run2")).df()

    # Gamma comparison — only include columns present in BOTH run predictions.
    # Deterministic linkage produces no gamma_ columns, so the intersection
    # may be empty. Querying a column that doesn't exist raises BinderException.
    gamma_run1 = {c for c in df_predict_run1.columns if c.startswith("gamma_")}
    gamma_run2 = {c for c in df_predict_run2.columns if c.startswith("gamma_")}
    gamma_cols  = sorted(gamma_run1 & gamma_run2)   # intersection only
    if gamma_cols:
        g_agg = ", ".join([f"ROUND(AVG({c}), 4) AS {c}" for c in gamma_cols])
        gdf1  = con.sql(f"SELECT 'Run 1' AS run, {g_agg} FROM df_predict_run1").df()
        gdf2  = con.sql(f"SELECT 'Run 2' AS run, {g_agg} FROM df_predict_run2").df()
        results["gamma_comparison"] = pd.concat([gdf1, gdf2], ignore_index=True)
    else:
        results["gamma_comparison"] = pd.DataFrame()

    con.close()
    return results
