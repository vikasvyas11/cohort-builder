# utils/state.py
# Session state initialisation and app-wide constants.
# All _init_state() calls are safe to call on every Streamlit rerun
# because defaults are only applied when a key is not already set.

import streamlit as st

# ── Standard flow fields ───────────────────────────────────────────────────────
ALL_FIELDS = ["first_name", "surname", "dob", "city", "email", "gender", "postcode"]

# Converted from a list to a dict (matching ADVANCED_LABELS/UPLOAD_LABELS)
# so the sidebar can mix integer and string page keys — this is what lets
# the new "profile" step slot in without renumbering any existing step.
STANDARD_LABELS = {
    0:          "Dataset Selection",
    "profile":  "Dataset Profile",
    1:          "Configure Fields and Blocking",
    2:          "Operation Mode",
    3:          "Linkage Type",
    4:          "Run Analysis",
    5:          "Compare Runs",
    6:          "Export Cohort",
}

ADVANCED_LABELS = {
    "advanced_setup":     "Advanced Setup (JSON)",
    "advanced_profile":   "Dataset Profile",
    "advanced_configure": "Methodology & Run",
    4: "Run Analysis",
    5: "Compare Runs",
    6: "Export Cohort",
}

UPLOAD_LABELS = {
    "upload_setup":     "Upload Datasets",
    "upload_eda":       "EDA and Cleaning",
    "upload_profile":   "Dataset Profile",
    "upload_configure": "Configure Fields",
    2: "Operation Mode",
    3: "Linkage Type",
    4: "Run Analysis",
    5: "Compare Runs",
    6: "Export Cohort",
}


def _init_state() -> None:
    """Set safe defaults for all session state keys on first load.
    Only sets a key if it is not already present — never overwrites user data."""
    defaults = {
        # ── Navigation ─────────────────────────────────────────────────────────
        "page":          0,
        "page_history":  [],
        "flow":          "standard",   # "standard" | "advanced" | "upload"

        # ── Standard flow datasets ──────────────────────────────────────────────
        "dataset_ready": False,
        "fakea":         None,
        "fakeb":         None,

        # ── Standard flow model config ──────────────────────────────────────────
        "selected_fields":   list(ALL_FIELDS),
        "blocking_toggles":  {f: True for f in ALL_FIELDS},
        "composite_rules":   {},
        "operation_mode":    None,
        "linkage_type":      None,
        "blocking_mode":     "OR",   # "OR" (any rule matches) or "AND" (all rules must match together)
        "hyperparams": {
            "max_iterations":  25,
            "em_convergence":  0.0001,
            "recall_estimate": 0.6,
        },

        # ── Upload flow comp types (passed to run_linkage) ──────────────────────
        # Only populated by the upload flow; None means use default comparisons.
        "upload_comp_types": None,

        # ── Advanced flow ───────────────────────────────────────────────────────
        "advanced_json":    None,
        "advanced_op_mode": "dedupe",

        # ── Shared analysis results ─────────────────────────────────────────────
        "run1_results":  None,
        "run1_metrics":  None,
        "run1_cm":       None,
        "run1_ts":       None,
        "run1_crl":      {},
        "run2_results":  None,
        "run2_metrics":  None,
        "run2_blocking_toggles": None,

        # ── Interactive blocking explorer ───────────────────────────────────────
        "coverage_matrix":    None,
        "explorer_toggles":   {},
        "explorer_threshold": 0.8,

        # ── Upload flow (all keys prefixed up_) ─────────────────────────────────
        # Raw data (before EDA)
        "up_raw_a":        None,
        "up_raw_b":        None,
        # User-chosen ID columns
        "up_id_col_a":     None,
        "up_id_col_b":     None,
        # Link mode
        "up_link_mode":    "dedupe_only",  # "dedupe_only"|"link_uploaded"|"link_sample"
        # Cleaned data (after EDA)
        "up_clean_a":      None,
        "up_clean_b":      None,
        # EDA results
        "up_types_a":      {},
        "up_types_b":      {},
        "up_eda_a":        {},
        "up_eda_b":        {},
        "up_corr_a":       [],
        "up_corr_b":       [],
        # Model configuration chosen on upload_configure page
        "up_sel_fields":   [],
        "up_comp_types":   {},
        "up_block_tog":    {},
        "up_comp_rules":   {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_run_results() -> None:
    """Wipe all cached run results so analysis page starts fresh.
    Call this whenever datasets or fields change."""
    for k in [
        "run1_results", "run1_metrics", "run1_cm", "run1_ts", "run1_crl",
        "run2_results", "run2_metrics", "run2_blocking_toggles",
        "coverage_matrix", "explorer_toggles",
    ]:
        st.session_state[k] = None
