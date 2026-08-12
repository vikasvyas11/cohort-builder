# pages/p_advanced.py
# Advanced flow: upload a pre-trained Splink model JSON and jump straight
# to prediction, skipping all EM training.

import io
import json
import urllib.request

import pandas as pd
import streamlit as st
from datetime import datetime

from modules.data_builder import build_datasets, get_library_status, load_nc_voter_dataset
from modules.splink_runner import run_linkage_from_json
from modules.metrics_engine import (
    compute_intra_metrics, compute_confusion_matrix,
    compute_truth_space, compute_crl_score,
)
from modules.splink_runner import build_coverage_matrix
from utils.helpers import compute_demographic_snapshot, render_demographic_breakdowns
from utils.nav import _back_button, _go_to
from utils.state import clear_run_results


def page_advanced_setup() -> None:
    _back_button("Back to landing")
    st.title("Advanced Setup: Upload Pre-trained Model JSON")
    st.write(
        "Upload a Splink 4.x model JSON (output of linker.misc.save_model_to_json()). "
        "Prediction runs directly from the trained probabilities — no EM training."
    )
    st.divider()

    # ── JSON upload ────────────────────────────────────────────────────────────
    st.subheader("1. Upload Model JSON")
    uploaded = st.file_uploader("Splink model JSON", type=["json"],
                                 help="Produced by linker.misc.save_model_to_json(). "
                                      "Must contain trained m/u probabilities.")
    if uploaded:
        try:
            model_json = json.loads(uploaded.read())
            st.session_state["advanced_json"] = model_json
            comps   = model_json.get("comparisons", [])
            brs     = model_json.get("blocking_rules_to_generate_predictions", [])

            # ── Detect matching methodology ───────────────────────────────────
            # If this JSON was exported by this app (Save model JSON button),
            # it carries an explicit "_app_linkage_type" marker — use it as-is.
            # Otherwise, infer a sensible default: a comparison level missing
            # m_probability/u_probability strongly suggests a deterministic
            # (untrained) model. Either way this is only a DEFAULT — the radio
            # button in the next step lets the user override it.
            _marker = model_json.get("_app_linkage_type")
            if _marker in ("deterministic", "probabilistic"):
                _detected = _marker
            else:
                _has_trained_levels = any(
                    lvl.get("m_probability") is not None and lvl.get("u_probability") is not None
                    for c in comps
                    for lvl in c.get("comparison_levels", [])
                )
                _detected = "probabilistic" if _has_trained_levels else "deterministic"
            st.session_state["advanced_detected_linkage_type"] = _detected

            st.success(f"JSON loaded: {len(comps)} comparisons, {len(brs)} blocking rules.")
            with st.expander("Summary", expanded=False):
                st.write(f"**Link type:** {model_json.get('link_type','?')}")
                st.write(f"**Fields:** {', '.join(c.get('output_column_name','?') for c in comps)}")
                st.write(f"**Detected methodology:** {_detected.capitalize()}"
                         + (" (from file marker)" if _marker else " (inferred — please confirm on the next step)"))
        except Exception as e:
            st.error(f"Cannot parse JSON: {e}")

    st.divider()

    # ── Dataset selection ──────────────────────────────────────────────────────
    # Mirrors the Standard flow's landing-page dataset options exactly, so the
    # data-preparation experience is the same regardless of which flow the
    # user entered through.
    st.subheader("2. Dataset")
    if not st.session_state["dataset_ready"]:
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("Use dummy dataset (fake1000)", use_container_width=True, type="primary"):
                with st.spinner("Building fake1000 dataset..."):
                    try:
                        _, fakea, fakeb = build_datasets()
                        st.session_state["fakea"]          = fakea
                        st.session_state["fakeb"]          = fakeb
                        st.session_state["dataset_ready"]  = True
                        st.session_state["nc_field_types"] = None  # not an NC dataset
                        libs = get_library_status()
                        if not libs["gender_guesser"]:
                            st.warning("gender-guesser not installed: random gender used.")
                        if not libs["pgeocode"]:
                            st.warning("pgeocode not installed: synthetic postcodes used.")
                        st.success("Dummy dataset loaded.")
                    except Exception as e:
                        st.error(f"Failed to build dataset: {e}")
        with dc2:
            if st.button("Use NC Voter Data (200k rows)", use_container_width=True):
                with st.spinner("Downloading voter_registry.csv from GitHub and running EDA…"):
                    try:
                        nc_df, nc_field_types, nc_eda_log = load_nc_voter_dataset(max_rows=200_000)
                        st.session_state["fakea"]          = nc_df
                        st.session_state["fakeb"]          = None
                        st.session_state["dataset_ready"]  = True
                        st.session_state["nc_field_types"] = nc_field_types
                        summ = nc_eda_log.get("summary", {})
                        st.success(
                            f"NC voter data loaded and cleaned: "
                            f"{summ.get('final_rows', len(nc_df)):,} records "
                            f"(removed {summ.get('rows_removed', 0):,} during EDA)."
                        )
                    except Exception as e:
                        st.error(f"NC data load failed: {e}")
    else:
        fakea = st.session_state["fakea"]
        st.success(f"Dataset A loaded: {len(fakea):,} records.")
        if st.button("Load a different dataset"):
            st.session_state["dataset_ready"] = False
            st.session_state["fakea"] = None
            st.session_state["fakeb"] = None
            st.rerun()

    st.divider()
    if st.session_state["dataset_ready"]:
        if st.button("Continue to dataset profile", type="primary"):
            _go_to("advanced_profile")
    else:
        st.info("Load a dataset to continue.")


def page_advanced_profile() -> None:
    """Pre-linkage dataset profile for the Advanced flow — same purpose and
    content as the Standard/Upload flows' profile step, inserted before the
    methodology/run step so users see composition before running prediction."""
    _back_button()
    st.title("Dataset Profile")
    st.write(
        "A quick look at your dataset's composition before running prediction "
        "with the uploaded model."
    )

    fakea = st.session_state.get("fakea")
    if fakea is None or fakea.empty:
        st.warning("No dataset loaded. Please go back and select a dataset.")
        if st.button("Go back"):
            _go_to("advanced_setup")
        return

    st.divider()
    _fb = st.session_state.get("fakeb")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records (Dataset A)", f"{len(fakea):,}")
    c2.metric("Columns", f"{fakea.shape[1]}")
    c3.metric("Dataset B", f"{len(_fb):,} records" if _fb is not None else "Not loaded")

    st.divider()
    st.subheader("Demographic Profile")
    try:
        snapshot = compute_demographic_snapshot(fakea)
        if snapshot:
            render_demographic_breakdowns(snapshot, key_prefix="adv_profile_demo")
        else:
            st.info("No recognised demographic columns found in this dataset.")
    except Exception as _e:
        st.warning(f"Could not compute demographic profile: {_e}")

    st.divider()
    if st.button("Continue to methodology & run", type="primary"):
        _go_to("advanced_configure")


def page_advanced_configure() -> None:
    """Matching methodology, operation mode, and the run button — the tail
    end of what used to be a single page_advanced_setup(), now its own step
    after the dataset profile."""
    _back_button()
    st.title("Methodology & Run")

    if not st.session_state["dataset_ready"]:
        st.warning("No dataset loaded. Please go back and select a dataset.")
        if st.button("Go to dataset selection"):
            _go_to("advanced_setup")
        return

    # ── Matching methodology ────────────────────────────────────────────────────
    # Same explicit choice offered at Step 4 in the Standard/Upload flows —
    # pre-selected from the uploaded JSON's detected/marked methodology, but
    # always overridable so this flow behaves identically to the others.
    st.subheader("1. Matching methodology")
    _default_lt = st.session_state.get("advanced_detected_linkage_type", "probabilistic")
    lt = st.radio(
        "Methodology used to train the uploaded model:",
        ["deterministic", "probabilistic"],
        format_func=lambda x: "Deterministic (exact blocking-rule matches)" if x == "deterministic"
                              else "Probabilistic (trained match weights)",
        horizontal=True,
        index=0 if _default_lt == "deterministic" else 1,
        key="advanced_linkage_type_choice",
    )
    st.session_state["advanced_linkage_type"] = lt

    st.divider()

    # ── Operation mode + threshold ─────────────────────────────────────────────
    st.subheader("2. Operation mode")
    op = st.radio("Mode:", ["dedupe", "link_dedupe"],
                  format_func=lambda x: "Deduplication only" if x == "dedupe"
                                        else "Link and deduplicate",
                  horizontal=True,
                  index=0 if st.session_state["advanced_op_mode"] == "dedupe" else 1)
    st.session_state["advanced_op_mode"] = op
    threshold = st.slider("Cluster probability threshold", 0.5, 0.99, 0.8, 0.01)

    st.divider()

    model_json = st.session_state.get("advanced_json")
    ready      = model_json and st.session_state["dataset_ready"]
    if not ready:
        st.info("Upload a JSON file (previous steps) to continue.")

    if ready and st.button("Run prediction from uploaded model", type="primary"):
        with st.spinner("Running prediction (no training)..."):
            try:
                fakea = st.session_state["fakea"]
                fakeb = st.session_state["fakeb"] if op == "link_dedupe" else None
                chosen_lt = st.session_state.get("advanced_linkage_type", "probabilistic")
                results = run_linkage_from_json(model_json, fakea, fakeb, op, threshold,
                                                 linkage_type=chosen_lt)
                metrics = compute_intra_metrics(results["df_predict"], results["df_cluster"])
                cm      = compute_confusion_matrix(results["df_predict"], fakea, fakeb, op)
                ts      = compute_truth_space(results["df_predict"], fakea, fakeb, op)
                crl     = compute_crl_score(ts)
                fields  = results["run_config"]["selected_fields"]
                cov     = build_coverage_matrix(results["df_predict"], fields)

                clear_run_results()
                st.session_state.update({
                    "run1_results":    results,
                    "run1_metrics":    metrics,
                    "run1_cm":         cm,
                    "run1_ts":         ts,
                    "run1_crl":        crl,
                    "coverage_matrix": cov,
                    "explorer_toggles": dict(results["run_config"]["blocking_toggles"]),
                    "operation_mode":  op,
                    "linkage_type":    chosen_lt,
                })
                st.success(
                    f"Prediction complete: {results['n_edges']:,} edges, "
                    f"{results['n_clusters']:,} clusters."
                )
                _go_to(4)
            except Exception as e:
                st.error(str(e))
