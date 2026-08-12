# pages/p_upload.py
# Upload flow: dataset loading, EDA, field configuration.
import io
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from utils.helpers import compute_demographic_snapshot, render_demographic_breakdowns
from modules.eda_engine import (
    run_full_eda, find_high_correlation_pairs,
    suggest_comparison_types, suggest_blocking_rules,
    introduce_errors_for_sample,
)
from utils.nav import _back_button, _go_to
from utils.state import clear_run_results


# =============================================================================
# FILE LOADING HELPERS
# =============================================================================

def _read_file(file_obj) -> pd.DataFrame:
    raw = file_obj.read()
    name = file_obj.name.lower()
    for sep in (["\t", ","] if name.endswith(".txt") else [",", "\t"]):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str,
                             encoding="latin-1", on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse {file_obj.name} as CSV or TSV.")


def _read_url(url: str, nrows: int, sep: str | None = None) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read()
    if sep is None:
        sep = "\t" if url.lower().endswith(".txt") else ","
    return pd.read_csv(io.BytesIO(raw), sep=sep, nrows=nrows or None,
                       dtype=str, encoding="latin-1", on_bad_lines="skip")


def _read_local(path: str, nrows: int, sep: str | None = None) -> pd.DataFrame:
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if sep is None:
        sep = "\t" if path.lower().endswith(".txt") else ","
    return pd.read_csv(path, sep=sep, nrows=nrows or None,
                       dtype=str, encoding="latin-1", on_bad_lines="skip")


def _ensure_unique_id(df: pd.DataFrame, id_col: str | None, prefix: str) -> pd.DataFrame:
    df = df.copy()
    if id_col and id_col in df.columns and id_col != "unique_id":
        df = df.rename(columns={id_col: "unique_id"})
    elif "unique_id" not in df.columns:
        df.insert(0, "unique_id",
                  prefix + "_" + pd.Series(range(len(df))).astype(str))
    # Defensive cast: a user-chosen ID column (e.g. a numeric customer ID)
    # would otherwise carry a non-string dtype into Splink, which is the
    # same known Splink gripe as fake1000's int unique_id.
    df["unique_id"] = df["unique_id"].astype(str)
    return df


def _sep_selectbox(key: str) -> str | None:
    label = st.selectbox("Separator", key=key,
                         options=["Auto-detect", "Tab (TXT/TSV)", "Comma (CSV)",
                                  "Pipe (|)", "Semicolon (;)"])
    return {
        "Auto-detect": None,
        "Tab (TXT/TSV)": "\t",
        "Comma (CSV)": ",",
        "Pipe (|)": "|",
        "Semicolon (;)": ";",
    }[label]


# =============================================================================
# PAGE 1 — UPLOAD SETUP
# =============================================================================

def page_upload_setup() -> None:
    _back_button("Back to landing")
    st.title("Upload Your Datasets")
    st.write(
        "Load Dataset A (required) and optionally Dataset B. "
        "Supported sources: file upload, URL, or local file path. "
        "All formats: CSV or tab-delimited TXT."
    )
    st.divider()

    # ── Load Dataset A ─────────────────────────────────────────────────────────
    st.subheader("1. Dataset A (required)")
    tab_file_a, tab_url_a, tab_path_a = st.tabs(
        ["Upload file", "Load from URL", "Local file path"]
    )

    with tab_file_a:
        f = st.file_uploader("CSV or TXT file", type=["csv", "txt"], key="fu_a")
        if f:
            try:
                df = _read_file(f)
                st.session_state["up_raw_a"] = df
                st.success(f"Dataset A: {len(df):,} rows × {df.shape[1]} columns")
                with st.expander("Preview"):
                    st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(str(e))

    with tab_url_a:
        url_a = st.text_input("URL (direct link to CSV or TXT)", key="url_a")
        nr_a = st.number_input("Max rows", 1000, 5_000_000, 50_000, key="nr_a")
        sep_a = _sep_selectbox("sep_a")
        if url_a and st.button("Load Dataset A from URL", key="btn_url_a"):
            with st.spinner("Downloading..."):
                try:
                    df = _read_url(url_a, nr_a, sep_a)
                    st.session_state["up_raw_a"] = df
                    st.success(f"Dataset A: {len(df):,} rows × {df.shape[1]} columns")
                except Exception as e:
                    st.error(str(e))

    with tab_path_a:
        st.caption("Use this for files larger than Streamlit's 200 MB upload limit.")
        path_a = st.text_input("Absolute file path", key="path_a",
                               placeholder=r"C:\Users\...\data.txt")
        nr_pa = st.number_input("Max rows (0 = all)", 0, 5_000_000, 100_000, key="nr_pa")
        sep_pa = _sep_selectbox("sep_pa")
        if path_a and st.button("Load Dataset A from path", key="btn_path_a"):
            with st.spinner("Reading file..."):
                try:
                    df = _read_local(path_a, nr_pa, sep_pa)
                    st.session_state["up_raw_a"] = df
                    st.success(f"Dataset A: {len(df):,} rows × {df.shape[1]} columns")
                except Exception as e:
                    st.error(str(e))

    raw_a = st.session_state.get("up_raw_a")
    if raw_a is None:
        st.info("Load Dataset A above to continue.")
        return

    st.divider()

    # ── ID column for Dataset A ────────────────────────────────────────────────
    st.subheader("2. Unique identifier for Dataset A")
    id_opts_a = ["[Auto-generate unique_id]"] + list(raw_a.columns)
    id_a = st.selectbox("ID column (Dataset A):", id_opts_a, key="id_sel_a",
                        help="Column that uniquely identifies each record. Auto-generate if none exists.")
    st.session_state["up_id_col_a"] = None if id_a.startswith("[") else id_a
    st.divider()

    # ── Dataset B ──────────────────────────────────────────────────────────────
    st.subheader("3. Dataset B (optional)")
    link_mode = st.radio(
        "How to set up Dataset B:",
        options=[
            "Deduplicate Dataset A only (no Dataset B)",
            "Upload / load Dataset B",
            "Generate a customized error sample of Dataset A (for testing linking)",
        ],
        key="up_link_mode_radio",
    )
    mode_map = {
        "Deduplicate Dataset A only (no Dataset B)": "dedupe_only",
        "Upload / load Dataset B": "link_uploaded",
        "Generate a customized error sample of Dataset A (for testing linking)": "link_sample",
    }
    st.session_state["up_link_mode"] = mode_map[link_mode]

    if link_mode == "Upload / load Dataset B":
        tab_file_b, tab_url_b, tab_path_b = st.tabs(
            ["Upload file", "Load from URL", "Local file path"]
        )
        with tab_file_b:
            fb = st.file_uploader("CSV or TXT file (Dataset B)", type=["csv", "txt"], key="fu_b")
            if fb:
                try:
                    df_b = _read_file(fb)
                    st.session_state["up_raw_b"] = df_b
                    st.success(f"Dataset B: {len(df_b):,} rows × {df_b.shape[1]} columns")
                    with st.expander("Preview"):
                        st.dataframe(df_b.head(5), use_container_width=True)
                except Exception as e:
                    st.error(str(e))

        with tab_url_b:
            url_b = st.text_input("URL for Dataset B", key="url_b")
            nr_b = st.number_input("Max rows (B)", 1000, 5_000_000, 50_000, key="nr_b")
            sep_b = _sep_selectbox("sep_b")
            if url_b and st.button("Load Dataset B from URL", key="btn_url_b"):
                with st.spinner("Downloading Dataset B..."):
                    try:
                        df_b = _read_url(url_b, nr_b, sep_b)
                        st.session_state["up_raw_b"] = df_b
                        st.success(f"Dataset B: {len(df_b):,} rows × {df_b.shape[1]} columns")
                    except Exception as e:
                        st.error(str(e))

        with tab_path_b:
            path_b = st.text_input("Absolute path for Dataset B", key="path_b",
                                   placeholder=r"C:\Users\...\data_b.txt")
            nr_pb = st.number_input("Max rows (0 = all, B)", 0, 5_000_000, 100_000, key="nr_pb")
            sep_pb = _sep_selectbox("sep_pb")
            if path_b and st.button("Load Dataset B from path", key="btn_path_b"):
                with st.spinner("Reading Dataset B..."):
                    try:
                        df_b = _read_local(path_b, nr_pb, sep_pb)
                        st.session_state["up_raw_b"] = df_b
                        st.success(f"Dataset B: {len(df_b):,} rows × {df_b.shape[1]} columns")
                    except Exception as e:
                        st.error(str(e))

        raw_b = st.session_state.get("up_raw_b")
        if raw_b is not None:
            id_opts_b = ["[Auto-generate unique_id]"] + list(raw_b.columns)
            id_b = st.selectbox("ID column (Dataset B):", id_opts_b, key="id_sel_b")
            st.session_state["up_id_col_b"] = None if id_b.startswith("[") else id_b

    st.divider()
    if st.button("Continue to EDA and Cleaning", type="primary"):
        for k in ["up_clean_a", "up_clean_b", "up_types_a", "up_types_b",
                  "up_eda_a", "up_eda_b", "up_corr_a", "up_corr_b",
                  "up_sel_fields", "up_comp_types", "up_block_tog", "up_comp_rules"]:
            st.session_state[k] = None if k.endswith(("_a", "_b")) else {} if k.endswith(
                ("types", "rules", "tog")) else []
        clear_run_results()
        _go_to("upload_eda")


# =============================================================================
# PAGE 2 — EDA AND CLEANING
# =============================================================================

def _eda_section(label: str, raw_df: pd.DataFrame, id_col: str | None,
                 key_clean: str, key_types: str, key_eda: str, key_corr: str,
                 key_dropped: str) -> None:
    if st.session_state.get(key_clean) is None:
        with st.spinner(f"Running EDA on {label}..."):
            df_c, ftypes, _, log = run_full_eda(raw_df.copy(), id_col=id_col)
            id_type_cols = [c for c, t in ftypes.items() if t == "id"]
            corr = find_high_correlation_pairs(df_c, id_type_cols)
            st.session_state[key_clean] = df_c
            st.session_state[key_types] = ftypes
            st.session_state[key_eda] = log
            st.session_state[key_corr] = corr

    df_c = st.session_state[key_clean]
    ftypes = st.session_state[key_types]
    log = st.session_state[key_eda]
    corr = st.session_state[key_corr]
    summ = log.get("summary", {})

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Original rows", f"{summ.get('original_rows', 0):,}")
    k2.metric("Rows removed", f"{summ.get('rows_removed', 0):,}",
              delta=f"-{summ.get('rows_removed', 0):,}", delta_color="inverse")
    k3.metric("Remaining rows", f"{summ.get('final_rows', 0):,}")
    k4.metric("Cols removed", f"{summ.get('cols_removed', 0):,}",
              delta=f"-{summ.get('cols_removed', 0):,}", delta_color="inverse")

    with st.expander(f"Step-by-step cleaning log ({label})", expanded=False):
        changed = log.get("field_names", {}).get("changed", {})
        if changed:
            st.write("**Field names standardised:**")
            for o, n in changed.items():
                st.write(f"  `{o}` → `{n}`")
        nr = log.get("null_rows_removed", {})
        st.write(f"Columns dropped (100% null): {len(log.get('null_columns_dropped', []))}")
        st.write(f"Rows removed (100% null): {nr.get('100%_null', 0):,}")
        st.write(f"Rows removed (n-1 null):  {nr.get('n-1_null', 0):,}")
        st.write(f"Rows removed (n-2 null):  {nr.get('n-2_null', 0):,}")
        st.write(f"Duplicate rows removed: {log.get('duplicates_removed', 0):,}")
        for col, fmt in log.get("dates_standardised", {}).items():
            st.write(f"Date standardised: `{col}` → format `{fmt}`")

    with st.expander(f"Detected field types ({label})", expanded=False):
        ct = suggest_comparison_types(ftypes)
        type_df = pd.DataFrame(
            [{"Column": c, "Type": t,
              "Comparison suggestion": ct.get(c, "ExactMatch")}
             for c, t in ftypes.items()]
        )
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    if corr:
        st.write(f"**High-correlation field pairs in {label}:**")
        dropped = set(st.session_state.get(key_dropped) or set())
        for col_a, col_b, score in corr:
            with st.container(border=True):
                cc1, cc2, cc3 = st.columns([2, 2, 2])
                cc1.write(f"**{col_a}**")
                cc2.write(f"**{col_b}**")
                cc3.write(f"Similarity: `{score:.4f}`")
                keep = st.radio(
                    f"Keep which field? ({label})",
                    options=[col_a, col_b, "Keep both"],
                    horizontal=True,
                    key=f"corr_{label}_{col_a}_{col_b}",
                )
                if keep == col_a:
                    dropped.add(col_b)
                elif keep == col_b:
                    dropped.add(col_a)
        st.session_state[key_dropped] = dropped
        if dropped:
            df_c = df_c.drop(columns=[c for c in dropped if c in df_c.columns])
            st.session_state[key_clean] = df_c
            ftypes = {c: t for c, t in ftypes.items() if c not in dropped}
            st.session_state[key_types] = ftypes

    st.dataframe(df_c.head(8), use_container_width=True)
    st.caption(f"{label}: {df_c.shape[0]:,} rows × {df_c.shape[1]} columns")
    csv = df_c.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(f"Download cleaned {label} CSV", data=csv,
                       file_name=f"cleaned_{label.lower().replace(' ', '_')}_{ts}.csv",
                       mime="text/csv", key=f"dl_{label}_{key_clean}")


def page_eda() -> None:
    _back_button("Back to upload")
    st.title("EDA and Data Cleaning")
    st.write(
        "The cleaning pipeline runs automatically: field name standardisation, "
        "null removal (100%, n-1, n-2), text cleaning, deduplication, and date "
        "standardisation. Results are cached — click 'Re-run EDA' to refresh."
    )

    raw_a = st.session_state.get("up_raw_a")
    if raw_a is None:
        st.error("No dataset loaded. Go back and upload Dataset A.")
        if st.button("Go to upload"):
            _go_to("upload_setup")
        return

    col_rerun, _ = st.columns([1, 5])
    if col_rerun.button("Re-run EDA (clears cache)", key="rerun_eda"):
        for k in ["up_clean_a", "up_clean_b", "up_types_a", "up_types_b",
                  "up_eda_a", "up_eda_b", "up_corr_a", "up_corr_b"]:
            st.session_state[k] = None

    # ── Dataset A EDA ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Dataset A — Cleaning Results")
    _eda_section("Dataset A", raw_a,
                 st.session_state.get("up_id_col_a"),
                 "up_clean_a", "up_types_a", "up_eda_a", "up_corr_a", "up_dropped_a")

    # ── Dataset B EDA (if uploaded or sampled) ─────────────────────────────────
    mode = st.session_state.get("up_link_mode", "dedupe_only")
    raw_b = st.session_state.get("up_raw_b")

    if mode == "link_uploaded" and raw_b is not None:
        st.divider()
        st.subheader("Dataset B — Cleaning Results")
        _eda_section("Dataset B", raw_b,
                     st.session_state.get("up_id_col_b"),
                     "up_clean_b", "up_types_b", "up_eda_b", "up_corr_b", "up_dropped_b")

    elif mode == "link_sample":
        st.divider()
        st.subheader("Dataset B — Custom Error Sample Generation")
        clean_a = st.session_state.get("up_clean_a")
        ftypes_a = st.session_state.get("up_types_a", {})

        if clean_a is not None:
            # ── Only offer fields that exist in Dataset A (since B is derived from A)
            # Exclude identifier/meta columns — corrupting them breaks linkage logic
            EXCLUDE = {"unique_id", "cluster", "source_dataset"}
            # For link_uploaded, restrict to common cols if B already loaded
            clean_b_loaded = st.session_state.get("up_clean_b")
            if clean_b_loaded is not None and st.session_state.get("up_link_mode") == "link_uploaded":
                # Uploaded Dataset B: only common columns can be compared/corrupted
                usable_fields = sorted(
                    c for c in (set(clean_a.columns) & set(clean_b_loaded.columns))
                    if c not in EXCLUDE
                )
            elif clean_b_loaded is not None:
                # Derived Dataset B already exists: restrict to its columns (the
                # intersection of A and B, since B was sampled from A)
                usable_fields = sorted(
                    c for c in (set(clean_a.columns) & set(clean_b_loaded.columns))
                    if c not in EXCLUDE
                    and ftypes_a.get(c, "text") != "id"
                )
            else:
                # B not yet generated: offer all non-id, non-meta A columns
                usable_fields = [
                    c for c in clean_a.columns
                    if c not in EXCLUDE
                    and ftypes_a.get(c, "text") != "id"
                ]

            st.write(
                f"Select which of the **{len(usable_fields)} shared fields** to corrupt. "
                "Only fields present in Dataset A are eligible."
            )

            # ── Select All / Deselect All ───────────────────────────────────
            sel_col, desel_col, _ = st.columns([1, 1, 6])
            if sel_col.button("Select All", key="err_sel_all"):
                for c in usable_fields:
                    st.session_state[f"up_chk_{c}"] = True
            if desel_col.button("Deselect All", key="err_desel_all"):
                for c in usable_fields:
                    st.session_state[f"up_chk_{c}"] = False

            # ── Per-field checkboxes + rate sliders ─────────────────────────
            chosen_upload_errs = {}
            for col in usable_fields:
                ftype = ftypes_a.get(col, "unknown")
                # Sensible defaults by detected field type
                # Default 0 so users explicitly choose which fields to corrupt
                default_pct = 0

                with st.container(border=True):
                    uc1, uc2 = st.columns([1, 2])
                    chk = uc1.checkbox(
                        f"`{col}` ({ftype})",
                        value=st.session_state.get(f"up_chk_{col}", False),
                        key=f"up_chk_{col}",
                    )
                    if chk:
                        pct = uc2.slider(
                            "Corruption rate (%)", 0, 50, default_pct,
                            step=1, key=f"up_sld_{col}",
                        )
                        chosen_upload_errs[col] = pct / 100.0
                    else:
                        chosen_upload_errs[col] = 0.0

            sample_frac = st.slider(
                "Sample fraction for Dataset B", 0.1, 0.9, 0.5, 0.05,
                key="err_sample_frac",
                help="What fraction of Dataset A records to include in Dataset B.",
            )

            active_err_fields = [c for c, r in chosen_upload_errs.items() if r > 0]
            st.caption(
                f"Errors will be introduced into {len(active_err_fields)} field(s): "
                f"{', '.join(active_err_fields) if active_err_fields else 'none (all deselected)'}"
            )

            if st.button("Generate Dataset B with Custom Error Rates",
                         use_container_width=True, type="primary"):
                with st.spinner("Generating error sample..."):
                    try:
                        df_b = introduce_errors_for_sample(
                            df=clean_a,
                            field_types=ftypes_a,
                            sample_frac=sample_frac,
                            seed=42,
                            error_rates=chosen_upload_errs,
                        )
                        st.session_state["up_clean_b"] = df_b
                        st.success(
                            f"Dataset B generated: {len(df_b):,} records. "
                            f"Errors in: {', '.join(active_err_fields) or 'none'}."
                        )
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

        if st.session_state.get("up_clean_b") is not None:
            df_b = st.session_state["up_clean_b"]
            with st.expander("Dataset B preview", expanded=False):
                st.dataframe(df_b.head(5), use_container_width=True)
            st.caption(f"Dataset B: {len(df_b):,} rows × {df_b.shape[1]} columns")

    st.divider()
    if st.session_state.get("up_clean_a") is not None:
        if st.button("Continue to dataset profile", type="primary"):
            _go_to("upload_profile")


# =============================================================================
# PAGE 2.5 — DATASET PROFILE
# =============================================================================

def page_upload_profile() -> None:
    """Pre-linkage dataset profile — same purpose as the Standard flow's
    profile step, shown on the CLEANED (post-EDA) dataset."""
    _back_button("Back to EDA")
    st.title("Dataset Profile")
    st.write(
        "A quick look at your cleaned dataset's composition before you "
        "configure fields and blocking rules."
    )

    clean_a = st.session_state.get("up_clean_a")
    if clean_a is None or clean_a.empty:
        st.warning("No cleaned dataset available. Please go back and run EDA first.")
        if st.button("Go back"):
            _go_to("upload_eda")
        return

    st.divider()
    clean_b = st.session_state.get("up_clean_b")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records (Dataset A)", f"{len(clean_a):,}")
    c2.metric("Columns", f"{clean_a.shape[1]}")
    c3.metric("Dataset B", f"{len(clean_b):,} records" if clean_b is not None else "Not provided")

    st.divider()
    st.subheader("Demographic Profile")
    try:
        snapshot = compute_demographic_snapshot(clean_a)
        if snapshot:
            render_demographic_breakdowns(snapshot, key_prefix="up_profile_demo")
        else:
            st.info("No recognised demographic columns found in this dataset.")
    except Exception as _e:
        st.warning(f"Could not compute demographic profile: {_e}")

    st.divider()
    if st.button("Continue to field configuration", type="primary"):
        _go_to("upload_configure")


# =============================================================================
# PAGE 3 — FIELD CONFIGURATION
# =============================================================================

def page_upload_configure() -> None:
    _back_button("Back to EDA")
    st.title("Configure Fields and Blocking Rules")

    clean_a = st.session_state.get("up_clean_a")
    if clean_a is None:
        st.warning("No cleaned dataset. Go back to EDA first.")
        if st.button("Go to EDA"):
            _go_to("upload_eda")
        return

    ftypes_a = st.session_state.get("up_types_a", {})
    mode = st.session_state.get("up_link_mode", "dedupe_only")
    clean_b = st.session_state.get("up_clean_b")

    cols_a = set(clean_a.columns)
    cols_b = set(clean_b.columns) if clean_b is not None else cols_a
    common_cols = cols_a & cols_b

    if mode == "dedupe_only":
        available_cols = cols_a
    else:
        available_cols = common_cols

    # ── 1. Unique ID ──────────────────────────────────────────────────────────
    st.subheader("1. Unique identifier column")
    id_type_cols = [c for c, t in ftypes_a.items() if t == "id"]
    non_id_cols = [c for c in clean_a.columns if c not in id_type_cols]
    id_options = ["[Auto-generate]"] + id_type_cols + non_id_cols
    current_id = st.session_state.get("up_id_col_a")
    sel_id = st.selectbox(
        "Dataset A unique ID column:",
        options=id_options,
        index=(0 if current_id is None else
               (id_options.index(current_id) if current_id in id_options else 0)),
        help="This becomes the 'unique_id' column Splink requires.",
    )
    id_col_a = None if sel_id.startswith("[") else sel_id
    st.session_state["up_id_col_a"] = id_col_a
    st.caption(f"Using: {'auto-generated unique_id' if id_col_a is None else id_col_a}")

    if clean_b is not None and mode != "dedupe_only":
        cols_b_list = list(clean_b.columns)
        id_opts_b = ["[Auto-generate]"] + cols_b_list
        current_id_b = st.session_state.get("up_id_col_b")
        sel_id_b = st.selectbox(
            "Dataset B unique ID column:",
            options=id_opts_b,
            index=(0 if current_id_b is None else
                   (id_opts_b.index(current_id_b) if current_id_b in id_opts_b else 0)),
        )
        id_col_b = None if sel_id_b.startswith("[") else sel_id_b
        st.session_state["up_id_col_b"] = id_col_b

    for _k in [id_col_a, "unique_id"]:
        available_cols.discard(_k)

    st.divider()

    # ── 2. Field selection + comparison types ──────────────────────────────────
    st.subheader("2. Comparison fields and types")
    COMP_OPTIONS = [
        "NameComparison", "DateOfBirthComparison", "ExactMatch",
        "LevenshteinAtThresholds", "JaroWinklerAtThresholds",
        "EmailComparison", "PostcodeComparison",
    ]

    if mode != "dedupe_only":
        a_only_cols = cols_a - cols_b - {id_col_a, "unique_id"}
        if a_only_cols:
            st.info(
                f"Link mode active. The following Dataset A columns are NOT in "
                f"Dataset B and cannot be used for comparison: "
                f"{', '.join(sorted(a_only_cols)[:8])}{'...' if len(a_only_cols) > 8 else ''}"
            )

    comp_suggestions = suggest_comparison_types(ftypes_a)
    eligible = sorted(c for c in available_cols if c not in (id_col_a, "unique_id"))

    sa_col, da_col, _ = st.columns([1, 1, 6])
    if sa_col.button("Select All", key="up_sel_all"):
        for f in eligible:
            st.session_state[f"up_inc_{f}"] = True
    if da_col.button("Deselect All", key="up_desel_all"):
        for f in eligible:
            st.session_state[f"up_inc_{f}"] = False

    selected_fields = []
    comp_types = {}
    for col in eligible:
        suggested = comp_suggestions.get(col, "ExactMatch")
        r1, r2, r3 = st.columns([2, 2, 1])
        include = r1.checkbox(
            col,
            value=st.session_state.get(f"up_inc_{col}", True),
            key=f"up_inc_{col}",
        )
        if include:
            ct = r2.selectbox(
                "", COMP_OPTIONS,
                index=COMP_OPTIONS.index(suggested) if suggested in COMP_OPTIONS else 2,
                key=f"up_type_{col}", label_visibility="collapsed",
            )
            r3.caption(ftypes_a.get(col, ""))
            selected_fields.append(col)
            comp_types[col] = ct

    if not selected_fields:
        st.error("Select at least one comparison field.")
        return

    st.divider()

    # ── 3. Blocking rules ──────────────────────────────────────────────────────
    st.subheader("3. Blocking rules")
    st.write(
        "Each toggle creates an independent blocking rule for that field. "
        "For link mode, only fields present in BOTH datasets are shown."
    )
    block_suggestions = suggest_blocking_rules(ftypes_a)
    block_toggles = {}
    bt_cols = st.columns(3)
    for i, field in enumerate(selected_fields):
        enabled = bt_cols[i % 3].toggle(
            field,
            value=st.session_state.get("up_block_tog", {}).get(field,
                                                               block_suggestions.get(field, False)),
            key=f"up_block_{field}",
        )
        block_toggles[field] = enabled

    if not any(block_toggles.values()):
        st.error("At least one blocking rule must be enabled.")
        return

    # ── 4. Composite blocking rules ────────────────────────────────────────────
    comp_rules = dict(st.session_state.get("up_comp_rules") or {})
    with st.expander("Composite blocking rules", expanded=False):
        if len(selected_fields) >= 2:
            cb1, cb2, cb3 = st.columns([2, 2, 1])
            cf1 = cb1.selectbox("Field 1", selected_fields, key="up_cb_f1")
            cf2_opts = [f for f in selected_fields if f != cf1]
            cf2 = cb2.selectbox("Field 2", cf2_opts, key="up_cb_f2")
            if cb3.button("Add", key="up_cb_add"):
                comp_rules[f"{cf1}+{cf2}"] = True
        for key in list(comp_rules.keys()):
            parts = key.split("+")
            cr1, cr2 = st.columns([4, 1])
            cr1.code(f'l."{parts[0]}" = r."{parts[0]}" AND l."{parts[1]}" = r."{parts[1]}"')
            if cr2.button("Remove", key=f"up_rm_{key}"):
                del comp_rules[key]

    # ── 5. Training hyperparameters (matches Standard flow's Step 2 exactly) ──
    with st.expander("Training hyperparameters (probabilistic mode only)", expanded=False):
        hp = st.session_state.get("hyperparams", {})
        nhp = {}
        nhp["max_iterations"] = st.number_input(
            "Max EM iterations", 5, 500,
            value=hp.get("max_iterations", 25), step=5, key="up_hp_max_iter")
        nhp["em_convergence"] = st.number_input(
            "EM convergence", 1e-8, 0.01,
            value=hp.get("em_convergence", 0.0001), format="%.8f", key="up_hp_convergence")
        nhp["recall_estimate"] = st.slider(
            "Recall estimate for prior", 0.1, 0.99,
            value=hp.get("recall_estimate", 0.6), step=0.05, key="up_hp_recall")
        st.session_state["hyperparams"] = nhp

    _commit_upload_state(clean_a, clean_b, id_col_a,
                         st.session_state.get("up_id_col_b"),
                         selected_fields, block_toggles,
                         comp_types, comp_rules, mode)

    st.divider()
    st.info(
        f"**Ready to analyse:** {len(selected_fields)} comparison field(s), "
        f"{sum(block_toggles.values())} blocking rule(s) active."
    )

    if st.button("Continue to operation mode", type="primary"):
        clear_run_results()
        _go_to(2)


def _commit_upload_state(
        clean_a, clean_b, id_col_a, id_col_b,
        selected_fields, block_toggles, comp_types, comp_rules, mode,
) -> None:
    df_a = _ensure_unique_id(clean_a, id_col_a, "A")
    df_a["source_dataset"] = "A"
    # Assign cluster column using unique_id as ground-truth entity label.
    # Must be set on Dataset A BEFORE Dataset B is derived so B can inherit
    # matching cluster values, enabling the confusion matrix to work.
    if "cluster" not in df_a.columns:
        df_a["cluster"] = df_a["unique_id"].astype(str)

    df_b = None
    if clean_b is not None and mode != "dedupe_only":
        df_b = _ensure_unique_id(clean_b, id_col_b, "B")
        df_b["source_dataset"] = "B"
        # ── Assign cluster for Dataset B ──────────────────────────────────────
        # For link_sample mode: the cluster was set in introduce_errors_for_sample
        # BEFORE the _B suffix was appended to unique_id, so strip it to get the
        # matching A-side cluster value.  For separately uploaded Dataset B files,
        # use the raw unique_id (before suffixing) as the cluster label; this
        # won't give a perfect ground truth but prevents the "unavailable" error.
        if "cluster" not in df_b.columns:
            # Use the pre-suffix unique_id as the cluster ground truth
            df_b["cluster"] = df_b["unique_id"].astype(str).apply(
                lambda x: x.replace("_B", "") if x.endswith("_B") else x
            )
        # Suffix unique_id AFTER capturing cluster so B IDs don't collide with A
        if not df_b["unique_id"].astype(str).str.endswith("_B").all():
            df_b["unique_id"] = df_b["unique_id"].astype(str) + "_B"
        # For link_sample: the cluster stored in the sample equals the A unique_id
        # (set before suffixing in introduce_errors_for_sample). No change needed.

    st.session_state["fakea"] = df_a
    st.session_state["fakeb"] = df_b
    st.session_state["dataset_ready"] = True
    st.session_state["selected_fields"] = selected_fields
    st.session_state["blocking_toggles"] = block_toggles
    st.session_state["composite_rules"] = comp_rules
    st.session_state["upload_comp_types"] = comp_types
    st.session_state["up_sel_fields"] = selected_fields
    st.session_state["up_comp_types"] = comp_types
    st.session_state["up_block_tog"] = block_toggles
    st.session_state["up_comp_rules"] = comp_rules