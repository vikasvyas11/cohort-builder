# pages/p_landing.py
# Landing page — three mode cards: Standard, Upload Data, Advanced (JSON).

import streamlit as st
from modules.data_builder import (
    build_datasets, get_library_status, load_nc_voter_dataset, auto_select_nc_fields,
)
from utils.nav import _go_to, _back_button
from utils.state import clear_run_results
from utils.helpers import compute_demographic_snapshot, render_demographic_breakdowns


def page_landing() -> None:
    st.title("Cohort Builder")
    st.write(
        "Choose how you want to work. Standard mode walks you through every "
        "configuration step using the built-in fake1000 dataset. Upload mode "
        "lets you bring your own CSV or TXT files with a full EDA cleaning "
        "pipeline. Advanced mode accepts a pre-trained Splink model JSON and "
        "jumps straight to prediction and analysis."
    )
    st.divider()

    col_std, col_up, col_adv = st.columns(3, gap="large")

    # ── Standard mode ─────────────────────────────────────────────────────────
    with col_std:
        st.subheader("Standard Mode")
        st.caption("Guided workflow · built-in dataset")
        st.write(
            "Use the fake1000 dataset (1,000 synthetic UK records with name, "
            "DOB, city, email, gender, and postcode). Guided step-by-step "
            "through field selection, blocking rules, and linkage type."
        )
        if st.button("Use dummy dataset", use_container_width=True, type="primary"):
            with st.spinner("Building fake1000 dataset..."):
                try:
                    _, fakea, fakeb = build_datasets()
                    st.session_state["fakea"]         = fakea
                    st.session_state["std_fakeb"]     = fakeb
                    st.session_state["fakeb"]         = None
                    st.session_state["dataset_ready"] = True
                    st.session_state["flow"]          = "standard"
                    st.session_state["nc_field_types"] = None  # clear stale NC flag so the
                                                                # voter-dataset disclaimer doesn't
                                                                # leak into the dummy dataset flow
                    libs = get_library_status()
                    if not libs["gender_guesser"]:
                        st.warning("gender-guesser not installed: random gender used.")
                    if not libs["pgeocode"]:
                        st.warning("pgeocode not installed: synthetic postcodes used.")
                    st.success("Dataset loaded.")
                except Exception as e:
                    st.error(f"Failed to build dataset: {e}")

        st.divider()
        if st.button("Use North Carolina Voter Data", use_container_width=True):
            with st.spinner("Downloading voter_registry.csv from GitHub and running EDA…"):
                try:
                    nc_df, nc_field_types, nc_eda_log = load_nc_voter_dataset(max_rows=200_000)
                    st.session_state["fakea"]         = nc_df
                    st.session_state["std_fakeb"]     = None   # no history file; user picks link mode
                    st.session_state["fakeb"]         = None
                    st.session_state["dataset_ready"] = True
                    st.session_state["flow"]          = "standard"
                    st.session_state["nc_field_types"] = nc_field_types  # for Dataset B generation
                    # Auto-select fields from EDA-inferred field types so they
                    # always reflect actual cleaned column names, not hardcoded
                    # guesses that may not survive name normalisation.
                    sel, toggles = auto_select_nc_fields(nc_field_types, nc_df.columns)
                    st.session_state["selected_fields"]  = sel
                    st.session_state["blocking_toggles"] = toggles
                    summ = nc_eda_log.get("summary", {})
                    st.success(
                        f"NC voter data loaded and cleaned: "
                        f"{summ.get('final_rows', len(nc_df)):,} records "
                        f"(removed {summ.get('rows_removed', 0):,} during EDA). "
                        "Suggested blocking: ncid / voter_reg_num. "
                        "Choose 'Deduplication only' or generate a sample in Operation Mode."
                    )
                except Exception as e:
                    st.error(f"NC data load failed: {e}")

    # ── Upload mode ───────────────────────────────────────────────────────────
    with col_up:
        st.subheader("Upload Your Data")
        st.caption("CSV or TXT · automated EDA · your fields")
        st.write(
            "Upload one or two CSV/TXT files. The app cleans and standardises "
            "your data (field names, nulls, duplicates, dates) then guides you "
            "through field configuration and blocking rules. Supports URL and "
            "local file path loading for large files."
        )
        if st.button("Upload dataset", use_container_width=True):
            clear_run_results()
            st.session_state["flow"] = "upload"
            _go_to("upload_setup")

    # ── Advanced mode ─────────────────────────────────────────────────────────
    with col_adv:
        st.subheader("Advanced Mode")
        st.caption("Pre-trained model JSON · skip training")
        st.write(
            "Upload a Splink model JSON produced by "
            "linker.misc.save_model_to_json(). Skips all EM training and "
            "jumps straight to prediction, interactive blocking explorer, "
            "and PDF report. Trained models can be saved from the analysis page."
        )
        if st.button("Upload model JSON", use_container_width=True):
            st.session_state["flow"] = "advanced"
            _go_to("advanced_setup")

    # ── Preview if standard dataset is loaded ─────────────────────────────────
    if st.session_state["dataset_ready"] and st.session_state["flow"] == "standard":
        st.divider()
        st.subheader("Dataset A — Preview")
        st.dataframe(st.session_state["fakea"].head(5), use_container_width=True)
        _fb = st.session_state.get("std_fakeb")
        st.caption(
            f"Dataset A: {len(st.session_state['fakea']):,} records"
            + (f"  |  Dataset B available: {len(_fb):,} records (50% sample with controlled errors)"
               if _fb is not None else "")
        )

        # ── NC Voter Registry: Demographic Cohort Filter (Tier 1) ─────────────
        if st.session_state.get("nc_field_types") is not None:
            from modules.cohort_filter import (
                TIER1_CATEGORICAL_FIELDS, TIER1_RANGE_FIELDS,
                default_filters, apply_cohort_filters, cohort_summary,
                filters_to_json, filters_from_json,
                filters_to_query_param, filters_from_query_param,
            )

            # Snapshot the unfiltered load once so filters always apply
            # against the full data, never a previously-filtered subset.
            _raw_a = st.session_state.get("nc_raw_fakea")
            if _raw_a is None:
                _raw_a = st.session_state["fakea"].copy()
                st.session_state["nc_raw_fakea"] = _raw_a

            with st.expander("🔎 Demographic Cohort Filter (NC Voter Registry)", expanded=False):
                st.caption(
                    "Build a cohort using race, ethnicity, party, gender, birth year, "
                    "age, and birth state. Leave a selector empty to include all "
                    "values for that field. All active filters combine with AND."
                )

                # ── Restore from URL once per session (survives a page refresh) ──
                if "cohort_restored_from_url" not in st.session_state:
                    st.session_state["cohort_restored_from_url"] = True
                    _qp = st.query_params.get("cohort")
                    if _qp:
                        try:
                            st.session_state["cohort_filters_cache"] = filters_from_query_param(_qp)
                        except Exception:
                            pass

                _defaults = st.session_state.get("cohort_filters_cache") or default_filters(_raw_a)

                # ── Load a saved cohort definition file ──────────────────────────
                _uploaded_cohort = st.file_uploader(
                    "Load a saved cohort definition (.json)", type=["json"], key="cohort_json_upload"
                )
                if _uploaded_cohort is not None:
                    try:
                        _loaded = filters_from_json(_uploaded_cohort.read().decode("utf-8"))
                        st.session_state["cohort_filters_cache"] = _loaded
                        for _f, _sel in _loaded.get("categorical", {}).items():
                            st.session_state[f"cohort_cat_{_f}"] = _sel
                        for _f, _rng in _loaded.get("ranges", {}).items():
                            st.session_state[f"cohort_rng_{_f}"] = tuple(_rng)
                        st.session_state["cohort_inc_missing_birth_state"] = _loaded.get(
                            "birth_state_include_missing", True
                        )
                        st.success("Cohort definition loaded.")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Could not load cohort file: {_e}")

                rst_col, _ = st.columns([1, 5])
                if rst_col.button("Reset filters", key="cohort_reset"):
                    _fresh = default_filters(_raw_a)
                    for _f in TIER1_CATEGORICAL_FIELDS:
                        st.session_state[f"cohort_cat_{_f}"] = []
                    for _f in TIER1_RANGE_FIELDS:
                        st.session_state[f"cohort_rng_{_f}"] = tuple(_fresh["ranges"][_f])
                    st.session_state["cohort_inc_missing_birth_state"] = True
                    st.session_state["cohort_filters_cache"] = _fresh
                    st.rerun()

                filters = {"categorical": {}, "ranges": {}, "birth_state_include_missing": True}

                # ── Categorical filters ───────────────────────────────────────────
                cc1, cc2 = st.columns(2)
                for _i, (field, label_map) in enumerate(TIER1_CATEGORICAL_FIELDS.items()):
                    _target_col = cc1 if _i % 2 == 0 else cc2
                    _options = (sorted(_raw_a[field].dropna().unique().tolist())
                                if field in _raw_a.columns else [])
                    _fmt = (lambda code, _lm=label_map:
                            f"{code} — {_lm[code]}" if _lm and code in _lm else str(code))
                    _sel = _target_col.multiselect(
                        field.replace("_", " ").title(),
                        options=_options,
                        default=st.session_state.get(
                            f"cohort_cat_{field}", _defaults["categorical"].get(field, [])
                        ),
                        format_func=_fmt,
                        key=f"cohort_cat_{field}",
                    )
                    filters["categorical"][field] = _sel
                    if field == "birth_state":
                        _target_col.checkbox(
                            "Include records with missing birth state",
                            value=st.session_state.get("cohort_inc_missing_birth_state", True),
                            key="cohort_inc_missing_birth_state",
                        )
                        filters["birth_state_include_missing"] = st.session_state[
                            "cohort_inc_missing_birth_state"
                        ]

                # ── Range filters ──────────────────────────────────────────────────
                rc1, rc2 = st.columns(2)
                for _i, field in enumerate(TIER1_RANGE_FIELDS):
                    _target_col = rc1 if _i % 2 == 0 else rc2
                    _full_lo, _full_hi = _defaults["ranges"].get(field, [0, 0])
                    if _full_lo == _full_hi:
                        continue  # no usable data for this field
                    _rng = _target_col.slider(
                        field.replace("_", " ").title(),
                        min_value=_full_lo, max_value=_full_hi,
                        value=st.session_state.get(f"cohort_rng_{field}", (_full_lo, _full_hi)),
                        key=f"cohort_rng_{field}",
                    )
                    filters["ranges"][field] = list(_rng)

                # ── Live count + persistence ────────────────────────────────────────
                _filtered = apply_cohort_filters(_raw_a, filters)
                st.session_state["cohort_filters_cache"] = filters
                st.query_params["cohort"] = filters_to_query_param(filters)

                st.metric("Records matching this cohort", f"{len(_filtered):,} of {len(_raw_a):,}")
                st.caption(cohort_summary(filters))

                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button("Apply cohort filter to Dataset A", type="primary", key="cohort_apply"):
                        st.session_state["fakea"]     = _filtered
                        st.session_state["std_fakeb"] = None
                        st.session_state["fakeb"]     = None
                        st.success(
                            f"Dataset A filtered to {len(_filtered):,} records. "
                            "Continue below to configure fields."
                        )
                with ac2:
                    _cohort_name = st.text_input("Cohort name (for export)", value="", key="cohort_export_name")
                    st.download_button(
                        "Download cohort definition (.json)",
                        data=filters_to_json(filters, name=_cohort_name),
                        file_name=f"{_cohort_name or 'cohort'}.json",
                        mime="application/json",
                        key="cohort_download",
                    )

        st.divider()
        if st.button("Continue to dataset profile", type="primary"):
            _go_to("profile")

    st.divider()
    i1, i2, i3 = st.columns(3, gap="medium")
    with i1:
        st.markdown("**How cohort building works**")
        st.write(
            "Configure fields, blocking rules, and linkage type. "
            "The model identifies matching records and groups them into entity clusters. "
            "Export the cohort as a CSV with cluster IDs."
        )
    with i2:
        st.markdown("**Linkage and deduplication**")
        st.write(
            "Probabilistic linkage uses Fellegi-Sunter EM training to assign "
            "match probabilities. Deterministic applies exact-match rules. "
            "Both produce entity clusters for cohort building."
        )
    with i3:
        st.markdown("**What you will see**")
        st.write(
            "Match probability distributions, gamma scores, cluster metrics, "
            "Venn diagram, confusion matrix (Precision/Recall/F1/CRL), "
            "interactive blocking explorer, and a downloadable PDF report."
        )


def page_profile() -> None:
    """Pre-linkage dataset profile — sits between dataset selection and field
    configuration so the user can see what they're working with before they
    commit to fields/blocking rules, rather than only seeing demographic
    composition after linkage has already run."""
    _back_button()
    st.title("Dataset Profile")
    st.write(
        "A quick look at your dataset's composition before you configure "
        "fields and blocking rules."
    )

    fakea = st.session_state.get("fakea")
    if fakea is None or fakea.empty:
        st.warning("No dataset loaded. Please go back and select a dataset.")
        if st.button("Go back"):
            _go_to(0)
        return

    st.divider()
    _fb = st.session_state.get("std_fakeb")
    if _fb is None:
        _fb = st.session_state.get("fakeb")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records (Dataset A)", f"{len(fakea):,}")
    c2.metric("Columns", f"{fakea.shape[1]}")
    c3.metric("Dataset B", f"{len(_fb):,} records" if _fb is not None else "Not yet generated")

    st.divider()
    st.subheader("Demographic Profile")
    try:
        snapshot = compute_demographic_snapshot(fakea)
        if snapshot:
            render_demographic_breakdowns(snapshot, key_prefix="profile_demo")
        else:
            st.info("No recognised demographic columns found in this dataset.")
    except Exception as _e:
        st.warning(f"Could not compute demographic profile: {_e}")

    st.divider()
    if st.button("Continue to field configuration", type="primary"):
        _go_to(1)
