# Cohort Builder

A Streamlit application for record linkage and deduplication using Splink and DuckDB. Built at Swansea University as an MVP for cohort construction workflows, targeting both non-technical and technical users.

## Access it online at https://cb-swansea.streamlit.app/


## Three workflows

**Standard mode** — guided workflow for non-technical users. Loads either the built-in fake1000 dataset or the North Carolina voter registry dataset, offers an optional demographic cohort filter and a pre-linkage Dataset Profile step, then walks through field selection, blocking rules (including OR/AND blocking mode), operation mode, and linkage type, before producing analysis and an exportable cohort. For the NC voter registry dataset, a derived Dataset B can be generated with deterministic, per-record error introduction (see "Dataset B options (NC Voter Registry flow)" below).  
**Upload mode** — bring your own data. Upload one or two CSV files, run the automated EDA cleaning pipeline, configure fields and blocking rules, then follow the same analysis, comparison, and export steps as standard mode.  
**Advanced mode** — for power users with a pre-trained Splink model. Upload a model JSON file, skip all training, and jump straight to prediction, interactive blocking exploration, and export. Models trained in standard or upload mode can be saved as JSON and reused here.

--- 
**Upload mode** - for users to upload their own datasets, clean and standardise the fields, and then run the analysis. Users can generate an error prone dataset from their original dataset to test out the linkage model.

You can save your exisiting model on Splink using the following code:
```
# Save model to JSON
linker.misc.save_model_to_json("test_splink_model.json", overwrite=True)
```

Before uploading, please check if your file has the following format to ensure consistency between runs and the app accepts the uploaded JSON file
```
{
  "link_type": "dedupe_only",
  "unique_id_column_name": "unique_id",
  "probability_two_random_records_match": 0.000812,
  "comparisons": [ ... ],
  "blocking_rules_to_generate_predictions": [ ... ]
}
```
Please ensure, comparisons contains m and u probabilities. 

---

## Dataset Profile & Demographic Cohort Filtering (NC Voter Registry)

Before configuring fields or blocking rules, Standard mode offers two things specific to the NC voter registry dataset:

- **Demographic Cohort Filter** — build a cohort from Tier-1 demographic fields (`gender_code`, `race_code`, `ethnic_code`, `party_cd`, `birth_year`, `age_at_year_end`, `birth_state`) before linkage runs at all. Filters combine as AND across fields, OR within a field's selected values. Cohort definitions can be exported/imported as JSON for reuse, and persist across a same-tab refresh via the URL.
- **Dataset Profile step** — a dedicated navigation step showing the demographic composition of whatever's currently loaded (post-filter, if applied), so you have insight into the data before committing to field and blocking choices — not just after linkage has already run.

---

## Features

- Probabilistic linkage via Expectation-Maximisation (Splink 4.x + DuckDB backend)
- Deterministic linkage with exact-match blocking rules — requires agreement on multiple selected fields (not just a single blocking-rule hit) to prevent unrelated records being chained into one oversized cluster
- Deduplication only, or cross-dataset linkage (Dataset A + Dataset B)
- North Carolina voter registry dataset (200k rows) available directly from Standard mode, with automated EDA and a derived Dataset B generator
- Three-mode sidebar switcher to move between Standard, Upload, and Advanced flows at any time
- Back navigation with history stack on every page
- Save trained model as JSON for reuse in Advanced mode
- Interactive blocking explorer: toggle rules on/off, live df_predict table update, one-click re-clustering
- Composite blocking rules (e.g. first_name + surname as a single rule)
- Exposed training hyperparameters: EM iterations, convergence threshold, recall estimate
- Confusion matrix with ground truth from the cluster column: TP, FP, FN, Precision, Recall, F1, F*, FDR, FNR
- Precision-Recall curve and CRL (Composite Reliability of Linkage) score
- Full metrics suite covering linkage-metrics: match weight histogram, gamma scores, cluster size distribution, confusion matrix, Venn diagram, inter-run edge comparison
- Clickable sidebar navigation with back button, dedicated Dataset Profile step, and jump-to-export shortcut
- Configurable **blocking mode** — OR (match if any active rule agrees, default) or AND (all active fields must agree simultaneously, for stricter precision)
- **Demographic cohort filtering** on the NC voter registry dataset (race, ethnicity, gender, party, birth year/state, age) — build a cohort by demographic criteria before linkage even runs, with exportable/re-importable cohort definitions
- Pre-linkage **Dataset Profile** step showing demographic composition before you configure fields or blocking rules
- Post-linkage demographic breakdown, **Linked vs. Unlinked comparison** (does linkage systematically miss certain demographic groups), and **real-time Match Quality by Demographic Group** — shows how match confidence for specific groups shifts live as you toggle blocking rules
- Interactive Blocking Explorer with a side-by-side **rule-cascade waterfall chart**: a static "original" view (all rules active) alongside a live "toggled" view showing how disabling a rule redistributes its edges to other rules instead of simply losing them
- Pre-flight candidate-pair estimation before running, with a tiered memory-risk warning
- `@st.cache_data`-backed caching on dataset loading and heavy aggregations; core coverage-matrix computation runs as a single DuckDB query for performance on large pairs tables
- SeRP-style downloadable PDF report with eight sections

---

## EDA pipeline (Upload mode)
When you upload a CSV the following cleaning steps run automatically:

1. Field name standardisation — lowercase, underscores, strip trailing numbers and special characters
2. Field type detection — infers semantic type (first_name, surname, dob, gender, location, postcode, email, id) from column names to drive comparison and blocking suggestions
3. Remove 100%-null columns — columns where every value is missing are dropped
4. Remove 100%-null rows — rows with no values at all are dropped
5. Remove n-1 null rows — rows with only one non-null value are dropped
6. Remove n-2 null rows — rows with only two non-null values are dropped
7. Text cleaning — strip whitespace, Title Case for name fields, lowercase for all other text
8. Duplicate removal — exact duplicate rows are dropped
9. Date standardisation — parses common date formats (DD/MM/YYYY, YYYYMMDD, etc.) and converts to YYYY-MM-DD
10. Correlation check — finds pairs of non-ID columns with >= 95% value-level agreement and asks which field to keep
11. EDA summary display — shows rows removed per step, fields changed, detected types, and a cleaned data preview
12. Download cleaned CSV — the cleaned dataset can be saved before proceeding

### Dataset B options (Upload mode)

- Upload a second CSV directly as Dataset B
- Create a 30% sample of Dataset A with controlled errors introduced (14% name typos, 5% missing DOBs, 15% email variations, 11% city abbreviations, 7% gender errors) for testing linkage
- Deduplication only (no Dataset B required)  

### Dataset B options (NC Voter Registry flow, Standard mode)

Unlike the Upload flow's percentage-rate error model above, the NC voter registry flow introduces errors **deterministically on every record** of each selected field:

- Text fields — a configurable number of letters (default 2) are randomly changed on every record
- Full-date fields (e.g. `registr_dt`) — shift by +1 month / +1 day, year unchanged
- Year-only fields (e.g. `birth_year`) — shift by a configurable +/- amount (default 1 year)
- Fields to corrupt are auto-populated from the blocking rules chosen in the field-selection step, and can be adjusted before generating Dataset B
- Sample fraction (how much of Dataset A becomes Dataset B) is independently configurable

---

## Post-linkage demographic insight and the Blocking Explorer

Every run (Standard, Upload, or Advanced) produces three complementary, deliberately separate demographic views, available for Run 1 and Run 2 alike:

1. **Population composition** — what the cohort looks like, by demographic field.
2. **Linked vs. Unlinked** — splits records by cluster size (linked = cluster size > 1, unlinked = singleton) and compares their demographic composition side by side. Answers whether linkage is systematically missing a particular group.
3. **Match Quality by Demographic Group** — an edge-level view (not population counts) showing mean match probability and % of edges above the cluster threshold, per demographic category. This is the view that's actually sensitive to blocking-rule toggles in real time.

The **Interactive Blocking Explorer** (Run 1's own explorer, a within-run toggle tab, and Run 2's explorer) lets you toggle individual blocking rules on/off and see all of the above update live, alongside a side-by-side **waterfall chart**:
- **Left ("Original")** — a static view assuming every configured rule is active, showing the run's true baseline.
- **Right ("Toggled, Live")** — reacts to your current toggle state: a disabled rule's edges either redistribute to a downstream rule that also covers them, or are shown as permanently lost — rather than the old, misleading "just subtract that rule's count."

Blocking rules combine as **OR** (any active rule matches — the default) or **AND** (all active fields must agree simultaneously), configurable per run.

---

## Project structure

```
cohort-builder-app/
├── app.py                    # Main Streamlit app (three flows, session-state navigation)
├── requirements.txt
├── dev-tools/                 # Not part of the running app — developer utilities
│   ├── repo.py                 # Consolidates the repo into one text file (for review/LLM context)
│   └── benchmark_caching.py    # Standalone performance/caching diagnostic script
├── modules/
│   ├── data_builder.py       # Builds fake1000 (gender + UK postcode) and loads NC voter registry data
│   ├── eda_engine.py         # Automated EDA/cleaning pipeline, shared by Upload flow and NC voter loading
│   ├── cohort_filter.py      # Tier-1 demographic cohort filtering framework
│   ├── splink_runner.py      # All Splink orchestration: linkage, JSON flow, coverage matrix, waterfall, re-clustering
│   ├── metrics_engine.py     # Linkage quality metrics + demographic breakdown computation
│   └── report_gen.py         # PDF report generator (incl. shared Venn diagram drawing)
├── flows/
│   ├── p_landing.py          # Standard flow: dataset selection, cohort filter, profile
│   ├── p_standard.py         # Standard flow: field config, operation mode, linkage type
│   ├── p_upload.py           # Upload flow
│   ├── p_advanced.py         # Advanced flow
│   ├── p_analysis.py         # SHARED: Run 1 results page (all flows)
│   └── p_compare_export.py   # SHARED: Run 2 comparison + Export page (all flows)
└── utils/
    ├── helpers.py             # Rendering functions + shared run orchestration
    ├── nav.py                 # Sidebar rendering and page navigation
    └── state.py                # Session-state initialisation and defaults
```

---

## Installation

```
pip install -r requirements.txt
streamlit run app.py
```

Optional dependencies for higher data quality in the generated dataset:

```
pip install gender-guesser pgeocode
```

Both fall back gracefully if not installed.

For the caching/performance diagnostic script (not required to run the app itself):

streamlit, splink, duckdb, pandas, numpy, plotly, fpdf2, matplotlib

---

## Testing the JSON upload feature
Generate a trained model JSON from any notebook or from the app itself (Save model JSON button on the analysis page), then upload it in Advanced mode. The JSON must be produced by linker.misc.save_model_to_json() or by the app's export function, which injects trained m/u probabilities into the comparison levels.

---

## Datasets

The built-in fake1000 dataset is derived from Splink's fake_1000, augmented with gender (inferred from first_name) and postcode (UK GeoNames lookup by city). Dataset B is a 50% sample of Dataset A with controlled errors: 14% first-name typos, 9% surname typos, 5% missing DOBs, 15% email variations, 11% city abbreviations, 7% gender errors.

---

## PDF report sections

1. Dataset information and completeness chart
2. Blocking rules and cumulative comparison count chart
3. Comparison methods
4. Model training with match weights chart and parameter estimates chart
5. Unlinkable records chart
6. Edge metrics and match weight histogram
7. Cluster metrics and dataset overlap Venn diagram
8. Confusion matrix with Precision-Recall curve and CRL score

---

## Known limitations and planned work

- Composite blocking rules currently limited to pairs of fields
- SAIL Databank provisioning on the export page is a placeholder for full deployment
- The EDA correlation check uses value-level co-occurrence for text fields, not statistical correlation; this is intentional and appropriate for record linkage use cases
- Upload mode currently accepts CSV only; DuckDB, Parquet, and Excel support is planned
- The Blocking Explorer's rule-cascade waterfall applies to OR-mode runs only; AND-mode runs show an explanatory note instead, since a single combined rule has nothing to cascade between
- No authentication or multi-tenancy — this is a research tool with per-session state, not a hosted multi-user product
- See `codebase-analysis-docs/CODEBASE_KNOWLEDGE.md` for a full technical breakdown, including known nuances and gotchas worth reading before changing code

---

## Related repositories

- linkage-workflow: JSON-driven Splink model configuration and notebook templates
- linkage-metrics: DuckDB SQL metric functions for intra- and inter-model linkage quality assessment
