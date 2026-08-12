# modules/eda_engine.py
# Automated EDA, field validation, profiling, and type-safe noise injection pipeline.
# Purpose: Manages the programmatic text cleaning, missingness parsing, correlation sweeps,
#          and generates dynamic error injections without triggering float type errors on missing entries.

import re  # Import standard regular expressions for pattern matching and string cleaning
import string  # Import standard string library to access ASCII characters for typographical errors
import numpy as np  # Import numpy for fast array processing and random mask evaluations
import pandas as pd  # Import pandas for data frame manipulation and metadata tracking
from typing import Optional  # Import Optional for clean type-hinting support


# =============================================================================
# AUTOMATED DATA CLEANING AND STANDARDIZATION FUNCTIONS
# =============================================================================

def clean_field_names(df: pd.DataFrame) -> tuple:
    """Standardises dataframe column titles to lowercase alphanumeric formats."""
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    original_columns = list(df_copy.columns)  # Extract the list of original incoming column titles
    standardized_map = {}  # Initialize an empty dictionary to record the field renamings

    for col in original_columns:  # Iterate through each column name in the dataframe
        clean_name = str(col).strip().lower()  # Strip whitespace and cast the title to lowercase
        clean_name = re.sub(r'[^a-z0-9_]', '_', clean_name)  # Replace non-alphanumeric marks with underscores
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')  # Remove redundant consecutive underscores
        if clean_name != col:  # Check if the newly formatted title differs from the baseline
            standardized_map[col] = clean_name  # Log the column transformation mapping inside the dictionary

    df_copy = df_copy.rename(columns=standardized_map)  # Apply the renaming map to the dataframe columns
    return df_copy, standardized_map  # Return the updated dataframe along with its transformation log


def remove_null_elements(df: pd.DataFrame) -> tuple:
    """Drops completely unassigned columns and rows with excessive missing data.

    Removal criteria (three separate passes, matching the spec exactly):
      Pass 1 — 100% null columns: columns where every value is null.
      Pass 2 — 100% null rows:   rows where every value is null.
      Pass 3 — n-1 null rows:    rows with only 1 non-null value  (thresh=2).
      Pass 4 — n-2 null rows:    rows with only 2 non-null values (thresh=3).

    Note: thresh=k in pandas means 'keep rows with at least k non-null values'.
    Setting thresh=3 removes rows that have fewer than 3 populated fields,
    which is exactly n-2 removal (only rows with ≥3 values survive).
    """
    df_copy = df.copy()
    initial_rows = len(df_copy)

    # ── Pass 1: Remove 100%-null columns ─────────────────────────────────────
    null_cols = [c for c in df_copy.columns if df_copy[c].isna().all()]
    df_copy = df_copy.drop(columns=null_cols)

    # ── Pass 2: Remove 100%-null rows ─────────────────────────────────────────
    df_copy = df_copy.dropna(how='all')
    rows_after_all_null = len(df_copy)

    # ── Pass 3: Remove rows with n-1 nulls (only 1 field has a value) ─────────
    df_copy = df_copy.dropna(thresh=2)   # keep rows with ≥2 non-null values
    rows_after_n1 = len(df_copy)

    # ── Pass 4: Remove rows with n-2 nulls (only 2 fields have a value) ───────
    df_copy = df_copy.dropna(thresh=3)   # keep rows with ≥3 non-null values
    final_rows = len(df_copy)

    log_summary = {
        "null_columns_dropped": null_cols,
        "100%_null":            initial_rows - rows_after_all_null,
        "n-1_null":             rows_after_all_null - rows_after_n1,
        "n-2_null":             rows_after_n1 - final_rows,
        "partial_null_removed": rows_after_all_null - final_rows,   # kept for compat
    }
    return df_copy, log_summary


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Removes extra spaces and normalizes character layouts in text columns."""
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    for col in df_copy.columns:  # Loop through every column present in the working dataframe
        try:  # Trap unexpected data conversion issues gracefully inside a try-catch block
            # Convert values to strings, strip leading/trailing spaces, and compress internal spaces
            df_copy[col] = df_copy[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            # Replace literal text variants of missing data with standard numpy NaN objects
            df_copy[col] = df_copy[col].replace(['nan', 'NAN', 'None', '', 'null', 'NULL', '<NA>', 'NA', 'N/A', 'n/a', ' '], np.nan)
        except Exception:  # Catch data type anomalies across columns gracefully
            continue  # Bypass columns that cannot be processed as standard text strings
    return df_copy  # Return the sanitized text dataframe structure


def remove_duplicate_records(df: pd.DataFrame) -> tuple:
    """Identifies and removes exact duplicate rows from the dataset."""
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    initial_rows = len(df_copy)  # Track the row volume before running duplicate lookups
    df_copy = df_copy.drop_duplicates()  # De-duplicate identical rows across the entire dataset matrix
    removed_count = initial_rows - len(df_copy)  # Compute the exact count of duplicate records removed
    return df_copy, removed_count  # Return the unique dataframe rows along with the removal count


def standardise_date_formats(df: pd.DataFrame) -> tuple:
    """Parses date fields and normalizes them into uniform YYYY-MM-DD character strings.

    Column selection deliberately excludes bare-year fields (e.g. birth_year)
    even though their name contains "birth"/"dob"/"date" — pandas' date
    parser accepts a plain 4-digit string like "1975" and silently returns
    Timestamp("1975-01-01"), which would corrupt a year-only field into a
    fabricated full date. Worse, if parsing succeeds for only SOME rows in
    a column (mixed source formatting), two records with the identical real
    value can end up with different final strings and silently stop
    matching under exact-string blocking. A column only qualifies as a real
    date candidate if a sample of its values contains actual date
    separators (-, /, or a space, e.g. "1975-04-02" or "04/02/1975") —
    a bare year like "1975" does not.
    """
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    standardized_log = {}  # Initialize an empty dictionary to document successful date conversions

    def _looks_like_full_date(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).str.strip()
        sample = sample[sample != ""].head(50)
        if sample.empty:
            return False
        has_separator = sample.str.contains(r"[-/ ]", regex=True)
        return (has_separator.mean() > 0.5)  # Majority must look like a real date, not a bare year

    # Target fields with names containing birth, date, or dob patterns,
    # AND whose actual values look like full dates rather than bare years.
    date_candidates = [
        c for c in df_copy.columns
        if ('date' in c or 'dob' in c or 'birth' in c) and _looks_like_full_date(df_copy[c])
    ]
    for col in date_candidates:  # Loop through each potential date column candidate
        try:  # Wrap the conversion attempt in a try block to handle malformed strings
            # Parse the column rows into a standard timestamp object
            parsed_dates = pd.to_datetime(df_copy[col], errors='coerce')
            if parsed_dates.notna().sum() > 0:  # Verify that at least some dates were parsed successfully
                df_copy[col] = parsed_dates.dt.strftime('%Y-%m-%d')  # Reformat matching values to YYYY-MM-DD
                standardized_log[col] = "YYYY-MM-DD"  # Log the successful formatting rule
        except Exception:  # Bypass columns containing text that cannot be parsed as a date
            continue
    return df_copy, standardized_log  # Return the formatted date dataframe alongside its logging tag


# =============================================================================
# COHORT ANALYSIS AND FIELD TYPE PROFILE ENGINES
# =============================================================================

def run_full_eda(df: pd.DataFrame, id_col: Optional[str] = None) -> tuple:
    """Executes the complete automated data cleaning and profiling pipeline."""
    eda_manifest_log = {}  # Instantiate the primary repository metadata tracking dictionary
    original_rows = len(df)  # Record the initial row count before running cleaning rules

    # 1. Clean and standardize column titles
    df_working, shifted_names = clean_field_names(df)
    eda_manifest_log["field_names"] = {"changed": shifted_names}

    # 2. Remove rows and columns that are entirely null
    df_working, null_summary = remove_null_elements(df_working)
    eda_manifest_log["null_columns_dropped"] = null_summary["null_columns_dropped"]
    eda_manifest_log["null_rows_removed"] = {
        "100%_null": null_summary["100%_null"],
        "n-1_null":  null_summary["n-1_null"],
        "n-2_null":  null_summary["n-2_null"],
    }

    # 3. Normalize text fields and remove extra whitespaces
    df_working = clean_text_fields(df_working)

    # 4. Standardize date formats for target fields
    df_working, date_summary = standardise_date_formats(df_working)
    eda_manifest_log["dates_standardised"] = date_summary

    # 5. Remove exact duplicate records
    df_working, duplicate_count = remove_duplicate_records(df_working)
    eda_manifest_log["duplicates_removed"] = duplicate_count

    # 6. Build the final operational summary report
    eda_manifest_log["summary"] = {
        "original_rows": original_rows,
        "rows_removed": original_rows - len(df_working),
        "final_rows": len(df_working),
        "cols_removed": len(null_summary["null_columns_dropped"])
    }

    # 7. Infer field semantic data types based on column names and values
    inferred_types = {}
    mapped_id = shifted_names.get(id_col, id_col) if id_col else None

    for col in df_working.columns:  # Loop through every cleaned column to determine its type
        if col == "unique_id" or col == mapped_id:
            inferred_types[col] = "id"  # Lock unique tracking index keys to the id type parameter
        elif "first" in col or "given" in col:
            inferred_types[col] = "first_name"  # Assign first name tags
        elif "last" in col or "sur" in col or "family" in col:
            inferred_types[col] = "surname"  # Assign surname tags
        elif "name" in col:
            inferred_types[col] = "full_name"  # Assign general full name tags
        elif "date" in col or "dob" in col or "birth" in col:
            inferred_types[col] = "dob"  # Assign date tags to date fields
        elif "email" in col:
            inferred_types[col] = "email"  # Assign email tags to email fields
        elif "post" in col or "zip" in col:
            inferred_types[col] = "postcode"  # Assign geographic postcodes tags
        elif "gender" in col or "sex" in col:
            inferred_types[col] = "gender"  # Assign demographic gender tags
        elif "city" in col or "town" in col or "county" in col:
            inferred_types[col] = "location"  # Assign geographic location tags
        else:
            inferred_types[col] = "text"  # Default back to basic text categorization

    return df_working, inferred_types, eda_manifest_log["summary"], eda_manifest_log


def find_high_correlation_pairs(df: pd.DataFrame, id_cols: list) -> list:
    """Identifies highly correlated columns to flag redundant fields before matching."""
    correlated_pairs_list = []  # Initialize an empty list container to hold correlated column pairs
    feature_cols = [c for c in df.columns if c not in id_cols and c not in ("unique_id", "cluster", "source_dataset")]

    for i in range(len(feature_cols)):  # Run a nested double loop to compare columns side-by-side
        for j in range(i + 1, len(feature_cols)):
            col_a = feature_cols[i]  # Target column A
            col_b = feature_cols[j]  # Target column B
            try:  # Wrap inside a try block to handle non-numeric or empty categories safely
                # Calculate the percentage of rows where column A exactly matches column B
                exact_match_ratio = (df[col_a] == df[col_b]).mean()
                if exact_match_ratio > 0.85:  # Flag pairs that have an exact agreement rate above 85%
                    correlated_pairs_list.append((col_a, col_b, exact_match_ratio))  # Log the pair parameters
            except Exception:
                continue
    # Sort the list so the most highly correlated columns appear first
    return sorted(correlated_pairs_list, key=lambda x: x[2], reverse=True)


def suggest_comparison_types(field_types: dict) -> dict:
    """Recommends optimal Splink comparison library functions based on field types."""
    recommendation_map = {}  # Initialize an empty dictionary to hold the comparison recommendations
    for col, ftype in field_types.items():  # Loop through every column type entry
        if ftype == "id":
            continue  # Skip unique identifier columns
        elif ftype == "first_name":
            recommendation_map[col] = "JaroWinklerAtThresholds"  # Suggest Jaro-Winkler for names
        elif ftype == "surname":
            recommendation_map[col] = "JaroAtThresholds"  # Suggest Jaro for surnames
        elif ftype == "dob":
            recommendation_map[col] = "DateOfBirthComparison"  # Suggest date matching rules for DOB
        elif ftype == "email":
            recommendation_map[col] = "EmailComparison"  # Suggest email matching rules
        elif ftype == "postcode":
            recommendation_map[col] = "PostcodeComparison"  # Suggest postcode matching rules
        else:
            recommendation_map[col] = "ExactMatch"  # Fall back to exact matching for everything else
    return recommendation_map  # Return the mapping dictionary


def suggest_blocking_rules(field_types: dict) -> dict:
    """Flags high-cardinality fields as initial blocking rule candidates."""
    blocking_suggestions = {}  # Initialize an empty dictionary to hold the blocking recommendations
    for col, ftype in field_types.items():  # Loop through every column type entry
        # Recommend blocking on stable categorical fields like names, dates, and locations
        if ftype in ("first_name", "surname", "dob", "postcode", "location"):
            blocking_suggestions[col] = True  # Mark the field as an appropriate blocking rule candidate
        else:
            blocking_suggestions[col] = False  # Mark the field as disabled for default blocking
    return blocking_suggestions  # Return the blocking mapping configuration


# =============================================================================
# TYPE-SAFE CUSTOMIZABLE NOISE INJECTION MODULE (FIX FOR LEN() ERROR)
# Purpose: Handles customizable error rate sliders robustly across missing values.
# =============================================================================

def introduce_errors_for_sample(
        df: pd.DataFrame,
        field_types: dict,
        sample_frac: float = 0.5,
        seed: int = 42,
        error_rates: dict = None,
) -> pd.DataFrame:
    """Generates Dataset B by injecting customizable typographical and missingness errors into populated fields."""
    import random  # Import the standard random module for string character adjustments
    rng = np.random.default_rng(seed)  # Instantiate an isolated numpy random generator using a static seed
    random.seed(seed)  # Enforce a static seed for the standard random framework

    # Downsample Dataset A to isolate the branch evaluation cohort
    sample = df.sample(frac=sample_frac, random_state=seed).copy()
    if error_rates is None:  # Fall back to empty defaults if no error parameters are provided
        error_rates = {}  # Initialize as an empty dictionary to avoid iteration errors

    for col, ftype in field_types.items():  # Loop through each field type configuration
        if col not in sample.columns:
            continue  # Skip column entries that do not match the dataframe schema

        rate = error_rates.get(col, 0.0)  # Retrieve the custom error rate percentage set by the user
        if rate <= 0.0:
            continue  # Skip error injection for fields with an error rate of 0%

        # STABILITY SECURITY GUARD 1: Intersect random mask with .notna()
        # This prevents floating point NaN representations from entering string code blocks
        mask = (rng.random(len(sample)) < rate) & sample[col].notna()
        if not mask.any():
            continue  # If no rows qualify for mutation under this field, skip column processing

        # STABILITY SECURITY GUARD 2: Explicitly typecast and guard variables against non-string execution
        if ftype in ('first_name', 'surname', 'full_name'):
            def corrupt_string_value(val):
                s_val = str(val)  # Force string conversion to protect len() attributes
                if len(s_val) <= 1:
                    return s_val  # Return base value if text cannot sustain meaningful substitutions
                return "".join(
                    char if random.random() > 0.5 else random.choice(string.ascii_lowercase) for char in s_val)

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_string_value)

        elif ftype == 'dob':
            # Safe operation: Force assignment of uniform float missingness tags
            sample.loc[mask, col] = np.nan

        elif ftype == 'email':
            def corrupt_email_value(val):
                s_val = str(val)  # Protect string separation functions from float breakdowns
                if "@" in s_val:
                    parts = s_val.split('@', 1)
                    return parts[0] + str(random.randint(1, 9)) + "@" + parts[1]
                return s_val + str(random.randint(1, 9))

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_email_value)

        elif ftype in ('location', 'postcode'):
            def corrupt_location_value(val):
                s_val = str(val)  # Protect length scans and substring slicing tasks
                if len(s_val) > 3:
                    return s_val[:3].upper()  # Return sliced shorthand tag format
                return s_val.swapcase()  # Reverse typography casing parameters as alternative noise

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_location_value)

        elif ftype == 'gender':
            def corrupt_gender_value(val):
                s_val = str(val).upper()  # Uniformly clean standard character properties
                current = "M" if "M" in s_val else ("F" if "F" in s_val else "U")
                choices = [g for g in ("M", "F", "U") if g != current]
                return random.choice(choices)  # Random DIFFERENT valid gender code, never a raw letter

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_gender_value)

        else:
            def corrupt_generic_text(val):
                s_val = str(val)  # Standard generic string fallback task wrapper
                if len(s_val) > 1:
                    return s_val[:-1] + random.choice(string.ascii_lowercase)
                return s_val

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_generic_text)

    # Suffix ID columns with _B — guards against NaN, floats, and already-suffixed values
        # ── Assign ground-truth cluster BEFORE renaming unique_id ─────────────────
        # The cluster value must equal the Dataset A unique_id so the confusion
        # matrix can match A records to their derived B counterparts.
        # We use the pre-suffix unique_id value as the shared cluster identifier.
        if "unique_id" in sample.columns:
            # Capture the original A-side unique_id as the cluster ground truth
            sample["cluster"] = sample["unique_id"].apply(
                lambda x: str(x) if pd.notna(x) else "NA"
            )
        elif "cluster" not in sample.columns:
            # Fallback: use integer row position as cluster if no unique_id exists
            sample["cluster"] = range(1, len(sample) + 1)

        # ── Suffix ID columns with _B to separate them from Dataset A IDs ─────────
        id_cols = [c for c, t in field_types.items() if t == 'id' and c in sample.columns]
        for id_col in id_cols:
            def _safe_b_suffix(val):
                s = str(val) if pd.notna(val) else "NA"
                return s if s.endswith('_B') else s + '_B'

            sample[id_col] = sample[id_col].apply(_safe_b_suffix)

        sample['source_dataset'] = 'B'
        return sample


# =============================================================================
# DETERMINISTIC, SPEC-DRIVEN ERROR INTRODUCTION — NC VOTER REGISTRY FLOW
# Unlike introduce_errors_for_sample() above (probabilistic per-row RATES),
# this touches EVERY row of EVERY active column with a fixed transformation.
# =============================================================================

def _classify_field_kind(series: pd.Series) -> str:
    """Classify a column as 'full_date', 'year_only', or 'text' by sampling
    its non-null values.

    Bare-year detection runs BEFORE date parsing, because pd.to_datetime
    silently accepts a 4-digit year (e.g. '1999' -> 1999-01-01), which would
    otherwise misclassify a year-only column as a full date. A field must be
    the majority type across the sample; otherwise it falls back to 'text'.
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return "text"
    sample = non_null if len(non_null) <= 500 else non_null.sample(500, random_state=42)

    year_hits = sample.str.match(r'^(18|19|20)\d{2}$').sum()
    if year_hits / len(sample) > 0.5:
        return "year_only"

    parsed = pd.to_datetime(sample, errors="coerce")
    if parsed.notna().sum() / len(sample) > 0.5:
        return "full_date"

    return "text"


# Tier-1 categorical NC voter fields whose values are constrained to a fixed
# code set. These must never go through generic letter-corruption — a random
# letter swap on "M"/"F"/"U", "W"/"B"/"A"..., or "DEM"/"REP"/"UNA"/"LIB"
# produces values outside the known code set, which then show up as raw,
# uninterpretable garbage in the demographic pie chart legends.
NC_CATEGORICAL_CODE_VALUES = {
    "gender_code": ["M", "F", "U"],
    "race_code":   ["A", "B", "I", "M", "O", "P", "U", "W"],
    "ethnic_code": ["HL", "NL", "UN"],
    "party_cd":    ["DEM", "REP", "UNA", "LIB"],
}


def introduce_nc_voter_errors(
        df: pd.DataFrame,
        field_types: dict,
        sample_frac: float = 0.3,
        seed: int = 42,
        active_fields: Optional[list] = None,
        letters_to_change: int = 2,
        year_shift: int = 1,
) -> pd.DataFrame:
    """Generates the NC voter registry Dataset B using deterministic,
    spec-driven error introduction. Every row of every active column is
    modified (not a probabilistic subset):

      - Categorical code fields (gender_code, race_code, ethnic_code,
        party_cd): reassigned to a random DIFFERENT value from that field's
        known valid code set (see NC_CATEGORICAL_CODE_VALUES) — never a
        random letter, so demographic charts always show real categories.
      - Text fields:              `letters_to_change` letters changed per record (default 2).
      - Full-date fields (e.g. registr_dt): month +1, day +1, year unchanged.
      - Year-only fields (e.g. birth_year):  year shifted by +/- `year_shift` (default 1).

    active_fields: columns to corrupt. Defaults to every column except the
    Splink bookkeeping columns (unique_id, source_dataset, cluster).
    letters_to_change: how many characters to randomly change per record in
    text fields — the "how much error" control exposed in the UI.
    year_shift: magnitude of the random +/- shift applied to year-only fields.
    """
    import random
    import calendar
    from datetime import date
    random.seed(seed)

    def _corrupt_categorical_code(value, valid_values):
        if pd.isna(value):
            return value
        current = str(value).strip().upper()
        choices = [v for v in valid_values if v != current]
        if not choices:
            choices = valid_values
        return random.choice(choices)

    def _shift_full_date(value):
        if pd.isna(value):
            return value
        parsed = pd.to_datetime(str(value), errors="coerce")
        if pd.isna(parsed):
            return value  # unparseable — leave untouched

        year  = parsed.year
        month = parsed.month + 1
        day   = parsed.day + 1
        if month > 12:
            month = 1   # wrap within the same year — the year itself never changes
        last_day = calendar.monthrange(year, month)[1]
        if day > last_day:
            day = last_day
        return date(year, month, day).strftime("%Y-%m-%d")

    def _shift_year_only(value):
        if pd.isna(value):
            return value
        try:
            year = int(str(value).strip())
        except ValueError:
            return value
        return str(year + random.choice([-1, 1]) * year_shift)

    def _corrupt_text_exact_two(value):
        if pd.isna(value):
            return value
        s = str(value)
        letter_positions = [i for i, ch in enumerate(s) if ch.isalpha()]
        if not letter_positions:
            return s  # nothing alphabetic to corrupt (pure numeric/code fields)
        n_to_change = min(max(letters_to_change, 0), len(letter_positions))
        positions = random.sample(letter_positions, n_to_change)
        chars = list(s)
        for pos in positions:
            original = chars[pos]
            pool = string.ascii_uppercase if original.isupper() else string.ascii_lowercase
            choices_pool = [c for c in pool if c.lower() != original.lower()]
            chars[pos] = random.choice(choices_pool)
        return "".join(chars)

    sample = df.sample(frac=sample_frac, random_state=seed).copy()

    skip_cols = {"unique_id", "source_dataset", "cluster"}
    targets = (active_fields if active_fields is not None
               else [c for c in sample.columns if c not in skip_cols])

    for col in targets:
        if col not in sample.columns or col in skip_cols:
            continue

        if col in NC_CATEGORICAL_CODE_VALUES:
            valid_values = NC_CATEGORICAL_CODE_VALUES[col]
            sample[col] = sample[col].apply(
                lambda v, _vv=valid_values: _corrupt_categorical_code(v, _vv)
            )
            continue

        kind = _classify_field_kind(sample[col])
        if kind == "full_date":
            sample[col] = sample[col].apply(_shift_full_date)
        elif kind == "year_only":
            sample[col] = sample[col].apply(_shift_year_only)
        else:
            sample[col] = sample[col].apply(_corrupt_text_exact_two)

    # ── Ground-truth cluster BEFORE renaming unique_id (mirrors
    #    introduce_errors_for_sample so downstream confusion-matrix /
    #    metrics code keeps working unchanged) ─────────────────────────────
    if "unique_id" in sample.columns:
        sample["cluster"] = sample["unique_id"].apply(
            lambda x: str(x) if pd.notna(x) else "NA"
        )
    elif "cluster" not in sample.columns:
        sample["cluster"] = range(1, len(sample) + 1)

    # ── Suffix ID columns with _B to separate them from Dataset A IDs ─────────
    id_cols = [c for c, t in field_types.items() if t == 'id' and c in sample.columns]
    for id_col in id_cols:
        def _safe_b_suffix(val):
            s = str(val) if pd.notna(val) else "NA"
            return s if s.endswith('_B') else s + '_B'
        sample[id_col] = sample[id_col].apply(_safe_b_suffix)

    sample['source_dataset'] = 'B'
    return sample
