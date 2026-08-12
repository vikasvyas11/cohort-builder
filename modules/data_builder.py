# =============================================================================
# modules/data_builder.py
# PURPOSE: Build the fake1000 dataset with gender + postcode columns,
#          then derive fakea (full, source A) and fakeb (50% sample with
#          controlled noise, source B).
# Mirrors logic from: linkage_workflow_demo_fake_data_new.ipynb
# =============================================================================

import random   # stdlib random for typo/email helpers
import string   # stdlib string constants for random character selection

import numpy as np
import pandas as pd
import streamlit as st

# ── Optional dependency: gender-guesser ───────────────────────────────────────
# Install with: pip install gender-guesser
try:
    import gender_guesser.detector as _gender_detector
    _GENDER_AVAILABLE = True
except ImportError:
    _GENDER_AVAILABLE = False   # Falls back to random assignment with realistic ratios

# ── Optional dependency: pgeocode ─────────────────────────────────────────────
# Install with: pip install pgeocode
try:
    import pgeocode as _pgeocode
    _PGEOCODE_AVAILABLE = True
except ImportError:
    _PGEOCODE_AVAILABLE = False  # Falls back to synthetic postcodes

# ── Splink dataset loader ─────────────────────────────────────────────────────
from splink.datasets import splink_datasets

# ── Cross-module imports (top-of-file: no circular import risk — neither
#    eda_engine.py nor cohort_filter.py imports anything from this module) ───
from modules.eda_engine import run_full_eda

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42           # Fixed seed for all reproducible operations
SAMPLE_FRAC = 0.50         # Fraction of fakea used to build fakeb
MAX_UNKNOWN_RATE = 0.10    # Maximum proportion of "Unknown" gender values allowed

# Maps gender-guesser output labels to app-internal values (M, F, Unknown)
_GENDER_MAP = {
    "male":          "M",
    "mostly_male":   "M",       # Strong lean → treat as M (per spec)
    "female":        "F",
    "mostly_female": "F",       # Strong lean → treat as F (per spec)
    "andy":          "Unknown", # Androgynous name – genuinely ambiguous
    "unknown":       "Unknown", # Name not in dictionary
}

# Known city abbreviations introduced as errors in fakeb
_CITY_ABBREVIATIONS = {
    "New York":    "NYC",
    "Los Angeles": "LA",
    "San Francisco": "SF",
    "Saint Louis": "St Louis",
    "Saint Paul":  "St Paul",
    "Washington":  "DC",
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _random_typo(text: str) -> str:
    """Replace one random character with a random lowercase letter.
    Simulates transcription errors in name fields."""
    if pd.isna(text) or len(str(text)) < 2:
        return text                             # Too short or NaN – nothing to corrupt
    text = str(text)
    pos = random.randint(0, len(text) - 1)     # Pick any position in the string
    replacement = random.choice(string.ascii_lowercase)  # Pick any a-z character
    return text[:pos] + replacement + text[pos + 1:]     # Swap the character at pos


def _email_variation(email: str) -> str:
    """Return a plausible alternative representation of an email address.
    Four variants: remove dots, append number, or swap to common domain."""
    if pd.isna(email):
        return email                            # Preserve NaN – no modification
    email = str(email)
    if "@" not in email:
        return email                            # Not a valid email – skip silently
    username, domain = email.split("@", 1)      # Split into local-part and domain

    # Four realistic variations that occur in real data entry
    variants = [
        username.replace(".", "") + "@" + domain,        # Remove dots: j.doe → jdoe
        username + str(random.randint(1, 99)) + "@" + domain,  # Append number
        username + "@gmail.com",                          # Common domain swap
        username + "@outlook.com",                        # Another common domain
    ]
    return random.choice(variants)              # Return one at random


def _abbreviate_city(city: str) -> str:
    """Return a shortened form of a city name.
    Known cities use _CITY_ABBREVIATIONS; others get a capital-initial acronym."""
    if pd.isna(city):
        return city                             # NaN city – return unchanged
    city = str(city)
    if city in _CITY_ABBREVIATIONS:
        return _CITY_ABBREVIATIONS[city]        # Use the known abbreviation
    words = city.split()
    if len(words) > 1:
        # E.g. "San Jose" → "SJ", "New Haven" → "NH"
        return "".join(w[0].upper() for w in words)
    return city                                 # Single-word city unchanged


def _assign_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'gender' column to df based on first_name inference.

    Rules applied in order:
      1. NaN first_name → always 'Unknown' (preserved, never reassigned)
      2. gender-guesser inference → maps to M / F / Unknown
      3. Unknown proportion capped at MAX_UNKNOWN_RATE; excess re-assigned
         randomly (excluding NaN-name rows which are protected)
      4. No NaN values permitted in the output column (assert enforced)
    """
    df = df.copy()
    rng = np.random.default_rng(RANDOM_SEED)    # Seeded RNG for reproducibility

    if _GENDER_AVAILABLE:
        # Use gender-guesser dictionary for name-based gender inference
        detector = _gender_detector.Detector()

        def _infer(name: str) -> str:
            """Return M / F / Unknown for a single first_name value."""
            if pd.isna(name):
                return "Unknown"                # NaN first_name → always Unknown
            result = detector.get_gender(str(name).strip().capitalize())
            return _GENDER_MAP.get(result, "Unknown")  # Default to Unknown if key missing

        df["gender"] = df["first_name"].apply(_infer)
    else:
        # Fallback when gender-guesser is not installed: random assignment
        # Realistic base rates: ~48% M, ~48% F, ~4% Unknown
        choices = rng.choice(
            ["M", "F", "Unknown"], size=len(df), p=[0.48, 0.48, 0.04]
        )
        df["gender"] = choices

    # Safety net: replace any NaN that slipped through with "Unknown"
    df["gender"] = df["gender"].fillna("Unknown")

    # ── Cap Unknown at MAX_UNKNOWN_RATE ────────────────────────────────────────
    unknown_idx = df.index[df["gender"] == "Unknown"].tolist()  # All Unknown rows
    n_total = len(df)
    n_unknown = len(unknown_idx)
    max_allowed = int(np.floor(MAX_UNKNOWN_RATE * n_total))     # Maximum permitted

    if n_unknown > max_allowed:
        excess = n_unknown - max_allowed         # How many to reassign
        # Only rows where first_name is NOT NaN are eligible for reassignment
        reassignable = [
            i for i in unknown_idx if not pd.isna(df.loc[i, "first_name"])
        ]
        excess = min(excess, len(reassignable))  # Can't reassign more than available
        to_reassign = rng.choice(reassignable, size=excess, replace=False)
        df.loc[to_reassign, "gender"] = rng.choice(["M", "F"], size=excess)

    # Verify no NaN values remain – raise immediately if something is wrong
    assert df["gender"].isna().sum() == 0, "NaN values found in gender column"
    return df


def _assign_postcode_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'postcode' column by looking up real UK postcodes for each city.

    Records with NaN city → NaN postcode (per spec: no postcode without city).
    If pgeocode is unavailable, synthetic postcodes are generated as a fallback.
    """
    df = df.copy()
    rng = np.random.default_rng(RANDOM_SEED)

    if _PGEOCODE_AVAILABLE:
        # Load the GeoNames UK postcode reference (downloads on first use)
        nomi = _pgeocode.Nominatim("gb")
        geo = nomi._data[["postal_code", "place_name"]].dropna().copy()
        # Normalise place names to lowercase for case-insensitive matching
        geo["place_name_clean"] = geo["place_name"].str.strip().str.lower()
        # Build city → [list of postcodes] lookup dictionary
        lookup = geo.groupby("place_name_clean")["postal_code"].apply(list).to_dict()

        def _fetch(city):
            """Return a random postcode for city, or NaN if not found."""
            if pd.isna(city):
                return np.nan               # No city → no postcode (per spec)
            key = str(city).strip().lower()
            candidates = lookup.get(key)
            if not candidates:
                return np.nan               # City not in GeoNames → NaN postcode
            return rng.choice(candidates)   # Pick one postcode at random

        df["postcode"] = df["city"].apply(_fetch)
    else:
        # Fallback: synthetic postcodes in a plausible UK format
        def _synthetic(city):
            if pd.isna(city):
                return np.nan               # No city → no postcode
            prefix = str(city)[:2].upper().replace(" ", "X")  # Use first 2 letters
            n = rng.integers(1, 99)
            return f"{prefix}{n} {n}AB"     # e.g. "SW12 12AB"

        df["postcode"] = df["city"].apply(_synthetic)

    return df


def _introduce_fakeb_errors(fakeb: pd.DataFrame) -> pd.DataFrame:
    """Apply all controlled error rates to fakeb as specified:
      - 14% first_name typos
      - 9%  surname typos
      - 5%  missing DOB
      - 15% email variations
      - 11% city abbreviations
      - 7%  gender flips (including Unknown → random M/F)
    Uses numpy random so the seed is respected by the caller.
    """
    fakeb = fakeb.copy()
    rng_errors = np.random.default_rng(RANDOM_SEED + 10)   # Separate RNG for error rates

    # ── 14% first_name typos ──────────────────────────────────────────────────
    mask = np.random.rand(len(fakeb)) < 0.14
    fakeb.loc[mask, "first_name"] = fakeb.loc[mask, "first_name"].apply(_random_typo)

    # ── 9% surname typos ─────────────────────────────────────────────────────
    mask = np.random.rand(len(fakeb)) < 0.09
    fakeb.loc[mask, "surname"] = fakeb.loc[mask, "surname"].apply(_random_typo)

    # ── 5% missing DOB ────────────────────────────────────────────────────────
    mask = np.random.rand(len(fakeb)) < 0.05
    fakeb.loc[mask, "dob"] = None               # Replace with None (NaN) to simulate missing

    # ── 15% email variations ──────────────────────────────────────────────────
    mask = np.random.rand(len(fakeb)) < 0.15
    fakeb.loc[mask, "email"] = fakeb.loc[mask, "email"].apply(_email_variation)

    # ── 11% city abbreviations ────────────────────────────────────────────────
    mask = np.random.rand(len(fakeb)) < 0.11
    fakeb.loc[mask, "city"] = fakeb.loc[mask, "city"].apply(_abbreviate_city)

    # ── 7% gender errors ─────────────────────────────────────────────────────
    gender_error_mask = rng_errors.random(len(fakeb)) < 0.07
    error_idx = fakeb.index[gender_error_mask]

    def _flip(val: str) -> str:
        """Flip M→F, F→M, Unknown→randomly assigned real gender."""
        if val == "M":  return "F"
        if val == "F":  return "M"
        return rng_errors.choice(["M", "F"])    # Unknown gets a concrete value as the error

    fakeb.loc[error_idx, "gender"] = fakeb.loc[error_idx, "gender"].apply(_flip)
    return fakeb


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────




@st.cache_data(show_spinner="Loading and cleaning NC voter registry data...")
def load_nc_voter_dataset(max_rows: int = 200_000) -> tuple:
    """Load voter_registry.csv from the repo root, run the same EDA pipeline
    as the Upload flow, and return (df_clean, field_types, eda_log).

    Empty-string cells (produced when dtype=str is used for reading CSVs) are
    converted to NaN BEFORE EDA so that fully-null column removal works
    correctly for large sparse government datasets like NC voter registry.

    Cached on max_rows: re-selecting the same dataset size (e.g. navigating
    back to the landing page) returns instantly instead of re-downloading
    and re-cleaning the CSV from scratch.
    """
    import io
    import os
    from pathlib import Path

    # ── Locate voter_registry.csv ─────────────────────────────────────────────
    # modules/data_builder.py lives one level below the repo root
    repo_root  = Path(__file__).resolve().parent.parent
    local_path = repo_root / "voter_registry.csv"

    raw_bytes = None
    if local_path.exists():
        with open(local_path, "rb") as fh:
            raw_bytes = fh.read()
    else:
        # Streamlit Cloud clones to /mount/src/<repo-name>/
        mount_candidates = list(Path("/mount/src").glob("*/voter_registry.csv"))
        if mount_candidates:
            with open(mount_candidates[0], "rb") as fh:
                raw_bytes = fh.read()

    if raw_bytes is None:
        raise RuntimeError(
            "voter_registry.csv not found. "
            f"Expected: {local_path}. "
            "Ensure the file is committed to the root of your repository "
            "(same folder as app.py) and redeployed on Streamlit Cloud."
        )

    # ── Parse — try comma, tab, semicolon, pipe ───────────────────────────────
    df_raw = None
    for sep in (",", "\t", ";", "|"):
        try:
            candidate = pd.read_csv(
                io.BytesIO(raw_bytes), sep=sep, nrows=max_rows,
                dtype=str, encoding="latin-1", on_bad_lines="skip",
            )
            if candidate.shape[1] > 1:
                df_raw = candidate
                break
        except Exception:
            continue

    if df_raw is None or df_raw.empty:
        raise RuntimeError(
            "voter_registry.csv was found but could not be parsed. "
            "Check it is a valid CSV, TSV, or pipe-delimited file."
        )

     # ── Convert all blank/missing representations → NaN before EDA ───────────
    # dtype=str means empty cells arrive as "" and whitespace-padded cells as
    # " ". Government datasets like NC voter registry use both heavily.
    # isna() only catches actual NaN/None/pd.NA, so we must normalise first.
    df_raw = df_raw.apply(
        lambda col: col.str.strip().replace(
            {"": np.nan, "nan": np.nan, "NaN": np.nan, "NULL": np.nan,
             "null": np.nan, "N/A": np.nan, "n/a": np.nan, "NA": np.nan}
        ) if col.dtype == object else col
    )

    # ── Run the same EDA pipeline as the Upload flow ──────────────────────────
    df_clean, field_types, _, eda_log = run_full_eda(df_raw.copy())

    # ── Assign Splink-required columns ────────────────────────────────────────
    df_clean["source_dataset"] = "A"

    # unique_id: use ncid or voter_reg_num if present after EDA name cleaning
    uid_assigned = False
    for candidate in ("ncid", "voter_reg_num", "unique_id"):
        if candidate in df_clean.columns:
            if candidate != "unique_id":
                df_clean["unique_id"] = df_clean[candidate].astype(str)
            uid_assigned = True
            break
    if not uid_assigned:
        df_clean.insert(0, "unique_id",
                        "NC_" + pd.Series(range(len(df_clean))).astype(str))

    # ── Ground-truth cluster col ──────────────────────────────────────────────
    for gt_col in ("ncid", "voter_reg_num", "unique_id"):
        if gt_col in df_clean.columns:
            df_clean["cluster"] = df_clean[gt_col].astype(str)
            break

    # NOTE: an earlier version of this function cast gender_code/race_code/
    # ethnic_code/party_cd to pandas `category` dtype and birth_year/
    # age_at_year_end to nullable `Int64`, as a memory optimisation. This was
    # reverted — it caused Splink/DuckDB blocking comparisons on birth_year
    # to silently match zero pairs (nullable-dtype pandas/DuckDB interop
    # issue), producing 0 edges on the NC voter flow with no error raised.
    # Every column stays as the plain string dtype the rest of the pipeline
    # (error introduction, comparisons, blocking) is known to work with.
    return df_clean, field_types, eda_log


def auto_select_nc_fields(field_types: dict, columns, max_fields: int = 12) -> tuple:
    """Auto-select comparison fields and default blocking toggles for the NC
    voter registry dataset, ranked by inferred semantic type (first_name,
    surname, dob, ... down to generic text/id, lowest score first).

    Shared by the landing page's "Use NC Voter Data" button and the
    dev-tools/benchmark_caching.py diagnostic script, so both always agree
    on which fields the real app would pick — previously this heuristic was
    copy-pasted in both places and could silently drift out of sync.
    """
    EXCLUDE = {"unique_id", "source_dataset", "cluster"}
    TYPE_PRIORITY = {
        "first_name": 0, "surname": 1, "dob": 2,
        "postcode": 3, "email": 4, "location": 5,
        "gender": 6, "text": 7, "id": 99,
    }
    scored = sorted(
        [(TYPE_PRIORITY.get(field_types.get(c, "text"), 7), c)
         for c in columns if c not in EXCLUDE],
        key=lambda x: x[0],
    )
    selected_fields = [c for _, c in scored][:max_fields]
    HIGH_SEL_TYPES = {"first_name", "surname", "dob", "postcode", "email"}
    blocking_toggles = {
        f: (field_types.get(f, "text") in HIGH_SEL_TYPES or f in ("ncid", "voter_reg_num"))
        for f in selected_fields
    }
    return selected_fields, blocking_toggles


@st.cache_data(show_spinner=False)
def build_datasets() -> tuple:
    """Build and return (fake1000_df, fakea, fakeb) as pandas DataFrames.

    fake1000_df : 1000 records with columns:
                  unique_id, first_name, surname, dob, city, email,
                  cluster, gender, postcode
    fakea       : same as fake1000_df with source_dataset = "A"
    fakeb       : 500-record sample of fakea with controlled errors and
                  source_dataset = "B", unique_ids suffixed with "_B"

    This is the entry point called from the Streamlit app.
    All operations are seeded for reproducibility.
    """
    # Seed Python stdlib random for the helper functions that use it
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Step 1: Load base Splink fake_1000 dataset ────────────────────────────
    fake1000_df = splink_datasets.fake_1000.copy()

    # ── Step 2: Add gender column based on first_name ────────────────────────
    fake1000_df = _assign_gender_column(fake1000_df)

    # ── Step 3: Add postcode column based on city (UK lookup) ────────────────
    fake1000_df = _assign_postcode_column(fake1000_df)

    # ── Step 4: Build fakea = full dataset labelled source "A" ───────────────
    fakea = fake1000_df.copy()
    fakea["source_dataset"] = "A"               # Every record originates from dataset A
    # Splink's own fake_1000 ships unique_id as an int; Dataset B's ID gets a
    # "_B" suffix and becomes a string. Casting A to string here keeps both
    # datasets on one consistent dtype from the start — a known Splink gripe
    # (mixed int/str unique_id across tables causes join/comparison bugs).
    fakea["unique_id"] = fakea["unique_id"].astype(str)

    # ── Step 5: Build fakeb = 50% sample with errors, labelled source "B" ────
    fakeb = fakea.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED).copy()
    fakeb["unique_id"] = fakeb["unique_id"].astype(str) + "_B"  # Distinct IDs for B
    fakeb["source_dataset"] = "B"               # Tag all records as dataset B

    # ── Step 6: Introduce controlled errors into fakeb ───────────────────────
    fakeb = _introduce_fakeb_errors(fakeb)

    return fake1000_df, fakea, fakeb


def get_library_status() -> dict:
    """Return a dict indicating which optional libraries are available.
    Used by the app to inform the user about data quality of generated columns."""
    return {
        "gender_guesser": _GENDER_AVAILABLE,
        "pgeocode": _PGEOCODE_AVAILABLE,
    }
