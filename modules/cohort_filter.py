# =============================================================================
# modules/cohort_filter.py
# PURPOSE: Dynamic demographic cohort filtering for the NC voter registry
#          dataset (Tier 1 — core demographic fields only).
#
# Framework:
#   - CATEGORICAL fields (race_code, ethnic_code, party_cd, gender_code,
#     birth_state) use a multiselect of RAW CODES, labeled with human-
#     readable descriptions from layout_ncvoter.txt.
#   - RANGE fields (birth_year, age_at_year_end) use a min/max slider.
#   - All active filters combine with AND. An untouched (full-range or
#     empty) filter imposes NO restriction — it does not exclude records.
#   - Cohort definitions can be exported/imported as JSON for reuse across
#     sessions, and are also mirrored into the URL query string so they
#     survive a same-tab page refresh.
# =============================================================================

import base64
import json
from datetime import datetime, timezone

import pandas as pd

# ── Human-readable labels, sourced from layout_ncvoter.txt ──────────────────
RACE_CODE_LABELS = {
    "A": "Asian",
    "B": "Black or African American",
    "I": "American Indian or Alaska Native",
    "M": "Two or More Races",
    "O": "Other",
    "P": "Native Hawaiian or Pacific Islander",
    "U": "Undesignated",
    "W": "White",
}

ETHNIC_CODE_LABELS = {
    "HL": "Hispanic or Latino",
    "NL": "Not Hispanic or Not Latino",
    "UN": "Undesignated",
}

# NOTE: party codes are NOT defined in layout_ncvoter.txt. These are the
# standard NCSBE registered-party codes seen in the extracted data
# (DEM/REP/UNA/LIB) plus two other codes NCSBE sometimes uses. Please
# confirm/correct if your county file differs.
PARTY_CD_LABELS = {
    "DEM": "Democratic",
    "REP": "Republican",
    "UNA": "Unaffiliated",
    "LIB": "Libertarian",
    "GRE": "Green",
    "CST": "Constitution",
}

GENDER_CODE_LABELS = {
    "F": "Female",
    "M": "Male",
    "U": "Undesignated",
}

# Tier 1 field configuration — label_map is None for fields shown as raw codes
TIER1_CATEGORICAL_FIELDS = {
    "race_code":   RACE_CODE_LABELS,
    "ethnic_code": ETHNIC_CODE_LABELS,
    "party_cd":    PARTY_CD_LABELS,
    "gender_code": GENDER_CODE_LABELS,
    "birth_state": None,
}
TIER1_RANGE_FIELDS = ["birth_year", "age_at_year_end"]

COHORT_JSON_VERSION = 1


# =============================================================================
# FILTER STATE HELPERS
# =============================================================================

def default_filters(df: pd.DataFrame) -> dict:
    """Build an unrestricted default filter state from the actual data
    (empty selections = no restriction; ranges default to the full min/max
    actually present in df)."""
    filters = {"categorical": {}, "ranges": {}, "birth_state_include_missing": True}

    for field in TIER1_CATEGORICAL_FIELDS:
        filters["categorical"][field] = []

    for field in TIER1_RANGE_FIELDS:
        if field in df.columns:
            numeric = pd.to_numeric(df[field], errors="coerce").dropna()
            if not numeric.empty:
                filters["ranges"][field] = [int(numeric.min()), int(numeric.max())]
            else:
                filters["ranges"][field] = [0, 0]
        else:
            filters["ranges"][field] = [0, 0]

    return filters


def apply_cohort_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the Tier-1 demographic cohort filter to df. All active filters
    combine with AND. A categorical filter that is empty, or a range filter
    that spans the column's full actual min/max, is treated as untouched
    and imposes no restriction."""
    mask = pd.Series(True, index=df.index)

    for field, selected in filters.get("categorical", {}).items():
        if not selected or field not in df.columns:
            continue
        if field == "birth_state" and filters.get("birth_state_include_missing", True):
            mask &= df[field].isin(selected) | df[field].isna()
        else:
            mask &= df[field].isin(selected)

    for field, bounds in filters.get("ranges", {}).items():
        if field not in df.columns or not bounds:
            continue
        lo, hi = bounds
        numeric = pd.to_numeric(df[field], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            continue
        full_lo, full_hi = int(valid.min()), int(valid.max())
        if lo <= full_lo and hi >= full_hi:
            continue   # untouched / full range — no restriction from this field
        mask &= numeric.notna() & numeric.between(lo, hi)

    return df[mask].copy()


def cohort_summary(filters: dict) -> str:
    """Short human-readable description of the active filters."""
    parts = []
    for field, selected in filters.get("categorical", {}).items():
        if selected:
            parts.append(f"{field} in {selected}")
    for field, bounds in filters.get("ranges", {}).items():
        if bounds:
            parts.append(f"{field} between {bounds[0]} and {bounds[1]}")
    return "; ".join(parts) if parts else "No filters active (full dataset)"


# =============================================================================
# EXPORT / IMPORT — reusable "cohort definition" files
# =============================================================================

def filters_to_json(filters: dict, name: str = "") -> str:
    payload = {
        "version": COHORT_JSON_VERSION,
        "name":    name or "unnamed_cohort",
        "created": datetime.now(timezone.utc).isoformat(),
        "filters": filters,
    }
    return json.dumps(payload, indent=2)


def filters_from_json(json_text: str) -> dict:
    payload = json.loads(json_text)
    if "filters" not in payload:
        raise ValueError("Not a valid cohort definition file (missing 'filters' key).")
    return payload["filters"]


# =============================================================================
# URL QUERY-PARAM PERSISTENCE — survives a browser refresh in the same tab
# =============================================================================

def filters_to_query_param(filters: dict) -> str:
    raw = json.dumps(filters, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def filters_from_query_param(encoded: str) -> dict:
    raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
    return json.loads(raw.decode("utf-8"))