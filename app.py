import streamlit as st

st.set_page_config(
    page_title="Cohort Builder",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after set_page_config) ──────────────────────────────────────────
from utils.state import _init_state
from utils.nav import _render_sidebar, _go_to

from flows.p_landing         import page_landing, page_profile
from flows.p_standard        import page_configure, page_operation, page_linkage_type
from flows.p_advanced        import page_advanced_setup, page_advanced_profile, page_advanced_configure
from flows.p_upload          import page_upload_setup, page_eda, page_upload_profile, page_upload_configure
from flows.p_analysis        import page_analysis
from flows.p_compare_export  import page_comparison, page_export


def main() -> None:
    _init_state()
    _render_sidebar()

    flow = st.session_state.get("flow", "standard")
    page = st.session_state["page"]

    # Pages shared across all flows
    shared = {4: page_analysis, 5: page_comparison, 6: page_export}

    if flow == "advanced":
        router = {
            "advanced_setup":     page_advanced_setup,
            "advanced_profile":   page_advanced_profile,
            "advanced_configure": page_advanced_configure,
            **shared,
        }
        router.get(page, page_advanced_setup)()

    elif flow == "upload":
        router = {
            "upload_setup":     page_upload_setup,
            "upload_eda":       page_eda,
            "upload_profile":   page_upload_profile,
            "upload_configure": page_upload_configure,
            2: page_operation,
            3: page_linkage_type,
            **shared,
        }
        router.get(page, page_upload_setup)()

    else:  # standard
        router = {
            0: page_landing,
            "profile": page_profile,
            1: page_configure,
            2: page_operation,
            3: page_linkage_type,
            **shared,
        }
        router.get(page, page_landing)()


if __name__ == "__main__":
    main()
