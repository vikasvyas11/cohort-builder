# utils/nav.py
# Page navigation helpers and sidebar renderer.

import streamlit as st
from utils.state import STANDARD_LABELS, ADVANCED_LABELS, UPLOAD_LABELS, clear_run_results


def _go_to(page) -> None:
    """Navigate to page, pushing current page onto history stack."""
    current = st.session_state["page"]
    history = st.session_state["page_history"]
    if not history or history[-1] != current:
        history.append(current)
    st.session_state["page_history"] = history
    st.session_state["page"] = page
    st.rerun()


def _go_back() -> None:
    """Navigate to previous page by popping history stack."""
    history = st.session_state["page_history"]
    if history:
        prev = history.pop()
        st.session_state["page_history"] = history
        st.session_state["page"] = prev
        if prev == 0:
            st.session_state["flow"] = "standard"
        st.rerun()


def _back_button(label: str = "Previous Step") -> None:
    """Render a back button; only shows when history is non-empty."""
    if st.session_state["page_history"]:
        if st.button(f"<- {label}", key=f"back_{st.session_state['page']}_{label}"):
            _go_back()


def _render_sidebar() -> None:
    """Sidebar: mode switcher radio + step navigation for the active flow."""
    flow = st.session_state.get("flow", "standard")
    page = st.session_state["page"]

    st.sidebar.title("Cohort Builder")

    # ── Mode switcher ──────────────────────────────────────────────────────────
    st.sidebar.caption("Active mode")
    mode_labels = ["Standard", "Upload Data", "Advanced (JSON)"]
    mode_map    = {"standard": 0, "upload": 1, "advanced": 2}
    selected    = st.sidebar.radio(
        "mode_switch",
        options=mode_labels,
        index=mode_map.get(flow, 0),
        label_visibility="collapsed",
    )

    if selected == "Standard" and flow != "standard":
        clear_run_results()
        st.session_state.update({"flow": "standard", "page": 0, "page_history": []})
        st.rerun()
    elif selected == "Upload Data" and flow != "upload":
        clear_run_results()
        st.session_state.update({"flow": "upload", "page": "upload_setup", "page_history": []})
        st.rerun()
    elif selected == "Advanced (JSON)" and flow != "advanced":
        st.session_state.update({"flow": "advanced", "page": "advanced_setup", "page_history": []})
        st.rerun()

    st.sidebar.divider()

    # ── Step navigation for the active flow ───────────────────────────────────
    if flow == "advanced":
        _nav_buttons(ADVANCED_LABELS, page, "nav_adv_")
    elif flow == "upload":
        _nav_buttons(UPLOAD_LABELS, page, "nav_up_")
    else:
        _nav_buttons(STANDARD_LABELS, page, "nav_std_")

    st.sidebar.divider()

    # ── Global shortcuts ───────────────────────────────────────────────────────
    if st.session_state["page_history"]:
        if st.sidebar.button("Go back", key="sb_back"):
            _go_back()

    if st.session_state.get("run1_results") and page not in (6, "advanced_setup", "upload_setup"):
        if st.sidebar.button("Jump to Export", key="sb_export"):
            _go_to(6)

    st.sidebar.divider()
    st.sidebar.caption("All data processed in memory. Nothing written to disk.")


def _nav_buttons(labels: dict, page, prefix: str) -> None:
    """Render sidebar nav buttons for a dict-keyed flow (advanced / upload)."""
    for key, label in labels.items():
        if key == page:
            st.sidebar.markdown(f"**-> {label}**")
        else:
            if st.sidebar.button(label, key=f"{prefix}{key}"):
                _go_to(key)
