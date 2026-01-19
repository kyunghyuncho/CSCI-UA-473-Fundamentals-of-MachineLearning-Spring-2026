# flake8: noqa E501
"""
UI Component Utilities for Streamlit Pages

This module provides reusable UI components that appear across multiple pages
of the Streamlit application.

Components:
-----------
- display_footer(): Consistent footer with course attribution
- display_dataset_selector(): Sidebar widget to switch between datasets
"""

import os

import streamlit as st

from utils.dataset_config import DATASETS


def display_footer():
    """
    Display a consistent footer across all Streamlit pages.

    The footer includes:
    - Course name (Machine Learning)
    - Instructor name (Kyunghyun Cho)
    - Styled with HTML for centered, gray text

    Usage:
    ------
    In any Streamlit page, at the end:

    ```python
    from utils.ui import display_footer

    # ... page content ...

    display_footer()  # Add footer at bottom
    ```

    Educational Note:
    -----------------
    This demonstrates the DRY (Don't Repeat Yourself) principle:
    - Define the footer once in a centralized location
    - Reuse across all pages with a single function call
    - Update in one place to change footer everywhere

    Alternative approaches:
    - Streamlit components (more complex, more customizable)
    - Custom CSS file (more styling control)
    - Page template function (includes header + footer)
    """
    # Horizontal rule separator
    st.markdown("---")

    # Footer content with HTML styling
    # unsafe_allow_html=True allows custom HTML/CSS
    # (Use with caution - only for trusted content)
    st.markdown(
        """
        <div style="text-align: center; color: grey;">
            <p>Course material for <b>Machine Learning</b> created and taught by <b>Kyunghyun Cho</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_dataset_selector():
    """
    Displays a sidebar widget to select the active dataset.

    Updates st.session_state['current_dataset'] with the selected key.
    Returns the configuration dictionary for the selected dataset.
    """
    st.sidebar.header("Dataset Selection")

    # Get available datasets
    options = list(DATASETS.keys())
    format_func = lambda x: DATASETS[x]["name"]

    # Initialize session state if needed
    if "current_dataset" not in st.session_state:
        st.session_state["current_dataset"] = "tmdb"

    # Sidebar selection
    selected = st.sidebar.selectbox(
        "Choose Dataset",
        options,
        index=options.index(st.session_state["current_dataset"]),
        format_func=format_func,
        key="dataset_selector",
    )

    # Update state
    st.session_state["current_dataset"] = selected

    return DATASETS[selected]


def display_math_foundation(file_path):
    """
    Displays the mathematical foundation content from a markdown file.

    Args:
        file_path (str): Relative path to the markdown file from the project root.
                         e.g., "pages/math/lesson_1_embeddings.md"
    """

    # Check if file exists
    if os.path.exists(file_path):
        with st.expander("📚 Mathematical Foundations", expanded=False):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content, unsafe_allow_html=True)
    else:
        # Fail silently or log warning, but don't break the app if file missing
        # This allows us to add the call to the page before the file is ready if needed
        pass
