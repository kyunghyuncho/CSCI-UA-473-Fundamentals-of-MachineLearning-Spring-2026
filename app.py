"""
Machine Learning Course - Main Application Entry Point

This is the homepage of the interactive ML course application that serves as the
landing page and navigation hub for all course lessons.

What Students Learn Here:
--------------------------
- How the course is structured (lesson progression)
- Which datasets we're working with (TMDB Movies, NYC Airbnb)
- How to navigate the Streamlit multi-page interface
- What prerequisites and concepts each lesson covers

Technical Notes for Developers:
---------------------------------
Streamlit's multi-page app feature automatically:
1. Scans the pages/ directory for Python files matching pattern N_*.py
2. Creates a sidebar navigation menu in numerical order
3. Makes each file a separate page with isolated state
4. Shares session_state across pages for persistent data

This design pattern separates concerns:
- app.py: Homepage and course overview
- pages/: Individual lesson implementations
- utils/: Shared functionality (data, models, UI components)
"""

import streamlit as st

from utils.ui import display_footer

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================
# CRITICAL: set_page_config() MUST be the first Streamlit command on any page.
# This configures browser tab title, favicon, and overall layout settings.
# If called after any other Streamlit command, it will raise an error.
#
# Key Options Explained:
# ----------------------
# page_title : str
#     Text displayed in the browser tab/window title bar.
#     Helps users identify the app when multiple tabs are open.
#
# page_icon : str or PIL.Image
#     Icon displayed in the browser tab (favicon).
#     Can be an emoji (like "🎬") or path to an image file.
#     Emojis are simple and work cross-platform without additional files.
#
# layout : str, options=["centered", "wide"]
#     Controls the maximum width of the app content:
#     - "centered" (default): Limits content to ~800px, good for text-heavy apps
#     - "wide": Uses full browser width, excellent for dashboards and visualizations
#
#     We use "wide" because this course has many charts, tables, and side-by-side
#     comparisons that benefit from horizontal space.
#
# Why This Matters:
# -----------------
# Good page configuration improves user experience by:
# 1. Making the app identifiable in browser tabs
# 2. Using screen real estate efficiently
# 3. Setting professional visual standards
st.set_page_config(
    page_title="ML Course",
    page_icon="🎬",
    layout="wide",  # Use full width for better chart visibility
)

# ========================================================================
# MAIN CONTENT: HOMEPAGE / LANDING PAGE
# ========================================================================
# This section serves as the entry point and course overview for students.
#
# Design Philosophy:
# ------------------
# The homepage should:
# 1. Welcome and orient new users
# 2. Explain the course structure and philosophy
# 3. Describe available datasets
# 4. Provide clear navigation guidance
# 5. Set expectations for prerequisites and time commitment
#
# We use Streamlit's markdown support extensively to create rich, formatted
# text that's more engaging than plain text but easier to maintain than HTML.

# Main header - this is the first thing students see
# st.title() renders as an <h1> HTML element with Streamlit's styling
st.title("🎬 Introduction to Machine Learning")

# Main course introduction using markdown for formatting
# Markdown allows: headers (##), bold (**), lists (- or 1.), code (`code`), etc.
st.markdown(
    """
## Welcome to the Interactive ML Course!

This application is your hands-on companion for learning machine learning fundamentals.
Instead of just reading about algorithms, you'll **interact** with them, **visualize** how they work,
and **experiment** with real datasets.

### 🎯 Course Philosophy

**Learn by doing, not just reading.** Every lesson includes:
- ✅ Interactive visualizations to build intuition
- ✅ Configurable models to experiment with hyperparameters
- ✅ Real datasets (movies and Airbnb listings) for practical experience
- ✅ Detailed explanations of what's happening under the hood

### 📊 Datasets We Use

We'll explore ML using two real-world datasets:

**1. TMDB 5000 Movies** 🎥
- 5,000 movies with titles, plot summaries, genres, ratings, revenue
- Movie posters (images) for vision-based tasks
- Learn to: search movies by description, classify genres, predict ratings

**2. NYC Airbnb Listings** 🏠
- 5,000 Airbnb listings in New York City
- Descriptions, amenities, neighborhoods, prices, photos
- Learn to: find similar properties, predict prices, classify neighborhoods

**Switch between datasets** using the sidebar dropdown to see how the same ML techniques work on different domains!

### 📚 Course Modules

Use the **sidebar** (← over there!) to navigate to lessons. Here's what you'll learn:

---

#### [Lesson 1: From Text to Numbers (Embeddings)](/Lesson_1_Embeddings)
**Duration**: 30-45 minutes
**What you'll learn**:
- How computers convert text into numerical vectors (embeddings)
- What transformer models are and why they're powerful
- How to visualize 768-dimensional vectors
- Comparing text embeddings vs image embeddings

**Prerequisites**: Basic understanding of arrays/vectors

**Key Insight**: Embeddings capture semantic meaning - similar text gets similar numbers!

---

#### [Lesson 2: Vector Space & Retrieval](/Lesson_2_Retrieval)
**Duration**: 45-60 minutes
**What you'll learn**:
- Similarity metrics: cosine similarity, Euclidean distance, dot product
- Building a semantic search engine using embeddings
- Finding similar items by text description or image
- Nearest neighbor search algorithms

**Prerequisites**: Lesson 1 (Embeddings)

**Key Insight**: Search engines don't just match keywords - they find semantically similar content!

---

#### [Lesson 3: Data Splits (Train/Val/Test)](/Lesson_3_Data_Splits)
**Duration**: 30 minutes
**What you'll learn**:
- Why we split data into training, validation, and test sets
- How to prevent data leakage
- Analyzing distribution differences across splits
- Creating reproducible random splits

**Prerequisites**: None (foundational concept)

**Key Insight**: To know if your model works, you must test it on data it's never seen!

---

#### [Lesson 4: PCA (Principal Component Analysis)](/Lesson_4_PCA)
**Duration**: 60-75 minutes
**What you'll learn**:
- Dimensionality reduction: compressing 768 dimensions to 2 or 3
- Training deep autoencoders (non-linear PCA)
- Visualizing high-dimensional data in 2D scatter plots
- Projecting new data into the learned space
- Understanding reconstruction error and bottleneck representations

**Prerequisites**: Lesson 1 (Embeddings), Lesson 3 (Data Splits)

**Key Insight**: Most of the "information" in high-dimensional data lives in a much smaller  space!

---

#### [Lesson 5: MDS (Multidimensional Scaling)](/Lesson_5_MDS)
**Duration**: 60 minutes
**What you'll learn**:
- An alternative to PCA that preserves distances instead of variance
- Different distance functions: Euclidean, cosine, correlation
- When to use MDS vs PCA
- Applying MDS to text and image embeddings

**Prerequisites**: Lesson 4 (PCA)

**Key Insight**: PCA finds axes of maximum variance, MDS preserves pairwise distances - different goals!

---

#### [Lesson 6: Genre Classification (Multi-Label)](/Lesson_6_Genre_Classification)
**Duration**: 75-90 minutes
**What you'll learn**:
- Multi-label classification (one movie → multiple genres)
- Training neural networks with PyTorch Lightning
- Handling class imbalance with weighted loss
- Evaluation metrics: precision, recall, F1 score
- Testing predictions on new custom inputs

**Prerequisites**: Lesson 1 (Embeddings), Lesson 3 (Data Splits)

**Key Insight**: Movies can have multiple genres simultaneously - this is different from single-label classification!

---

---

#### [Lesson 7: Regression (Predicting Values)](/Lesson_7_Regression)
**Duration**: 90+ minutes
**What you'll learn**:
- Three types of regression: point estimation, probabilistic (MDN), quantile (SQR)
- When to use each approach
- Handling skewed distributions with log transforms
- Quantifying uncertainty in predictions
- Visualizing predicted probability distributions

**Prerequisites**: Lesson 1 (Embeddings), Lesson 3 (Data Splits)

**Key Insight**: Instead of just predicting a single number, you can predict entire distributions!

---

#### [Lesson 8: Cross-Modal Retrieval](/Lesson_8_Cross_Modal_Retrieval)
**Duration**: 60-75 minutes
**What you'll learn**:
- Mapping text and images to a shared vector space
- Training Dual Encoder models with Contrastive Learning
- Retrieving images from text descriptions (and vice-versa)
- Building a multi-modal search engine

**Prerequisites**: Lesson 1 (Embeddings), Lesson 2 (Retrieval)

**Key Insight**: By learning a shared space, we can translate between different modalities like text and images!

---

**Ready to start learning?** Select a lesson from the sidebar! 👈
"""
)

# NOTE: Lavender commented out the data part since the first few lectures will
# focus on fundamental instead of the implementation.

# ---

# ### 🚀 Getting Started

# 1. **Make sure you've processed the data** (see README.md):
#    ```bash
#    python process_data.py          # For TMDB Movies
#    python process_airbnb.py        # For Airbnb Listings (optional)
#    ```

# 2. **Select a lesson from the sidebar** (← look left!)

# 3. **Try both datasets** to see how the same techniques work on different data

# 4. **Experiment!** Change hyperparameters, try different configurations, see what happens

# ### 💡 Tips for Success

# - **Take your time**. Each lesson builds on previous ones.
# - **Experiment freely**. You can't break anything - just reload the page!
# - **Read the explanations**. We explain not just HOW but WHY things work.
# - **Try both datasets**. Seeing the same technique on different data builds intuition.
# - **Use the "Help" hover text** on controls to understand each parameter.

# ### 📖 Need More Background?

# If you encounter unfamiliar concepts:
# - Check the **README.md** for prerequisites and references
# - Each lesson has detailed explanations in markdown sections
# - Code comments explain technical details
# - Try the **scripts/verify_*.py** scripts to test your understanding

# ### ⚠️ Important Notes

# - **First time?** Start with Lesson 1 - each lesson builds on previous concepts.
# - **No data?** You'll see "Data not found" errors. Run the processing scripts first (see README).
# - **Slow training?** Reduce epochs/batch size or use a smaller hidden layer.
# - **Something broke?** Refresh the page. Streamlit state is ephemeral.

# ========================================================================
# SIDEBAR GUIDANCE
# ========================================================================
# Streamlit's Multi-Page App Feature:
# ------------------------------------
# When you run app.py, Streamlit automatically:
# 1. Scans the pages/ directory for files matching the pattern: N_*.py
#    (where N is a number for ordering)
# 2. Creates a navigation sidebar with page links in numerical order
# 3. Extracts page titles from the filenames (converts underscores to spaces)
# 4. Handles routing between pages automatically
#
# The sidebar is created WITHOUT us writing any code for it!
# We just add helpful contextual information below the auto-generated nav.
#
# Why This Design Pattern?
# -------------------------
# - Easy to add new lessons: just create pages/9_New_Lesson.py
# - Automatic ordering: rename files to change sequence
# - Isolated state: Each page has its own namespace but shares st.session_state
# - No routing code needed: Streamlit handles all navigation

# Success message to guide new users to the navigation menu
# The 👆 emoji visually points to where the lesson links appear
st.sidebar.success("👆 Select a lesson above to begin!")

# Add helpful quick links for common resources
# Using markdown("---") creates a horizontal divider for visual separation
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.markdown(
    "- [GitHub Repository](https://github.com/kyunghyuncho/CSCI-UA-473-Fundamentals-of-MachineLearning-Spring-2026#)"
)
st.sidebar.markdown(
    "- [Course Syllabus](https://docs.google.com/document/d/1yMcxM8_CX0ACe6Y2RytBK6yrty7UUoVgVCkEJZADTFM/edit?usp=sharing)"
)
st.sidebar.markdown(
    "- [Installation Guide](https://github.com/kyunghyuncho/CSCI-UA-473-Fundamentals-of-MachineLearning-Spring-2026/blob/main/README.md)"
)

# ========================================================================
# FOOTER
# ========================================================================
# Display a consistent footer across all pages using our shared UI component.
#
# Software Engineering Best Practice: DRY (Don't Repeat Yourself)
# ----------------------------------------------------------------
# Instead of copying footer HTML/markdown to every page, we:
# 1. Define the footer once in utils/ui.py
# 2. Import and call display_footer() on each page
#
# Benefits:
# - Single source of truth: Update footer in one place
# - Consistency: All pages guaranteed to have identical footer
# - Maintainability: Easy to add elements like version info or links
# - Testability: Can unit test footer rendering separately
#
# This pattern applies to any UI component used across multiple pages:
# headers, sidebars, data loading patterns, chart configurations, etc.
display_footer()
