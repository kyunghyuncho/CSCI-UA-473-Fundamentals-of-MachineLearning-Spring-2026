import io
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Import checkers
from labs.lab3_embeddings_retrieval.level_checks import (
    check_step_1_loading,
    check_step_2_encoding,
    check_step_3_query_doc,
    check_step_4_similarity,
    check_step_5_knn,
)

try:
    pass

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import transformers: {e}")
    TRANSFORMERS_AVAILABLE = False
except RuntimeError as e:
    print(
        f"Warning: Runtime error importing transformers (likely torchvision issue): {e}"
    )
    TRANSFORMERS_AVAILABLE = False

# Hardcoded config to make this lab standalone
TMDB_CONFIG = {
    "path": "data/processed/tmdb_embedded.parquet",
    "text_embedding_col": "embedding",
}


def render_embeddings_lab():
    st.header("Lab 3: Build Your Own Embedding System")

    st.markdown(
        """
    ### 🎯 The Mission

    Imagine you're building the search bar for **Netflix** or **IMDb**.

    A user types: *"movies about time travel paradoxes"*

    A keyword search might fail if the movie description doesn't contain the exact word "paradox".
    But a **Semantic Search** system understands that *"Back to the Future"* and *"Interstellar"* are perfect matches, even if they don't use those exact words.

    **In this lab, you will build this search engine from scratch.**

    ### 🗺️ The Roadmap

    1.  **The Brain**: Load a Transformer model (`nomic-embed-text`) that understands English.
    2.  **The Translation**: Convert text into **Embeddings** (lists of 768 numbers).
    3.  **The Match**: Use **Cosine Similarity** to measure how close two ideas are.
    4.  **The Engine**: Build a **k-NN (k-Nearest Neighbors)** search to find the best movies in a database.

    Let's get coding! 🚀
    """
    )

    render_text_coding_lab()


def render_text_coding_lab():
    # Initialize session state for variables if not present
    if "lab3_vars" not in st.session_state:
        st.session_state["lab3_vars"] = {}

    # --- STEP 1 ---
    st.subheader("Step 1: The Engine")
    st.info("Goal: Load the `nomic-ai/nomic-embed-text-v1.5` model.")

    st.markdown(
        """
    First, we need to load the brain of our operation: the **Transformer** model.

    **What is a Transformer?**
    Think of a Transformer as a "Universal Translator".
    - But instead of translating English to Spanish, it translates **English to Numbers**.
    - It reads the entire sentence at once (using "Attention") to understand context.
    - Example: In "Apple fell from the tree" vs "Apple stock fell", it knows "Apple" means different things!

    We are using the `sentence_transformers` library, which makes using these powerful models easy.

    **Your Task:**
    1. Import `SentenceTransformer` from `sentence_transformers`
    2. Initialize the model with `model_name_or_path="nomic-ai/nomic-embed-text-v1.5"`
    3. Check out the documentation for the model [here](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
    """
    )

    _answer_code_1 = """# Import the library
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)"""  # noqa: F841

    default_code_1 = """TODO: Add your code here"""

    code_1 = st.text_area(
        "Step 1 Code:", value=default_code_1, height=150, key="code_1"
    )

    if st.button("Run Step 1"):
        # Reset vars to ensure clean state
        st.session_state["lab3_vars"] = {}

        with st.spinner("Downloading/Loading model... (this takes ~10s first time)"):
            try:
                # Capture stdout to show print statements
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()

                # Execute user code
                exec(code_1, {}, st.session_state["lab3_vars"])

                sys.stdout = old_stdout
                output = mystdout.getvalue()
                if output:
                    st.code(output)

                # Verify
                passed, msg = check_step_1_loading(st.session_state["lab3_vars"])
                if passed:
                    st.success(msg)
                    st.session_state["step_1_done"] = True
                else:
                    st.error(msg)
            except Exception as e:
                sys.stdout = old_stdout
                st.error(f"Runtime Error: {e}")

    # --- STEP 2 ---
    if st.session_state.get("step_1_done", False):
        st.divider()
        st.subheader("Step 2: From Text to Numbers")
        st.info("Goal: Convert text into 768-dimensional vectors.")

        st.markdown(
            """
        Now that we have the `model`, we can use its `.encode()` method.

        **Your Task:**
        1. Create a list of strings named `texts` (e.g., "Hello world", "Machine learning")
        2. Call `model.encode(texts)`. This will return a list of 768-dimensional vectors.
        3. Store the result in `embeddings`

        **Experiment:**
        What values do you see in the vectors? Are they unit vectors? Can you make them unit vectors?
        """
        )

        _answer_code_2 = """# 'model' is available from Step 1
texts = ["Action movie with explosions", "Romantic comedy with a happy ending"]

# Generate embeddings (try with and without normalization!)
embeddings = model.encode(texts, normalize_embeddings=True)"""  # noqa: F841

        default_code_2 = """TODO: Add your code here"""
        code_2 = st.text_area(
            "Step 2 Code:", value=default_code_2, height=150, key="code_2"
        )

        if st.button("Run Step 2"):
            if "model" not in st.session_state["lab3_vars"]:
                st.error("⚠️ Model not found! Please run Step 1 again.")
            else:
                try:
                    exec(code_2, {}, st.session_state["lab3_vars"])
                    passed, msg = check_step_2_encoding(st.session_state["lab3_vars"])
                    if passed:
                        st.success(msg)
                        st.session_state["step_2_done"] = True

                        # Show Visualization immediately
                        st.write("### 👁️ Visualization")
                        st.write("Here is what your text looks like to the machine:")

                        embeddings = st.session_state["lab3_vars"]["embeddings"]
                        texts = st.session_state["lab3_vars"]["texts"]

                        # Calculate norm to show if it's normalized
                        norms = np.linalg.norm(embeddings, axis=1)
                        is_normalized = np.allclose(norms, 1.0, atol=1e-3)

                        if is_normalized:
                            st.success(f"✅ Vectors are normalized! Norms: {norms}")
                        else:
                            st.warning(
                                f"⚠️ Vectors are NOT normalized. Norms: {norms}. Did you set `normalize_embeddings=True`?"
                            )

                        cols = st.columns(2)

                        for i in range(min(2, len(texts))):
                            with cols[i]:
                                emb = embeddings[i]
                                txt = texts[i]

                                # Reshape 768 -> 24x32 for grid view
                                grid = emb.reshape(24, 32)
                                fig = px.imshow(
                                    grid,
                                    title=f"'{txt}'",
                                    color_continuous_scale="RdBu",
                                    aspect="auto",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        st.info(
                            "👀 **Observe:** Do the patterns look random? Or do you see similar 'hot spots' (red/blue areas) in both if the texts are similar?"
                        )
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Runtime Error: {e}")

    # --- STEP 3: Query vs Document ---
    if st.session_state.get("step_2_done", False):
        st.divider()
        st.subheader("Step 3: The Asymmetric Trap (Query vs Document)")
        st.info("Goal: Understand why queries and documents are treated differently.")

        st.markdown("### 🧠 Think about it")
        user_answer = st.text_area(
            "Why would a search engine need to distinguish between a user's question (query) and a webpage (document)?",
            placeholder="Type your answer here...",
        )

        if user_answer:
            st.success("Thanks for your thought! Here is the reason:")
            st.markdown(
                """
            **Asymmetric Search**:
            - **Queries** are often short, imprecise questions ("scary movie space").
            - **Documents** are long, detailed descriptions ("Alien is a 1979 science fiction horror film...").

            If we embedded them exactly the same way, the vector for "scary movie" might not match the vector for the *description* of Alien.

            **The Nomic Trick**:
            This model was trained with special **prefixes** to handle this:
            - Start queries with: `search_query: `
            - Start docs with: `search_document: `

            This tells the model *how* to process the text.
            """
            )

            st.markdown("### 💻 Code It")
            st.markdown(
                """
            **Your Task:**
            1. Define a query string `query` (e.g. "What is the best movie about dreams?") prepended with `search_query: `
            2. Define a document string `doc` (e.g. description of Inception "Dom Cobb is a skilled thief, the absolute best in the dangerous art of extraction, stealing valuable secrets from deep within the subconscious during the dream state.") prepended with `search_document: `
            3. Encode them into `query_emb` and `doc_emb` using `model.encode(..., normalize_embeddings=True)`.
            """
            )

            _answer_code_3 = """# 'model' is available
query_text = "search_query: What is the best movie about dreams?"
doc_text = "search_document: Dom Cobb is a skilled thief, the absolute best in the dangerous art of extraction, stealing valuable secrets from deep within the subconscious during the dream state."

query_emb = model.encode(query_text, normalize_embeddings=True)
doc_emb = model.encode(doc_text, normalize_embeddings=True)"""  # noqa: F841

            default_code_3 = """TODO: Add your code here"""

            code_3 = st.text_area(
                "Step 3 Code:", value=default_code_3, height=200, key="code_3"
            )

            if st.button("Run Step 3"):
                if "model" not in st.session_state["lab3_vars"]:
                    st.error("⚠️ Model missing. Restart Step 1.")
                else:
                    try:
                        exec(code_3, {}, st.session_state["lab3_vars"])
                        passed, msg = check_step_3_query_doc(
                            st.session_state["lab3_vars"]
                        )

                        if passed:
                            st.success(msg)
                            st.session_state["step_3_done"] = True

                            # Visual Comparison
                            q_emb = st.session_state["lab3_vars"]["query_emb"]
                            d_emb = st.session_state["lab3_vars"]["doc_emb"]

                            st.write("### 🆚 Vector Comparison")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(
                                    px.imshow(
                                        q_emb.reshape(24, 32),
                                        title="Query Vector",
                                        color_continuous_scale="RdBu",
                                    ),
                                    use_container_width=True,
                                )
                            with col2:
                                st.plotly_chart(
                                    px.imshow(
                                        d_emb.reshape(24, 32),
                                        title="Document Vector",
                                        color_continuous_scale="RdBu",
                                    ),
                                    use_container_width=True,
                                )

                            st.info(
                                "Even though the text lengths are vastly different, they are mapped to the SAME 768-dimensional space so we can compare them!"
                            )
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Runtime Error: {e}")

    # --- STEP 4 ---
    if st.session_state.get("step_3_done", False):
        st.divider()
        st.subheader("Step 4: Calculating Similarity")
        st.info("Goal: Measure how similar the query is to the document.")

        st.markdown(
            r"""
        Now we have our `query_emb` and `doc_emb`. Do they match?

        The most common way to compare embeddings is **Cosine Similarity**, which measures the cosine of the angle between two vectors.

        $$ \text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$

        **Why Dot Product?**
        The Nomic model (like many others) outputs **normalized** vectors (length = 1).
        When $\|A\| = 1$ and $\|B\| = 1$, the formula simplifies to just the dot product!

        $$ \text{similarity} = A \cdot B $$

        **Your Task:**
        1. Calculate the dot product of `query_emb` and `doc_emb`. What shape do the vectors have? What shape does the result have? Hint: Use `@` for matrix multiplication.
        2. Store it in `similarity`.
        """
        )

        _answer_code_4 = """# 'query_emb' and 'doc_emb' are available

# Calculate dot product
similarity = query_emb @ doc_emb.T"""  # noqa: F841

        default_code_4 = """TODO: Add your code here"""

        code_4 = st.text_area(
            "Step 4 Code:", value=default_code_4, height=150, key="code_4"
        )

        if st.button("Run Step 4"):
            if "query_emb" not in st.session_state["lab3_vars"]:
                st.error("⚠️ Embeddings not found! Please run Step 3 again.")
            else:
                try:
                    # Make numpy and pandas available in exec environment
                    st.session_state["lab3_vars"]["np"] = np
                    st.session_state["lab3_vars"]["pd"] = pd

                    exec(code_4, {}, st.session_state["lab3_vars"])
                    passed, msg = check_step_4_similarity(st.session_state["lab3_vars"])
                    if passed:
                        st.success(msg)
                        sim_score = st.session_state["lab3_vars"]["similarity"]
                        st.metric("Query-Document Match Score", f"{sim_score:.4f}")

                        if (
                            sim_score > 0.4
                        ):  # Nomic scores can be lower than typical cosine sim
                            st.balloons()
                            st.write(
                                "🎉 **Match Found!** The model connects the concept of 'dreams' to the plot of Inception."
                            )
                        else:
                            st.write(
                                "The score is low. Maybe the query wasn't specific enough?"
                            )

                        st.session_state["step_4_done"] = True

                        # Visualization of Dot Product
                        st.write("### 📐 Visualization")

                        q_vec = st.session_state["lab3_vars"]["query_emb"]
                        d_vec = st.session_state["lab3_vars"]["doc_emb"]

                        # Create a bar chart comparing the first 50 dimensions
                        # to show how components align (or don't)
                        n_dims = 50
                        df_vis = pd.DataFrame(
                            {
                                "Dimension": list(range(n_dims)) * 2,
                                "Value": np.concatenate(
                                    [q_vec[:n_dims], d_vec[:n_dims]]
                                ),
                                "Type": ["Query"] * n_dims + ["Document"] * n_dims,
                            }
                        )

                        fig = px.bar(
                            df_vis,
                            x="Dimension",
                            y="Value",
                            color="Type",
                            barmode="group",
                            title=f"Comparing First {n_dims} Dimensions",
                            color_discrete_map={
                                "Query": "#EF553B",
                                "Document": "#636EFA",
                            },
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.info(
                            f"**Score: {sim_score:.4f}**\n\nThe dot product sums up the product of matching dimensions. Where both bars are positive (or both negative), they contribute positively to the score."
                        )

                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Runtime Error: {e}")

    # --- STEP 5: KNN Search ---
    if st.session_state.get("step_4_done", False):
        st.divider()
        st.subheader("Step 5: The Search Engine (k-NN)")
        st.info("Goal: Find the best match among many documents.")

        st.markdown(
            """
        Real search engines don't just compare one pair. They compare the query against **millions** of documents.

        Let's build a mini search engine with a small corpus (collection of documents).

        **Algorithm:**
        1. **Embed** the entire corpus (batch processing).
        2. **Embed** the query.
        3. **Compute Scores**: Calculate dot product between query and ALL corpus vectors.
        4. **Sort**: Find the indices of the highest scores (Top-K).

        **Your Task:**
        1. I've provided a list of movies `corpus`.
        2. Encode them into `corpus_embeddings` (remember the prefix `search_document:`!).
        3. Calculate `scores = corpus_embeddings @ query_emb.T`.
        4. Use `np.argsort(scores)[::-1]` to sort descending.
        5. Get the top 3 indices.
        """
        )

        default_code_5 = """# 'model' and 'query_emb' are available
corpus = [
    "The Matrix: A computer hacker learns from mysterious rebels about the true nature of his reality.",
    "Titanic: A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
    "Avengers: Earth's mightiest heroes must come together and learn to fight as a team.",
    "Interstellar: A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
    "Shrek: A mean lord exiles fairytale creatures to the swamp of a grumpy ogre."]
"""

        _answer_code_5 = """# 'model' and 'query_emb' are available
corpus = [
    "The Matrix: A computer hacker learns from mysterious rebels about the true nature of his reality.",
    "Titanic: A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
    "Avengers: Earth's mightiest heroes must come together and learn to fight as a team.",
    "Interstellar: A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
    "Shrek: A mean lord exiles fairytale creatures to the swamp of a grumpy ogre."
]

# 1. Prepend prefix
corpus_prefixed = ["search_document: " + doc for doc in corpus]

# 2. Embed Corpus
corpus_embeddings = model.encode(corpus_prefixed, normalize_embeddings=True)

# 3. Compute Scores (Matrix Multiplication)
# (N, 768) @ (768,) -> (N,)
scores = corpus_embeddings @ query_emb.T

# 4. Find Top K
# argsort sorts ascending, so we use [::-1] to reverse it
top_k_indices = np.argsort(scores)[::-1][:3]

print(f"Top matches for '{query_text}':")
for idx in top_k_indices:
    print(f"Score: {scores[idx]:.4f} - {corpus[idx]}")
"""  # noqa: F841
        code_5 = st.text_area(
            "Step 5 Code:", value=default_code_5, height=350, key="code_5"
        )

        if st.button("Run Step 5"):
            if "query_emb" not in st.session_state["lab3_vars"]:
                st.error("⚠️ Embeddings not found! Please run Step 3 again.")
            else:
                try:
                    # Make numpy available
                    st.session_state["lab3_vars"]["np"] = np

                    # Capture stdout
                    old_stdout = sys.stdout
                    sys.stdout = mystdout = io.StringIO()

                    exec(code_5, {}, st.session_state["lab3_vars"])

                    sys.stdout = old_stdout
                    output = mystdout.getvalue()
                    if output:
                        st.code(output)

                    passed, msg = check_step_5_knn(st.session_state["lab3_vars"])
                    if passed:
                        st.balloons()
                        st.success("🏆 **You built a Search Engine!**")
                        st.session_state["step_5_done"] = True

                        st.markdown(
                            """
                        **Summary:**
                        You just implemented the core loop of Google, Spotify, and Amazon search:
                        1. Embed Query
                        2. Dot Product with Database
                        3. Sort & Retrieve
                        """
                        )
                    else:
                        st.error(msg)
                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"Runtime Error: {e}")

    # --- FINAL: Real Dataset ---
    if st.session_state.get("step_5_done", False):
        st.divider()
        st.header("🚀 Put it all together: Real Movie Search")
        st.markdown(
            """
        You've built the engine. Now let's give it some real fuel.

        We have pre-computed embeddings for **5,000 movies** (The TMDB dataset).
        Let's use YOUR logic to search through them.
        """
        )

        # Load the real dataset (using cache)
        try:
            # We use pandas read_parquet for speed
            df = pd.read_parquet(TMDB_CONFIG["path"])

            # Extract the pre-computed embeddings
            # They are stored as a column of lists/arrays
            # We need to stack them into a big matrix (N, 768)
            matrix = np.stack(df[TMDB_CONFIG["text_embedding_col"]].values)

            st.success(
                f"Loaded {len(df)} movies with {matrix.shape[1]}-dimensional embeddings."
            )

            st.markdown("#### Search the 5,000 Movies")
            user_query = st.text_input("Enter a search query:", "Time travel paradox")

            if user_query:
                # 1. Embed Query (using the model you loaded!)
                model = st.session_state["lab3_vars"]["model"]
                query_vec = model.encode(
                    "search_query: " + user_query, normalize_embeddings=True
                )

                # 2. Compute Scores
                scores = matrix @ query_vec

                # 3. Top K
                k = 5
                top_k_indices = np.argsort(scores)[::-1][:k]

                # Display Results
                st.write(f"Top {k} matches:")

                cols = st.columns(k)
                for i, idx in enumerate(top_k_indices):
                    row = df.iloc[idx]
                    score = scores[idx]

                    with cols[i]:
                        # Show poster if available
                        if row["local_poster_path"]:
                            st.image(row["local_poster_path"], use_column_width=True)
                        st.caption(f"**{row['title']}**")
                        st.write(f"Match: {score:.2f}")
                        with st.expander("Plot"):
                            st.write(row["overview"])

        except Exception as e:
            st.error(
                f"Could not load TMDB dataset. Make sure you've run `process_data.py`. Error: {e}"
            )


if __name__ == "__main__":
    render_embeddings_lab()
