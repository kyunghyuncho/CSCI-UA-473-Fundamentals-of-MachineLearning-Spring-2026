"""
Lab 0: It's Time to Try Vibe Coding!

This lab sets up uv environment and introduces vibe coding with antigravity.
Students will run the streamlit app, try vibe code with antigravity, and try to debug the code.
"""

import os
from datetime import datetime

import streamlit as st
import torch

from labs.lab0_trying_vibe_coding.problem import find_max_price
from utils.security import safe_eval
from utils.ui import display_footer

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================
st.set_page_config(
    page_title="Lab 0: It's Time to Try Vibe Coding",
    page_icon="🧙‍♀️",
    layout="wide",
)

# ========================================================================
# CONSTANTS & PATHS
# ========================================================================
LAB_DIR = os.path.join("labs", "lab0_trying_vibe_coding")

# ========================================================================
# PAGE HEADER
# ========================================================================
st.title("🧙‍♀️ Lab 0: It's Time to Try Vibe Coding")

st.success("🎉 Congratulations! You have successfully set up your environment!")
with st.expander("What did I just do? (Click to learn)"):
    st.markdown(
        """
        1. **Git Cloned**: You downloaded the code history. **Git** is like "Google Docs for code"—it tracks changes and lets us collaborate.
        2. **Antigravity**: You are using an **AI-native IDE** designed for "Vibe Coding" (coding using natural language, which can be both powerful in the right hand and error-prone if not used carefully).
        3. **uv**: You built a clean **Environment**. Think of it like a "Project Kitchen"—we want to keep our ingredients (libraries) separate from other projects so flavors don't mix. `uv` is a super-fast tool to build this kitchen, because it can install dependencies in parallel.
        4. **Streamlit**: You launched a **Web App**. Streamlit turns Python scripts into interactive websites instantly, without needing HTML/CSS.
        """
    )


# Add "Defying Gravity" Background Music
# Path to the specific theme music
music_file = "data/media/music/ua473-theme.mp3"

if os.path.exists(music_file):
    st.audio(music_file, format="audio/mp3", autoplay=True, loop=True)
    st.caption(
        "🎵 Playing: UA473 Theme. Created by LJ using Suno. https://suno.com/s/lJQNoFKgLTh4LzL6"
    )
else:
    # Fallback if file is missing (graceful degradation)
    st.warning(f"Background music file not found at {music_file}")


st.markdown("### Be a cracked engineer!")


st.image(
    "data/media/images/engineers.png",
    caption="10x Engineer -> Vibe Coder -> Cracked Engineer. Source: The Information.",
)
reflection = st.radio(
    "**Q1.** What is the difference between these three?",
    [
        "1️⃣ They are the same picture.",
        "2️⃣ 10x Engineer codes everything, Vibe Coder prompts AI, Cracked Engineer prompts humans.",
        "3️⃣ 10x Engineer is as productive as 10 normal engineers, Vibe Coder trusts AI blindly, Cracked Engineer uses AI but verifies carefully.",
    ],
    key="engineer_diff",
    index=None,
)

if (
    reflection
    == "3️⃣ 10x Engineer is as productive as 10 normal engineers, Vibe Coder trusts AI blindly, Cracked Engineer uses AI but verifies carefully."
):
    st.success("Spot on! Verification is key.")
elif reflection:
    st.error("Not quite. Think about who is in control.")

with st.expander("🎯 Goal"):
    st.markdown(
        """
        We should all aspire to be **"cracked engineers"** who can not only **"vibe code"** with AI
        but also **verify** the generations with strong fundamentals.

        Using AI is a superpower, but blindly trusting it is a weakness.
        In this lab, you will practice using AI to understand code, but you must also use your logic to find a critical bug that AI might miss (or even introduce!).
        """
    )

st.divider()

# ========================================================================
# MAIN CONTENT: THE UNSTABLE CART
# ========================================================================

st.markdown("### 🛒 The Unstable Cart")
st.markdown(
    """
    You are building a shopping cart feature.
    We need to find the **most expensive item** to verify your credit card limit.
    Here's what the AI generated for you:
    """
)

st.code(
    """
def find_max_price(prices):
    prices.sort()
    return prices[-1]
    """,
    language="python",
)

st.info(
    """
    **Exercise 1**:
    1. Open `labs/lab0_trying_vibe_coding/problem.py`.
    2. Highlight the code.
    3. Press `Cmd + Shift + L` to ask Antigravity Agent to:
       > Carefully comment every line.
    (You can also use Copilot Agent with `Cmd + Shift + I` if you prefer.)
    """
)

with st.expander("What are those Red and Green lines? (Click to learn)"):
    st.markdown(
        """
        When Antigravity/Copilot Agent proposes changes, it shows you a **Diff**:
        - :red[**Red lines**]: Code being **removed**.
        - :green[**Green lines**]: New code being **added**.

        **You are in control:**
        - **Accept (`Cmd + Enter`)**: Apply the change.
        - **Reject (`Cmd + Backspace`)**: Discard it.
        - **Accept All / Reject All**: Use the buttons in the chat interface to handle multiple files at once.
        """
    )

response = st.text_area("**Q2.** What do you think is wrong with this code?")

if not response:
    st.info("Please enter your thoughts above to proceed.")
    st.stop()

st.markdown(
    """
    Users are reporting that checking the price **scrambles their cart**!
    Hit the "Find Max Price 💰" button to see what happens.
    """
)

st.info(
    """
    **Exercise 2:** Fix the bug using Vibe Coding.

    1. Open `labs/lab0_trying_vibe_coding/problem.py`.
    2. Highlight the `find_max_price` function.
    3. Press `Cmd + Shift + L` to ask Antigravity Agent:
       > Why does this modify the input list? Fix it.
    (You can also use Copilot Agent with `Cmd + Shift + I` if you prefer.)
    4. Apply the fix by saving the file.
    5. Reset the cart below (🔄) and try again (💰)!
    """
)

if "cart" not in st.session_state:
    st.session_state["cart"] = [10, 5, 20, 3, 8]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Your Cart 🛒")
    st.write("Current Items (Prices):")
    st.code(str(st.session_state["cart"]), language="python")

    if st.button("Reset Cart 🔄"):
        st.session_state["cart"] = [10, 5, 20, 3, 8]
        st.rerun()

with col2:
    st.subheader("Checkout Action")
    st.write("Find the most expensive item to authorize payment.")

    if st.button("Find Max Price 💰", use_container_width=True):
        cart_ref = st.session_state["cart"]
        max_val = find_max_price(cart_ref)

        st.success(f"Max Price Found: ${max_val}")

        st.write("Cart Validation Check...")
        st.code(str(st.session_state["cart"]), language="python")

        if st.session_state["cart"] == [10, 5, 20, 3, 8]:
            st.balloons()
            username = os.getenv("USER") or os.getenv("USERNAME") or "Unknown User"
            st.success(
                f"🎉 Great job! You fixed the bug! The cart order is preserved. Verified by: {username}\n\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        elif st.session_state["cart"] == [3, 5, 8, 10, 20]:
            st.error("⚠️ BUG DETECTED: The cart order was changed!")
            st.markdown(
                "**Diagnosis**: The function `find_max_price` caused a **Side Effect**."
            )


# ========================================================================
# CONCLUSION
# ========================================================================
st.divider()
st.subheader("🚀 Conclusion")

st.markdown(
    """
    ### **Exercise 3:** Submission
    1. Take a **screenshot** of your success message (balloons) above (ensure your **username** is visible).
    2. Reply to the **CampusWire** thread with your screenshot.
    """
)

st.link_button("Go to CampusWire Thread", "https://campuswire.com/c/GFC1A6E10/feed/7")

# Confirmation
if st.checkbox("I have replied to the thread with my screenshot"):
    st.balloons()
    st.success("🎉 Congratulations! You have officially completed Lab 0.")

    st.markdown("### 📚 Resources for the Future Cracked Engineer")
    st.info(
        """
        **Antigravity & Effective Vibe Coding**

        - **Antigravity**: Your new favorite pair programmer. Use it to generate, debug, and explain code.
        - **Vibe Coding**: The art of coding with AI. It's fast, fun, and powerful.

        **Tips for Effective Vibe Coding:**
        1. **Verify Everything**: AI is smart, but you are the pilot. Always check the work.
        2. **Iterate**: Don't expect perfection on the first try. Use follow-up prompts to refine the result.
        3. **Read the Diffs**: Understanding *what* changed is just as important as the result.

        **Recap: The Stack**
        - **Git**: Keeps track of your code history.
        - **Environment**: Your project's isolated workspace.
        - **uv**: Fast tool to build and manage environments.
        - **Streamlit**: Turns Python scripts into web apps.
        """
    )

    # Bonus Section
    st.divider()
    if "bonus_unlocked" not in st.session_state:
        st.session_state["bonus_unlocked"] = False

    if not st.session_state["bonus_unlocked"]:
        if st.button("👑 I finished the lab early and I'm bored"):
            st.session_state["bonus_unlocked"] = True
            st.rerun()

    if st.session_state["bonus_unlocked"]:
        st.markdown("### 🧩 Bonus: Tensor Puzzles")
        st.markdown(
            "Prove your vibe is matched by your math. Puzzles from "
            "[srush/Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles)."
        )

        # -----------------------
        # Puzzle 1: Outer product
        # -----------------------
        st.info(
            """
            **Puzzle 1**: Implement **Outer Product**.

            Given two Tensors `a` of shape `(i)` and `b` of shape `(j)`,
            compute their outer product of shape `(i, j)` using **broadcasting**.

            *Constraint*: No loops. No `torch.outer`. One line of code.
            """
        )

        # Test tensors
        a = torch.randn(4)
        b = torch.randn(5)
        expected1 = a[:, None] * b[None, :]

        answer1 = st.text_input(
            "Your Code (assume `a` and `b` exist):",
            placeholder="YOUR CODE HERE",
            key="puzzle_1",
        )

        if answer1:
            try:
                result = safe_eval(
                    answer1,
                    {"torch": torch},
                    {"a": a, "b": b},
                )

                if torch.allclose(result, expected1):
                    st.success("✅ Correct!")
                else:
                    st.warning(
                        "Output shape or values are incorrect. "
                        "Hint: think `(i, 1)` times `(1, j)`."
                    )
            except Exception as e:
                st.error(f"❌ Error running your code:\n\n{e}")

        st.divider()

        # -----------------------
        # Puzzle 2: Identity
        # -----------------------
        st.info(
            """
            **Puzzle 2**: Implement **Identity Matrix**.

            Create a square identity matrix of size `j × j`
            using broadcasting.

            You may use `torch.arange(j)` and Boolean outputs are ok.

            *Constraint*: No loops. No `torch.eye`. One line of code.
            """
        )

        j = 6
        expected2 = torch.eye(j)

        answer2 = st.text_input(
            "Your Code (assume `j` exists):",
            placeholder="YOUR CODE HERE",
            key="puzzle_2",
        )

        if answer2:
            try:
                result = safe_eval(
                    answer2,
                    {"torch": torch},
                    {"j": j},
                )

                # allow bool or float identity
                if result.dtype == torch.bool:
                    result = result.float()

                if torch.equal(result, expected2):
                    st.balloons()
                    st.success("🎉 Double Cracked! You're ready for the big leagues.")
                    st.markdown("### 📸 Share your success")
                    st.write(
                        "Take a screenshot of this"
                        "and share it to the **Bonus CampusWire thread**. https://campuswire.com/c/GFC1A6E10/feed/9"
                    )
                else:
                    st.warning(
                        "Not quite. Hint: compare row indices to column indices."
                    )
            except Exception as e:
                st.error(f"❌ Error running your code:\n\n{e}")

# ========================================================================
# FOOTER
# ========================================================================
display_footer()
