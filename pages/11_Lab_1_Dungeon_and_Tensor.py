import os

import streamlit as st

from labs.lab1_dungeon_and_tensor.game_logic import add_log
from labs.lab1_dungeon_and_tensor.game_state import init_game, reset_game, set_seed
from labs.lab1_dungeon_and_tensor.levels import (
    check_level_6,
    check_level_7,
    check_level_8,
    check_level_9,
    check_level_10,
    check_level_11,
    get_levels,
)
from labs.lab1_dungeon_and_tensor.save_load import load_game, save_game
from labs.lab1_dungeon_and_tensor.ui_components import (
    render_boss_level,
    render_level,
    render_shop,
    show_game_over,
)
from utils.ui import display_footer

# ==============================================================================
# CONFIG & STATE
# ==============================================================================
st.set_page_config(
    page_title="Lab 1: Dungeon and Tensor", page_icon="⚔️", layout="wide"
)

init_game()
set_seed()

# ==============================================================================
# START SCREEN
# ==============================================================================
if not st.session_state.game_started:
    st.title("🏹 Lab 1: Dungeon and Tensor")
    st.markdown("### Enter the Tensor Dungeon")
    st.info(
        "Before you begin, you must attune yourself to the randomness of the world."
    )

    col1, col2 = st.columns(2)
    with col1:
        name_input = st.text_input("Hero Name", placeholder="e.g. Alyx the Tensor")
    with col2:
        seed_input = st.number_input(
            "Magic Number (Seed)",
            min_value=0,
            max_value=9999,
            value=42,
            help="This controls the randomness of your journey.",
        )

    if st.button("Enter Dungeon"):
        if name_input.strip():
            st.session_state.player_name = name_input
            st.session_state.magic_number = int(seed_input)
            st.session_state.game_started = True
            add_log(f"Hero {name_input} entered with Seed {seed_input}.", "info")
            st.rerun()
        else:
            st.error("You must have a name!")
    st.stop()


# Check Game Over
if st.session_state.hp <= 0 and st.session_state.game_started:
    show_game_over()


# ==============================================================================
# UI COMPONENTS
# ==============================================================================
st.title(f"🏹 Dungeon and Tensor: {st.session_state.player_name}")
st.markdown("### A Roguelike PyTorch Adventure")

# --- SIDEBAR (STATS) ---
with st.sidebar:
    st.header(f"Level {st.session_state.level}")
    # Fix: Clamp progress value to avoid StreamlitAPIException
    hp_percent = max(0.0, st.session_state.hp / st.session_state.max_hp)
    st.progress(hp_percent, f"HP: {st.session_state.hp}/{st.session_state.max_hp}")
    st.metric("Gold", st.session_state.gold)
    if st.session_state.get("revival_count", 0) > 0:
        st.metric("Revivals", st.session_state.revival_count)

    st.divider()
    st.subheader("📜 Adventure Log")
    for log in st.session_state.logs[:10]:
        st.caption(log)

    if st.button("restart_run"):
        reset_game()

    st.divider()
    st.subheader("💾 Game Progress")
    if st.button("Save Game", use_container_width=True):
        save_game()
    if st.button("Load Game", use_container_width=True):
        load_game()

    # --- DEV CHEATS ---
    if os.environ.get("ST_DEV_MODE"):
        with st.expander("🛠️ Dev Tools"):
            jump_lvl = st.number_input(
                "Jump to Level", min_value=0, max_value=13, value=st.session_state.level
            )
            if st.button("✈️ Jump"):
                st.session_state.level = jump_lvl
                st.session_state.level_complete = False
                st.session_state.merchant_dice_rolled = False
                st.rerun()
            if st.button("💰 Add 100 Gold"):
                st.session_state.gold += 100
                st.rerun()
            if st.button("🛍️ Go to shop"):
                st.session_state.in_shop = True
                st.rerun()
            if st.button("🏆 Go to Success Screen"):
                st.session_state.level = 6
                st.session_state.boss_fight_started = True
                st.session_state.boss_phase = 2
                st.session_state.boss_9_solved = True
                st.session_state.boss_10_solved = True
                st.session_state.boss_11_solved = True
                st.rerun()

# --- SHOP UI ---
if st.session_state.in_shop:
    render_shop()

# --- MAIN GAME AREA ---
curr_level_id = st.session_state.level

if str(curr_level_id) not in st.session_state.get("completed_levels", []):
    # Ensure level_complete key exists
    if "level_complete" not in st.session_state:
        st.session_state.level_complete = False

if curr_level_id > 5:
    # --- BOSS LOBBY (Merchant Chance) ---
    if curr_level_id == 6 and not st.session_state.get("boss_fight_started", False):
        st.header("🔥 The Threshold of the Void")
        st.markdown(
            """
            You stand at the edge of the final boss floor.

            A spectral merchant has set up a small camp nearby.
            *\"Last chance to stock up, traveler...\"* he whispers.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🛍️ Visit the Traveling Salesman"):
                st.session_state.in_shop = True
                st.session_state.shop_from_boss_lobby = True
                st.rerun()

        with col2:
            if st.button("👹 ENTER BOSS FIGHT", type="primary"):
                st.session_state.boss_fight_started = True
                st.rerun()

        st.stop()

    # BOSS LEVEL
    st.header("👹 BOSS FLOOR: THE VANISHING GRADIENT")

    # --- BOSS STATE INIT ---
    if "boss_phase" not in st.session_state:
        st.session_state.boss_phase = 1

    # Initialize sub-task states if not present
    for i in range(6, 12):
        if f"boss_{i}_solved" not in st.session_state:
            st.session_state[f"boss_{i}_solved"] = False

    # --- PHASE 1: THE STRUCTURE OF SPACE ---
    if st.session_state.boss_phase == 1:
        st.warning(
            "Phase 1: The Structure of Space. Prove your understanding of the axioms."
        )

        c1, c2, c3 = st.columns(3)

        # Level 6: Axioms
        with c1:
            render_boss_level(
                6,
                "6. Vector Space",
                "Task: Define `is_commutative(add_func)`",
                "**Note:** Due to floating point error, you can check equality with `equal` or `torch.allclose(lhs, rhs)`",
                """
                **Input:** `add_func` (Callable) - A function `f(u, v)` representing vector addition. \n
                **Output:** `bool` - Return `True` if `add_func(u, v)` equals `add_func(v, u)` for random vectors `u`, `v`. \n
                """,
                "def is_commutative(add_func):\n    u = torch.randn(5)\n    v = torch.randn(5)\n    return ...",
                "Cast Axiom",
                check_level_6,
                "Axiom Failure",
                "Axiom Proven",
            )

        # Level 7: Inner Product
        with c2:
            render_boss_level(
                7,
                "7. Inner Product",
                "Task: Check Linearity: <cx+w, y> == c<x, y> + <w,y>",
                "**Note:** Due to floating point error, you can check equality with `torch.allclose(lhs, rhs)`",
                """
                **Input:** `func` (Callable) - An inner product function `f(u, v)`. \n
                **Output:** `bool` - Return `True` if linearity holds: `<cx+w, y> == c<x, y> + <w,y>`. \n
                """,
                "def check_inner_product(func):\n    u = torch.randn(10)\n    v = torch.randn(10)\n    w = torch.randn(10)\n    c = torch.randn(1).item()\n    # Check linearity with tolerance!\n    lhs = ...\n    rhs = ...\n    return ...",
                "Cast Inner Product",
                check_level_7,
                "Product Mismatch",
                "Inner Product Established",
                height="250px",
            )

        # Level 8: Norm
        with c3:
            render_boss_level(
                8,
                "8. L2 Norm",
                "Task: Implement `my_l2_norm(x)`",
                """For loop is not allowed. Consider using `torch.sum` and `torch.sqrt`.""",
                """
                **Input:** `x` (torch.Tensor) - A 1D tensor. \n
                **Output:** `float` - The L2 norm (Euclidean length) of `x`. \n
                """,
                "def my_l2_norm(x):\n    return ...",
                "Cast Norm",
                check_level_8,
                "Norm Failure",
                "Magnitude Calibrated",
            )

        # Check Phase 1 Completion
        if (
            st.session_state.boss_6_solved
            and st.session_state.boss_7_solved
            and st.session_state.boss_8_solved
        ):
            st.success(
                "The fabric of space stabilizes! The boss retreats to a higher dimension."
            )
            if st.button("Ascend to Phase 2"):
                st.session_state.boss_phase = 2
                st.rerun()

    # --- PHASE 2: THE BASIS OF REALITY ---
    elif st.session_state.boss_phase == 2:
        st.error("Phase 2: The Basis of Reality. Define the coordinates!")

        c1, c2, c3 = st.columns(3)

        # Level 9: Weighted IP
        with c1:
            render_boss_level(
                9,
                "9. Weighted IP",
                "Task: `weighted_ip(x, y, W)` = x^T W y",
                """1. @ is a shorthand for matrix multiplication.\n
                2. You can use `T` for transpose.""",
                """
                **Input:** `x`, `y` (torch.Tensor) - Vectors; `W` (torch.Tensor) - Weight matrix. \n
                **Output:** `float` or `Tensor` - The scalar result of `x^T W y`. \n
                """,
                "def weighted_ip(x, y, W):\n    return ...",
                "Cast Weight",
                check_level_9,
                "Weight Crush",
                "Weight Balanced",
            )

        # Level 10: Basis
        with c2:
            render_boss_level(
                10,
                "10. Basis Check",
                "Task: `is_basis(vectors)`",
                """
                1. You do NOT need to check for orthogonality. A basis can be non-orthogonal.\n
                2. You can use `torch.vstack(vectors)` to stack the vectors vertically.\n
                3. You can use `torch.linalg.matrix_rank` to check the rank of the matrix.\n
                4. You can use `torch.numel` as a replacement for `len`.
                """,
                """
                **Input:** `vectors` (List[torch.Tensor]) - A list of vectors. \n
                **Output:** `bool` - True if they form a basis, False otherwise. \n
                """,
                "def is_basis(vectors):\n    return ...",
                "Cast Basis",
                check_level_10,
                "Dependence Error",
                "Basis Constructed",
            )

        # Level 11: Change of Basis
        with c3:
            render_boss_level(
                11,
                "11. Change of Basis",
                "Task: `get_coordinates(v, B)` -> `c` s.t. `B @ c = v`",
                """*1. You can assume that `B` is a basis (linearly independent and spanning) and therefore invertible.\n
                2. You can use `torch.inverse` for matrix inversion.""",
                """
                **Input:** `v` (torch.Tensor) - Target vector; `B` (torch.Tensor) - Matrix with basis vectors as columns. \n
                **Output:** `torch.Tensor` - Coordinates `c` such that `B @ c = v`. \n
                """,
                "def get_coordinates(v, B):\n    # v: shape (n,)\n    # B: shape (n,n), cols are basis vectors\n    return ...",
                "Cast Coordinates",
                check_level_11,
                "Coordinate Shift Fail",
                "Perspective Shifted",
            )

        # Check Phase 2 Completion / VICTORY
        if (
            st.session_state.boss_9_solved
            and st.session_state.boss_10_solved
            and st.session_state.boss_11_solved
        ):
            st.balloons()
            st.success("YOU HAVE CONQUERED THE DIMENSIONS! VICTORY!")
            st.info("The tensor is yours to command.")

            st.divider()
            st.subheader(f"📖 {st.session_state.player_name}'s Journal")
            LEVELS = get_levels()
            for lvl_id in sorted(LEVELS.keys()):
                key = f"saved_frq_{lvl_id}"
                if key in st.session_state:
                    q = LEVELS[lvl_id].get("frq", "Unknown Question")
                    a = st.session_state[key]
                    if a:
                        st.markdown(f"**Floor {lvl_id}: {q}**")
                        st.info(f"_{a}_")

            st.divider()
            st.subheader("📜 Adventure Log")
            for log in st.session_state.logs:
                st.caption(log)

            st.divider()

            st.markdown("### 📸 Share your Victory")
            st.markdown(
                "Take a screenshot of your journal and log above, and share it to the CampusWire post!"
            )
            st.link_button(
                "Go to CampusWire Post", "https://campuswire.com/c/GFC1A6E10/feed/26"
            )

            st.divider()

            if st.button("Restart Adventure"):
                reset_game()


else:
    # NORMAL LEVELS
    LEVELS = get_levels()
    if curr_level_id in LEVELS:
        lvl_data = LEVELS[curr_level_id]
        render_level(curr_level_id, lvl_data)

# ========================================================================
# FOOTER
# ========================================================================
display_footer()
