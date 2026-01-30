import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import torch
from matplotlib.colors import ListedColormap
from streamlit_monaco import st_monaco

from utils.security import safe_exec

from .game_logic import add_log, damage_player
from .game_state import reset_game
from .levels import get_levels


def show_game_over():
    st.markdown(
        """
    <style>
    .game-over {
        text-align: center;
        padding: 50px;
        background-color: #2b0000;
        border-radius: 10px;
        border: 2px solid #ff0000;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="game-over"><h1>💀 YOU DIED 💀</h1></div>', unsafe_allow_html=True
    )
    st.error(
        f"Valiant attempt, {st.session_state.player_name}. But the gradient was too steep."
    )

    st.metric("Final Level", st.session_state.level)
    st.metric("Gold Collected", st.session_state.gold)

    st.divider()
    st.subheader("📖 Hero's Journal")

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
    if st.button("🔄 REINCARNATE (Respawn)", type="primary", use_container_width=True):
        reset_game()
    st.stop()


def render_shop():
    st.header("🛒 The Traveling Salesman")
    st.markdown(f"Welcome, {st.session_state.player_name}. Spend your Gold here.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/a/a2/Whiskyhogmanay2010.jpg",
            width=50,
        )
        st.subheader("Cho's Whiskey (+25 HP)")
        st.caption("Heals 25 HP! (drinking is only legal after 21)")
        if st.button("Buy (25 G)"):
            if st.session_state.gold >= 25:
                st.session_state.gold -= 25
                st.session_state.hp = min(100, st.session_state.hp + 25)
                add_log("Bought Cho's Whiskey. Glug glug (+25 HP)...", "loot")
                st.rerun()
            else:
                st.error("Not enough gold!")
    with c2:
        st.image(
            "https://minecraft.wiki/images/Totem_of_Undying_JE2_BE2.png?d56eb",
            width=50,
        )
        st.subheader("Becoming Gauss (revive)")
        st.caption(
            "Revives you with 50 HP on death. (Gauss's intellectual heritage is always alive.)"
        )
        if st.button("Buy (75 G)"):
            if st.session_state.gold >= 75:
                st.session_state.gold -= 75
                st.session_state.revival_count = (
                    st.session_state.get("revival_count", 0) + 1
                )
                add_log("Bought Becoming Gauss! (+1 Life)", "loot")
                st.rerun()
            else:
                st.error("Not enough gold!")

    with c3:
        st.image(
            "https://img.icons8.com/color/96/dice.png",
            width=50,
        )
        st.subheader("Bernoulli's Dice")
        st.caption("Re-roll a failed merchant check.")
        if st.button("Buy (50 G)"):
            if st.session_state.gold >= 50:
                st.session_state.gold -= 50
                st.session_state.dice_count = st.session_state.get("dice_count", 0) + 1
                add_log("Bought Bernoulli's Dice! (+1 Re-roll)", "loot")
                st.rerun()
            else:
                st.error("Not enough gold!")

    st.divider()
    if st.button("Leave Shop & Continue"):
        st.session_state.in_shop = False
        if not st.session_state.get("shop_from_boss_lobby", False):
            st.session_state.level += 1
        st.session_state.shop_from_boss_lobby = False
        st.session_state.level_complete = False  # Reset for next level
        st.session_state.merchant_dice_rolled = False
        st.rerun()
    st.stop()


def render_boss_level(
    level_id,
    title,
    caption,
    hint,
    description,
    default_code,
    button_text,
    checker_func,
    damage_label,
    success_msg="",
    height="200px",
):
    """
    Renders a boss level UI component.
    """
    st.subheader(title)
    st.caption(caption)
    if hint:
        with st.expander("Hint"):
            st.markdown(hint)

    if description:
        st.markdown(description)

    st.markdown("**Your Code:**")
    code = st_monaco(
        value=default_code,
        height=height,
        language="python",
        theme="vs-dark",
    )

    key_btn = f"btn_{level_id}"
    key_solved = f"boss_{level_id}_solved"

    if st.button(button_text, key=key_btn):
        try:
            loc = {"torch": torch}
            safe_exec(code, {"torch": torch, "np": np}, loc)
            success, msg = checker_func(loc)
            if success:
                st.success(msg)
                st.session_state[key_solved] = True
                st.rerun()
            else:
                st.error(msg)
                damage_player(10, damage_label)
        except Exception as e:
            st.error(f"Error: {e}")
            damage_player(10, "Syntax Error")

    if st.session_state.get(key_solved, False):
        st.info(f"✅ {success_msg}")


def draw_map():
    if "dungeon_map" in st.session_state and st.session_state.dungeon_map is not None:
        # Prepare map for visualization
        # Categories: 0=Floor (Grey), 1=Wall (Black), 2=Merchant (Gold)
        t_map = st.session_state.dungeon_map.detach().cpu().numpy()
        vis_map = np.zeros_like(t_map)

        # Vectorized mapping
        vis_map[(t_map >= 71) & (t_map < 98)] = 1  # Walls
        vis_map[t_map >= 98] = 2  # Merchant

        # Custom Colormap

        cmap = ListedColormap(["grey", "black", "gold"])

        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(
            vis_map,
            ax=ax,
            cbar=False,
            cmap=cmap,
            vmin=0,
            vmax=2,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f"{st.session_state.player_name}'s Dungeon Map")
        st.pyplot(fig)
        st.caption("Legend: Grey = Ground | Black = Wall | Gold = Merchant")
    else:
        st.write("Map not generated yet.")


def render_level(curr_level_id, lvl_data):
    st.subheader(lvl_data["title"])
    st.markdown(f"*{lvl_data['desc']}*")

    # 1. EDUCATIONAL PHASE (MCQ & FRQ)
    with st.expander(
        "📚 Knowledge Check (Earn 10 Gold!)",
        expanded=not st.session_state.level_complete,
    ):
        # MCQ
        mcq_data = lvl_data.get("mcq")
        if mcq_data:
            st.markdown(f"**Question:** {mcq_data['q']}")
            ans = st.radio(
                "Choose:", mcq_data["opts"], key=f"mcq_{curr_level_id}", index=None
            )
            if st.button("Submit Answer", key=f"btn_mcq_{curr_level_id}"):
                if ans == mcq_data["ans"]:
                    st.success(mcq_data["expl"])
                    if f"mcq_{curr_level_id}" not in st.session_state.answered_mcqs:
                        st.session_state.gold += 10
                        st.session_state.answered_mcqs.add(f"mcq_{curr_level_id}")
                        add_log("Correct Answer! +10 Gold", "loot")
                elif ans is not None:
                    damage_player(20, "Knowledge Check Failed")
                else:
                    st.warning("Please select an option.")

        # FRQ
        frq_q = lvl_data.get("frq")
        if frq_q:
            frq_val = st.text_input(f"Reflection: {frq_q}", key=f"frq_{curr_level_id}")
            if frq_val:
                st.caption(
                    f"✅ *{st.session_state.player_name}'s thoughts have been inscribed in the journal.*"
                )

    # 2. CODING PHASE
    st.info(f"**OBJECTIVE:** {lvl_data['task']} (Reward: 50 Gold)")
    col_editor, col_vis = st.columns([1.5, 1])

    with col_vis:
        st.caption("Visualizer")

        if curr_level_id == 0:
            st.write(f"{st.session_state.player_name}'s Dungeon Map:")
            draw_map()
        elif curr_level_id == 1:
            st.write("Target: (5, 3) Matrix of Ones")
            if "tensor_x" in st.session_state and isinstance(
                st.session_state.tensor_x, torch.Tensor
            ):
                st.write("Your Tensor X:")
                fig, ax = plt.subplots(figsize=(3, 3))
                sns.heatmap(
                    st.session_state.tensor_x.numpy(),
                    ax=ax,
                    cbar=False,
                    cmap="Blues",
                    annot=True,
                    fmt=".1f",
                )
                st.pyplot(fig)
        elif curr_level_id == 3:
            st.write("Target: 1st Column")

    with col_editor:
        if st.session_state.level_complete:
            st.success("Level Complete!")

            # --- LEVEL 0: EV QUESTIONS ---
            if curr_level_id == 0 and not st.session_state.ev_questions_solved:
                st.info(
                    "🧠 Wait! Before you proceed, you must demonstrate your understanding of Expectation."
                )
                st.markdown(
                    "The map has **100** tiles (10x10). Code: `torch.randint(0, 100, ...)`"
                )
                c1, c2 = st.columns(2)
                with c1:
                    ev_wall = st.number_input(
                        "Expected # of Walls (71-97 inclusive)?",
                        min_value=0,
                        max_value=100,
                        key="ev_wall_input",
                    )
                with c2:
                    ev_merch = st.number_input(
                        "Expected # of Merchants (98-99 inclusive)?",
                        min_value=0,
                        max_value=100,
                        key="ev_merch_input",
                    )

                if st.button("Submit Calculations"):
                    # Walls: 71..97 is 27 integers -> 27%
                    # Merchants: 98..99 is 2 integers -> 2%
                    if ev_wall == 27 and ev_merch == 2:
                        st.success("Correct! E[Walls] = 27, E[Merchants] = 2.")
                        st.session_state.ev_questions_solved = True
                        st.rerun()
                    else:
                        damage_player(
                            10, "Math Error! (Remember: count inclusive range)"
                        )
                st.stop()  # Stop here until solved

            # --- MERCHANT MECHANIC ---
            if (
                not st.session_state.merchant_dice_rolled
                and st.session_state.merchant_count > 0
            ):
                # NEW: Probability Check
                if not st.session_state.prob_question_solved:
                    st.info("📊 One last check. You are about to roll a 6-sided die.")
                    prob = st.number_input(
                        "What is the probability of rolling >= 4? (Enter as decimal 0.0-1.0)",
                        0.0,
                        1.0,
                    )
                    if st.button("Submit Probability"):
                        if abs(prob - 0.5) < 0.01:
                            st.success("Correct! P(>=4) = 3/6 = 0.5")
                            st.session_state.prob_question_solved = True
                            st.rerun()
                        else:
                            damage_player(
                                5,
                                "Probability Error! (Hint: outcomes 4, 5, 6 vs total 6)",
                            )
                    st.stop()
                st.markdown(
                    f"**{st.session_state.merchant_count} Merchants** are hidden in the dungeon."
                )
                st.markdown(
                    "Rules: Roll a **d6**. If you roll **4, 5, or 6** (Prob: 0.5), you find a merchant."
                )
                col_r, col_s = st.columns(2)
                with col_r:
                    if st.button(
                        "🎲 Roll for Merchant (Risk!)",
                        help="If you roll < 4, you find nothing.",
                    ):
                        st.session_state.merchant_dice_rolled = True

                        # Use offset RNG so results change on subsequent rolls (gameplay memory)
                        # We use magic_number + offset to ensure it's determined but advances
                        rng = np.random.default_rng(
                            st.session_state.magic_number
                            + st.session_state.get("rng_offset", 0)
                        )
                        roll = rng.integers(1, 7, endpoint=True)
                        st.session_state.rng_offset = (
                            st.session_state.get("rng_offset", 0) + 1
                        )

                        st.session_state.last_roll = roll
                        if roll >= 4:
                            st.session_state.shop_available = True
                            st.session_state.merchant_count -= 1
                            add_log(
                                f"Rolled {roll} (>=4). Merchant found! {st.session_state.merchant_count} left.",
                                "loot",
                            )
                        else:
                            st.session_state.shop_available = False
                            add_log(
                                f"Rolled {roll} (<4). The merchant remains hidden.",
                                "info",
                            )
                        st.rerun()
                with col_s:
                    if st.button("👇 Descend (Skip Merchant)"):
                        st.session_state.level += 1
                        st.session_state.level_complete = False
                        st.session_state.merchant_dice_rolled = False
                        st.rerun()

            elif st.session_state.merchant_dice_rolled:
                # UI Feedback
                if st.session_state.last_roll > 0:
                    st.write(f"🎲 Dice Roll: **{st.session_state.last_roll}**")

                if st.session_state.shop_available:
                    st.info(
                        f"🛒 **Merchant Spotted!** ({st.session_state.merchant_count} remaining in dungeon)"
                    )
                    if st.button("Proceed to Shop"):
                        st.session_state.in_shop = True
                        st.rerun()
                else:
                    st.info("🌑 The path is dark. The merchant did not appear.")

                    # Lucky Dice Logic
                    if st.session_state.get("dice_count", 0) > 0:
                        st.write(f"You have {st.session_state.dice_count} Lucky Dice.")
                        if st.button("🎲 Use Lucky Dice (Re-roll!)"):
                            st.session_state.dice_count -= 1
                            st.session_state.merchant_dice_rolled = False
                            add_log("Used Lucky Dice to re-roll!", "info")
                            st.rerun()

                    if st.button("Descend to Next Level"):
                        st.session_state.level += 1
                        st.session_state.level_complete = False
                        st.session_state.merchant_dice_rolled = False
                        st.rerun()
            else:
                # No merchants left
                st.info("🌑 No merchants left in the dungeon.")
                if st.button("Descend to Next Level"):
                    st.session_state.level += 1
                    st.session_state.level_complete = False
                    st.session_state.merchant_dice_rolled = False
                    st.rerun()

        else:
            if "hint" in lvl_data:
                with st.expander("💡 Hint"):
                    st.code(lvl_data["hint"])

            st.caption("Note: `torch` and `np` (numpy) are already imported for you.")

            st.markdown(f"**{st.session_state.player_name}'s Code:**")
            code_input = st_monaco(
                value=lvl_data["starter_code"],
                height="200px",
                language="python",
                theme="vs-dark",
            )

            if st.button("Run Rune ⚡"):
                # Enforce Journaling
                frq_key = f"frq_{curr_level_id}"
                has_frq = lvl_data.get("frq")
                user_reflection = st.session_state.get(frq_key, "")

                if has_frq and not user_reflection.strip():
                    damage_player(
                        20, "Empty Journal! Reflection required before casting."
                    )
                else:
                    # Persist the FRQ answer so it survives level transitions
                    if has_frq:
                        st.session_state[f"saved_frq_{curr_level_id}"] = user_reflection

                    try:
                        context = lvl_data["context_setup"]()
                        context["__code__"] = code_input
                        safe_exec(code_input, {"torch": torch, "np": np}, context)
                        success, feedback = lvl_data["checker"](context)

                        if success:
                            st.success(feedback)
                            st.session_state.xp += 100
                            st.session_state.gold += 50
                            add_log(f"Cleared Floor {curr_level_id}! +50 Gold", "level")

                            # Set level complete flag and rerun to update UI to 'Proceed' state
                            st.session_state.level_complete = True
                            st.rerun()
                        else:
                            st.warning(feedback)
                            damage_player(30, "Wrong Answer Recoil")
                    except Exception as e:
                        st.error(f"Execution Error: {e}")
                        damage_player(30, "Syntax Error Spark")
