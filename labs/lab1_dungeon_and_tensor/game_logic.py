import streamlit as st


def add_log(msg, type="info"):
    icon_map = {
        "combat": "⚔️",
        "error": "💀",
        "loot": "💰",
        "level": "🆙",
    }
    icon = icon_map.get(type, "ℹ️")
    st.session_state.logs.insert(0, f"{icon} {msg}")  # Prepend log


def damage_player(amount, reason=""):
    st.session_state.hp -= amount
    msg = f"{st.session_state.player_name} took {amount} DMG! {reason}"
    add_log(msg, "combat")
    if st.session_state.hp <= 0:
        # Revival Logic
        if st.session_state.get("revival_count", 0) > 0:
            st.session_state.revival_count -= 1
            st.session_state.hp = 50
            st.toast("🌟 TOTEM OF UNDYING CONSUMED!", icon="🌟")
            add_log("BUT REFUSED TO DIE! Totem consumed. HP restored to 50.", "combat")
        else:
            st.session_state.hp = 0  # Clamp at 0
            st.rerun()  # Trigger global game over screen
    else:
        st.error(f"💔 -{amount} HP: {reason}")  # Explicit HP Feedback
