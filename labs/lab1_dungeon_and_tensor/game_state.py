import numpy as np
import streamlit as st
import torch


def init_game():
    if "level" not in st.session_state:
        # Player State - Initialize FIRST
        st.session_state.player_name = ""
        st.session_state.magic_number = None
        st.session_state.game_started = False

        # Game State
        st.session_state.level = 0
        st.session_state.hp = 100
        st.session_state.max_hp = 100
        st.session_state.xp = 0
        st.session_state.gold = 0
        st.session_state.revival_count = 0
        st.session_state.dice_count = 0
        st.session_state.inventory = []
        st.session_state.logs = ["Welcome to the Tensor Dungeon. Prepare yourself."]
        st.session_state.in_shop = False
        st.session_state.dungeon_map = None
        st.session_state.answered_mcqs = set()

        # Merchant Mechanics
        st.session_state.merchant_count = 0
        st.session_state.merchant_dice_rolled = False
        st.session_state.last_roll = 0
        st.session_state.shop_available = False

        # Educational Quest State
        st.session_state.ev_questions_solved = False
        st.session_state.prob_question_solved = False

        # RNG State
        st.session_state.rng_offset = 0


def reset_game():
    st.session_state.clear()
    init_game()
    st.rerun()


def set_seed():
    if st.session_state.magic_number is not None:
        torch.manual_seed(st.session_state.magic_number)
        np.random.seed(st.session_state.magic_number)
