"""
Lab 2: Optimization

"""

import os
import sys
import time

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labs.lab2_optimization.data_utils import create_batched_indices, generate_dataset
from labs.lab2_optimization.gradient_clipping import simulate_gradient_clipping
from labs.lab2_optimization.gradient_descent import make_theta_array
from labs.lab2_optimization.loss_functions import doublewell_gradient, doublewell_loss
from labs.lab2_optimization.loss_surface import GradientDescentLinearRegression
from labs.lab2_optimization.visualization import draw_line

if "lab_part" not in st.session_state:
    st.session_state["lab_part"] = 1

if "part1_completed" not in st.session_state:
    st.session_state["part1_completed"] = False

if "part2_completed" not in st.session_state:
    st.session_state["part2_completed"] = False

if "part3_completed" not in st.session_state:
    st.session_state["part3_completed"] = False

if "part4_completed" not in st.session_state:
    st.session_state["part4_completed"] = False

if "part1" not in st.session_state:
    st.session_state["part1"] = {
        "rng": np.random.default_rng(),
        "pts": 50,
        "max_updates": 200,
        "init_alpha": 0.001,
        "init_batch": 1,
        "init_theta": (0, 4),
    }

if "part2" not in st.session_state:
    st.session_state["part2"] = {
        "rng": np.random.default_rng(42),
        "w_current": 0.0,
        "b_current": 0.0,
        "path_2d": [(0.0, 0.0)],
        "alpha": 0.1,
    }

part1 = st.session_state["part1"]
rng = part1["rng"]
pts = part1["pts"]
max_updates = part1["max_updates"]
init_alpha = part1["init_alpha"]
init_batch = part1["init_batch"]
init_theta = part1["init_theta"]

if "batch" in st.session_state:
    batch = st.session_state["batch"]
else:
    batch = init_batch

if "step" not in st.session_state:
    st.session_state["step"] = 0

if "running" not in st.session_state:
    st.session_state["running"] = False

if "animation_speed" not in st.session_state:
    st.session_state["animation_speed"] = 0.05

if "form_answers" not in st.session_state:
    st.session_state["form_answers"] = {}

if "part1_questions_submitted" not in st.session_state:
    st.session_state["part1_questions_submitted"] = False
if "part2_questions_submitted" not in st.session_state:
    st.session_state["part2_questions_submitted"] = False
if "part3_questions_submitted" not in st.session_state:
    st.session_state["part3_questions_submitted"] = False
if "part4_questions_submitted" not in st.session_state:
    st.session_state["part4_questions_submitted"] = False
if "part5_code_submitted" not in st.session_state:
    st.session_state["part5_code_submitted"] = False
if "part5_location_submitted" not in st.session_state:
    st.session_state["part5_location_submitted"] = False
if "part5_questions_submitted" not in st.session_state:
    st.session_state["part5_questions_submitted"] = False
if "part5_code_correct" not in st.session_state:
    st.session_state["part5_code_correct"] = False


def restore_form_keys(keys):
    saved = st.session_state["form_answers"]
    for key in keys:
        if key in saved and (
            key not in st.session_state
            or st.session_state.get(key) is None
            or st.session_state.get(key) == ""
        ):
            st.session_state[key] = saved[key]


def stash_form_keys(keys):
    saved = st.session_state["form_answers"]
    for key in keys:
        if key in st.session_state:
            saved[key] = st.session_state[key]


def mark_part1_submitted():
    st.session_state["part1_questions_submitted"] = True


def mark_submitted(flag_key):
    st.session_state[flag_key] = True


def update():
    (fit_coefs, df, _) = st.session_state["data"]
    alpha = st.session_state["alpha"]
    batch = st.session_state["batch"]
    # Regenerate pt_idx with current batch size and max_updates
    pt_idx = create_batched_indices(pts, batch, max_updates, rng)
    st.session_state["data"] = (fit_coefs, df, pt_idx)
    st.session_state["theta_arr"] = make_theta_array(
        df, pt_idx, alpha, batch, init_theta, max_updates, pts
    )
    st.session_state["step"] = 0


def run_step():
    st.session_state["running"] = True


def clear_data():
    if "data" in st.session_state:
        del st.session_state["data"]
    if "theta_arr" in st.session_state:
        del st.session_state["theta_arr"]
    st.session_state["step"] = 0


def update_part2():
    (df_train, df_val, true_coefs) = st.session_state["part2_data"]
    st.session_state["part2_alpha"]
    batch = st.session_state["part2_batch"]
    degree = st.session_state["part2_degree"]
    pts_train = len(df_train)

    create_batched_indices(pts_train, batch, part2_max_updates, part2_rng)
    init_theta = np.full(degree + 1, 0.001)
    init_theta[0] = float(df_train["y"].mean())


def clear_part2_data():
    pass


if "data" in st.session_state:
    (fit_coefs, df, pt_idx) = st.session_state["data"]
else:
    df, fit_coefs = generate_dataset(pts, rng)
    part1["init_theta"] = (0, float(df["y"].mean()))
    init_theta = part1["init_theta"]
    pt_idx = create_batched_indices(pts, init_batch, max_updates, rng)
    st.session_state["data"] = (fit_coefs, df, pt_idx)

if "theta_arr" in st.session_state:
    theta_arr = st.session_state["theta_arr"]
else:
    batch = st.session_state.get("batch", init_batch)
    alpha = st.session_state.get("alpha", init_alpha)
    theta_arr = make_theta_array(df, pt_idx, alpha, batch, init_theta, max_updates, pts)
    st.session_state["theta_arr"] = theta_arr

st.title("Lab 2: Downhill Descent: An Optimization Adventure 🏔️")

# --- DEV CHEATS ---
if os.environ.get("ST_DEV_MODE"):
    with st.sidebar:
        st.header("🛠️ Dev Tools")
        selected_part = st.selectbox(
            "Jump to Part",
            options=[1, 2, 3, 4, 5],
            index=max(0, min(4, st.session_state["lab_part"] - 1)),
        )
        if selected_part != st.session_state["lab_part"]:
            st.session_state["lab_part"] = selected_part
            st.rerun()


# ===== Part 1: Warm-up =====
if st.session_state["lab_part"] == 1:
    st.subheader("🔥 Part 1: Warm-up - Understanding Gradients")
    st.markdown(
        """Before we start skiing, let's make sure we understand the math behind gradient descent. Gradients tell us which direction to move and how fast!"""
    )

    st.markdown(
        """
    Below is the loss function $L(w) = (w - 3)^2$ for a weight value w. Your goal is to find the value of $w$ that minimizes $L(w)$ using gradient descent (we call this w*).
    """
    )

    st.markdown("### Visualizing the Gradient")
    x0 = st.slider(
        "Choose a parameter w and see how the tangent line changes!",
        min_value=-2.0,
        max_value=7.0,
        value=1.0,
        step=0.1,
        key="part1_x0",
    )
    f_x0 = (x0 - 3) ** 2
    grad_x0 = 2 * (x0 - 3)

    # Curve data
    x_curve = np.linspace(-2, 7, 200)
    y_curve = (x_curve - 3) ** 2
    df_curve = pd.DataFrame({"x": x_curve, "y": y_curve})

    # Tangent line at w0: y = L(w0) + L'(w0) * (w - w0)
    y_tangent = f_x0 + grad_x0 * (x_curve - x0)
    df_tangent = pd.DataFrame({"x": x_curve, "y": y_tangent})

    # Current point
    df_point = pd.DataFrame({"x": [x0], "y": [f_x0]})

    # Slope label just above/below the tangent line
    y_range = float(y_curve.max() - y_curve.min())
    y_offset = 0.5 * y_range
    label_y = f_x0 - y_offset
    intercept = f_x0 - grad_x0 * x0
    slope_label_df = pd.DataFrame(
        {"x": [x0], "y": [label_y], "label": [f"y = {grad_x0:.2f}w + {intercept:.2f}"]}
    )

    curve = (
        alt.Chart(df_curve)
        .mark_line(color="#4C78A8", strokeWidth=3)
        .encode(
            x=alt.X("x:Q", title="w"),
            y=alt.Y("y:Q", title="L(w)"),
        )
    )
    tangent = (
        alt.Chart(df_tangent)
        .mark_line(color="#F58518", strokeDash=[6, 4], strokeWidth=2)
        .encode(
            x="x:Q",
            y="y:Q",
        )
    )
    point = (
        alt.Chart(df_point)
        .mark_circle(color="#E45756", size=120)
        .encode(
            x="x:Q",
            y="y:Q",
        )
    )
    slope_label = (
        alt.Chart(slope_label_df)
        .mark_text(color="#F58518", size=18, align="center")
        .encode(
            x="x:Q",
            y="y:Q",
            text="label:N",
        )
    )

    chart = (curve + tangent + point + slope_label).properties(height=350)
    st.altair_chart(chart, use_container_width=True)

    if not st.session_state.get("part1_questions_submitted", False):
        restore_form_keys(
            [
                "q0_1",
                "q0_2",
                "q0_3",
                "q0_4",
                "q0_5",
                "q0_6",
                "q0_7",
            ]
        )
    with st.form("part1_questions"):
        q0_1 = st.text_area(
            "Q1: What is the gradient ∂L/∂w? (Hint: Derive from the formula for L(w)",
            key="q0_1",
            height=100,
        )

        q0_2 = st.text_area(
            "Q2: What is the gradient ∂L/∂w at w = 8?",
            key="q0_2",
            height=60,
        )

        q0_3 = st.radio(
            "Q3: What is the sign of the gradient at w = 8?",
            ["Positive", "Negative"],
            key="q0_3",
            index=None,
        )

        q0_4 = st.radio(
            "Q4: At w = 8, should we increase or decrease our estimate of w?",
            ["Increase", "Decrease"],
            key="q0_4",
            index=None,
        )

        q0_5 = st.radio(
            "Q5: Is the gradient at w = -1 the same direction as at w = 8?",
            ["Yes", "No"],
            key="q0_5",
            index=None,
        )

        q0_6 = st.radio(
            "Q6: Is the gradient at w = -1 the same magnitude as at w = 8?",
            ["Yes", "No"],
            key="q0_6",
            index=None,
        )

        q0_7 = st.text_area(
            "Q7: When would the gradient be 0 (what value of w)? What does this tell us about L(w) at this point?",
            key="q0_7",
            height=80,
        )

        if q0_7.strip():
            st.info(
                "The gradient is zero at w = 3 (you can confirm this with the visualization!). "
                "This tells us that we are at a **minimum** of the loss function!"
            )

        submitted = st.form_submit_button(
            "Submit answers", on_click=mark_part1_submitted
        )

        if submitted:
            stash_form_keys(
                [
                    "q0_1",
                    "q0_2",
                    "q0_3",
                    "q0_4",
                    "q0_5",
                    "q0_6",
                    "q0_7",
                ]
            )
            # Check that at least some answers were provided
            if (
                q0_1.strip()
                and q0_2.strip()
                and q0_3
                and q0_4
                and q0_5
                and q0_6
                and q0_7.strip()
            ):
                # Check q0_1 answer - should be 2x-6 or equivalent
                answer_normalized = q0_1.strip().replace(" ", "").lower()
                correct_q0_1 = any(
                    variant == answer_normalized
                    for variant in ["2w-6", "2(w-3)", "2*w-6", "2*(w-3)"]
                )

                # Check q0_2 answer
                answer_q0_2 = q0_2.strip().replace(" ", "").lower()
                correct_q0_2 = "10" == answer_q0_2  # or "10" in answer_q0_2

                # Check q0_3 answer - should be Positive
                correct_q0_3 = q0_3 == "Positive"

                # Check q0_4 answer
                correct_q0_4 = q0_4 == "Decrease"

                # Check q0_5 answer - should be No (different directions)
                # At x = -1: 2(-1) - 6 = -8 (negative, LEFT)
                # At x = 8: 2(8) - 6 = 10 (positive, RIGHT)
                correct_q0_5 = q0_5 == "No"

                # Check q0_6 answer - should be No (different magnitude)
                # At x = -1: |-8| = 8
                # At x = 8: |10| = 10
                correct_q0_6 = q0_6 == "No"

                # Check q0_7 answer - gradient is 0 at x = 3
                answer_q0_7 = q0_7.strip().replace(" ", "").lower()
                correct_q0_7 = "3" in answer_q0_7 and (
                    "min" in answer_q0_7 or "optim" in answer_q0_7
                )

                if (
                    correct_q0_1
                    and correct_q0_2
                    and correct_q0_3
                    and correct_q0_4
                    and correct_q0_5
                    and correct_q0_6
                    and correct_q0_7
                ):
                    st.session_state["part0_completed"] = True
                    st.success("🎉 You're ready for the slopes!")
                else:
                    st.session_state["part0_completed"] = False
                    error_msg = "Not quite right. "
                    if not correct_q0_1:
                        error_msg += "Check your derivative calculation for question 1 (hint: use the chain rule). "
                    if not correct_q0_2:
                        error_msg += "Check question 2 - plug w=8 into your gradient. "
                    if not correct_q0_3:
                        error_msg += "Check question 3 - determine the sign of the gradient at w=8. "
                    if not correct_q0_4:
                        error_msg += "Check question 4 - to minimize the loss, we move opposite the direction of the gradient. "
                    if not correct_q0_5:
                        error_msg += "Check question 5 - evaluate the gradient at w=-1 and compare the direction. "
                    if not correct_q0_6:
                        error_msg += "Check question 6 - evaluate the gradient at w=-1 and compare the magnitude. "
                    if not correct_q0_7:
                        error_msg += "Check question 7 - the gradient is zero at the minimum of the function. "
                    st.error(error_msg)
            else:
                st.session_state["part0_completed"] = False
                st.error("Please answer all questions before submitting.")

    st.session_state["part1_questions_submitted"] = False

    # Navigation button to move to Part 1 - only show if validation passed
    if st.session_state.get("part0_completed", False):
        st.markdown("---")
        if st.button(
            "Continue to Part 2 →", use_container_width=True, key="continue_to_part2"
        ):
            st.session_state["lab_part"] = 2
            st.rerun()


# ===== Part 2: Intro + Controls =====
elif st.session_state["lab_part"] == 2:
    st.subheader("🐰 Part 2: Bunny Hill - Learning the Basics")
    st.markdown(
        """Welcome to the Bunny Hill! The points plotted below are scenic viewpoints along the ski slope that you want to see on your way down the mountain. However, as a beginner skier, unfortunately you have not learned how to turn. Your goal is to find the perfect ski line $y = w_0 x + w_1$ that gets you close to the viewpoints. Set the parameters below to use gradient descent to gradually adjust your path closer to the viewpoints. Happy skiing! ⛷️"""
    )

    xmin = df["x"].min()
    xmax = df["x"].max()
    ymin = df["y"].min()
    ymax = df["y"].max()

    chart1 = draw_line(fit_coefs, [xmin, xmax])

    chart2 = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[xmin, xmax])),
            y=alt.Y(
                "y", scale=alt.Scale(domain=[ymin, ymax]), axis=alt.Axis(titleAngle=0)
            ),
            color=alt.Color(
                "color:N",
                legend=None,
                scale=alt.Scale(domain=[0, 1], range=["#B0B0B0", "#B0B0B0"]),
            ),
        )
    )

    fit_1, fit_0 = fit_coefs

    learn = st.slider(
        "Learning Rate",
        min_value=0.0,
        max_value=0.025,
        step=0.001,
        value=init_alpha,
        key="alpha",
        format="%.3f",
    )

    batch = st.slider(
        "Batch Size",
        min_value=1,
        max_value=pts,
        step=1,
        value=init_batch,
        key="batch",
    )

    # Track previous slider values to detect changes
    if "prev_alpha" not in st.session_state:
        st.session_state["prev_alpha"] = learn
    if "prev_batch" not in st.session_state:
        st.session_state["prev_batch"] = batch

    # Stop training if parameters change from previous values
    if (
        learn != st.session_state["prev_alpha"]
        or batch != st.session_state["prev_batch"]
    ):
        st.session_state["running"] = False
        st.session_state["prev_alpha"] = learn
        st.session_state["prev_batch"] = batch
        # Recompute theta_arr with new parameters
        (fit_coefs, df_data, _) = st.session_state["data"]
        pt_idx = create_batched_indices(pts, batch, max_updates, rng)
        st.session_state["data"] = (fit_coefs, df_data, pt_idx)
        st.session_state["theta_arr"] = make_theta_array(
            df_data, pt_idx, learn, batch, init_theta, max_updates, pts
        )
        st.session_state["step"] = 0

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button(
            "Start" if not st.session_state["running"] else "Stop",
            use_container_width=True,
        ):
            st.session_state["running"] = not st.session_state["running"]
            st.rerun()
    with col2:
        if st.button("Reset Training", use_container_width=True):
            update()
            st.session_state["running"] = False
            df["color"] = 0  # Reset colors
            # Recompute theta_arr with current learning rate and batch size
            (fit_coefs, df_data, _) = st.session_state["data"]
            pt_idx = create_batched_indices(pts, batch, max_updates, rng)
            st.session_state["data"] = (fit_coefs, df_data, pt_idx)
            st.session_state["theta_arr"] = make_theta_array(
                df_data, pt_idx, learn, batch, init_theta, max_updates, pts
            )
            st.rerun()
    with col3:
        if st.button("Get New Data", use_container_width=True, on_click=clear_data):
            st.rerun()
    step_counter = col4.empty()
elif st.session_state["lab_part"] == 3:
    st.markdown(
        """
        ## 🗻 Part 3: Scouting the Slope - Loss Landscapes
        On your second run of the bunny hill, you realize that you're late for hot chocolate at the lodge! ☕ Use the interactive 3D visualization to experiment with different learning rates and number of steps to find the fastest (but safest) route down the hill.
        Be careful though - too high of a learning rate might send you flying off the slope!
        """
    )

    # Generate regression data for Part 2
    if "part2_data_3d" not in st.session_state:
        np.random.seed(6)
        X_part2 = np.linspace(0, 10, 5)
        y_part2 = 2.5 * X_part2 + 3.0 + np.random.normal(0, 1, 5)
        # Add bias column
        X_part2_with_bias = np.column_stack((np.ones(len(X_part2)), X_part2))
        st.session_state["part2_data_3d"] = (X_part2_with_bias, y_part2)

    X_part2, y_part2 = st.session_state["part2_data_3d"]

    # Learning rate and batch size controls
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        learning_rate = st.slider(
            "Learning Rate (α)",
            min_value=0.001,
            max_value=0.025,
            step=0.001,
            value=0.001,
            key="lr_3d",
            format="%.4f",
        )
    with col2:
        st.metric("α", f"{learning_rate:.4f}")
    with col3:
        n_steps = st.number_input(
            "Number of steps",
            value=10,
            step=1,
            min_value=1,
            max_value=100,
            key="n_steps_3d",
        )

    # Set default starting position
    w_init = -1.0
    w1_init = -1.0

    # Run optimization
    if st.button("Run Optimization", use_container_width=True, type="primary"):
        initialization = np.array([w_init, w1_init])
        model = GradientDescentLinearRegression(
            learning_rate=learning_rate, max_iterations=n_steps
        ).fit(
            X_part2,
            y_part2,
            initialization=initialization,
            method="standard",
            verbose=False,
        )
        st.session_state["model_3d"] = model

    # Display 3D surface plot
    if "model_3d" in st.session_state:
        model = st.session_state["model_3d"]

        # Calculate final loss to show warnings early
        x_path = np.array(model.w_hist)[:, 0]
        y_path = np.array(model.w_hist)[:, 1]
        z_path = np.array(model.cost_hist)

        # Show warnings above the chart
        if model.learning_rate > 0.007:
            st.error(
                "☠️ You went too fast, and fell off a cliff! See your trajectory below, and try reducing the learning rate to stay on the slope."
            )
        elif z_path[-1] >= 3:
            st.warning(
                "⚠️ You didn't make it all the way to the lodge! Try adjusting the parameters to get a loss below 3!"
            )
        elif z_path[-1] < 1:
            st.success("🍫 Successfully made it to hot chocolate!")

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Loss", f"{z_path[-1]:.4f}")
        with col2:
            st.metric("Steps", len(model.w_hist) - 1)
        with col3:
            st.metric("Final w₀", f"{x_path[-1]:.3f}")
        with col4:
            st.metric("Final w₁", f"{y_path[-1]:.3f}")

    if "model_3d" in st.session_state:
        model = st.session_state["model_3d"]

        # Create loss surface grid
        x_lo, x_hi = -2, 2
        y_lo, y_hi = -1, 3
        x_space = np.linspace(x_lo, x_hi, 50)
        y_space = np.linspace(y_lo, y_hi, 50)
        x_mesh, y_mesh = np.meshgrid(x_space, y_space)
        z = np.zeros(x_mesh.shape)

        def compute_loss(X, y, w):
            y_pred = np.dot(X, w.T)
            loss = (y - y_pred) ** 2
            return np.mean(loss)

        for i in range(len(x_space)):
            for j in range(len(y_space)):
                z[i, j] = compute_loss(
                    X_part2, y_part2, np.array([x_mesh[i, j], y_mesh[i, j]])
                )

        # Create Plotly figure
        fig = go.Figure(
            data=[
                go.Surface(x=x_mesh, y=y_mesh, z=z, colorscale="Viridis", opacity=0.8)
            ]
        )

        # Add optimization path
        x_path = np.array(model.w_hist)[:, 0]
        y_path = np.array(model.w_hist)[:, 1]
        z_path = np.array(model.cost_hist)

        fig.add_trace(
            go.Scatter3d(
                x=x_path,
                y=y_path,
                z=z_path,
                mode="markers+lines",
                marker=dict(size=5, color="red", opacity=1),
                line=dict(color="red", width=3),
                name="",
            )
        )

        # Add step labels
        num_labels = min(len(x_path), 10)  # Limit to 10 labels for clarity
        label_indices = np.linspace(0, len(x_path) - 1, num_labels, dtype=int)

        for idx in label_indices:
            fig.add_trace(
                go.Scatter3d(
                    x=[x_path[idx]],
                    y=[y_path[idx]],
                    z=[z_path[idx]],
                    mode="text",
                    text=[f"Step {idx + 1}"],
                    textposition="top center",
                    textfont=dict(size=15, color="red"),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"Loss Surface",
            scene=dict(
                xaxis_title="w₀ (intercept)",
                yaxis_title="w₁ (slope)",
                zaxis_title="Loss",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            width=1000,
            height=700,
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    step_counter = st.empty()


# Create containers for live updates
title_container = st.empty()
chart_container1 = st.empty()
chart_container2 = st.empty()
info_container = st.empty()
loss_container = st.empty()

step = st.session_state["step"]


def update_charts_part1(step):
    df["color"] = 0
    df.loc[pt_idx[step * batch : (step + 1) * batch], "color"] = 1

    w0, w1 = theta_arr[step]
    chart1b = draw_line((w0, w1), [xmin, xmax], color="#B0B0B0")

    title_container.markdown(
        (
            "<div style='text-align:center; font-size:18px; font-weight:600; margin-bottom:10px;'>"
            "Current path: y = "
            f"<span style='color:#FF9800;'>{w0:.2f}</span>x + "
            f"<span style='color:#B388FF;'>{w1:.2f}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    c0, c1 = chart_container1.columns((7, 5))
    with c0:
        st.altair_chart(alt.layer(chart1b, chart2), use_container_width=True)

    df_theta = pd.DataFrame(theta_arr, columns=["w0", "w1"])
    w0_min = min(df_theta["w0"].min(), fit_1) - 2
    w0_max = max(df_theta["w0"].max(), fit_1) + 2
    w1_min = min(df_theta["w1"].min(), fit_0) - 2
    w1_max = max(df_theta["w1"].max(), fit_0) + 2

    chart_theta = (
        alt.Chart(df_theta.loc[:step])
        .mark_circle(color="#B0B0B0")
        .encode(
            x=alt.X(
                "w0",
                scale=alt.Scale(domain=[w0_min, w0_max]),
                axis=alt.Axis(
                    title="w₀",
                    titleColor="#FF9800",
                    labelColor="#FF9800",
                    tickColor="#FF9800",
                    domainColor="#FF9800",
                ),
            ),
            y=alt.Y(
                "w1",
                scale=alt.Scale(domain=[w1_min, w1_max]),
                axis=alt.Axis(
                    title="w₁",
                    titleColor="#B388FF",
                    labelColor="#B388FF",
                    tickColor="#B388FF",
                    domainColor="#B388FF",
                    titleAngle=0,
                ),
            ),
        )
    )

    fit_point = pd.DataFrame({"w0": [fit_1], "w1": [fit_0]})

    chart_fit_outer = (
        alt.Chart(fit_point)
        .mark_point(
            shape="circle",
            size=180,
            filled=False,
            stroke="#B0B0B0",
            strokeWidth=2,
        )
        .encode(
            x="w0",
            y="w1",
        )
    )

    chart_fit_inner = (
        alt.Chart(fit_point)
        .mark_point(
            shape="circle",
            size=40,
            color="#B0B0B0",
        )
        .encode(
            x="w0",
            y="w1",
        )
    )

    with c1:
        st.altair_chart(
            chart_theta + chart_fit_outer + chart_fit_inner,
            use_container_width=True,
        )

    # Calculate and display loss
    loss_arr = []
    for i, theta in enumerate(theta_arr[: step + 1]):
        w0, w1 = theta
        predictions = w0 * df["x"] + w1
        mse = ((predictions - df["y"]) ** 2).mean()
        loss_arr.append({"step": i, "loss": mse})

    df_loss = pd.DataFrame(loss_arr)
    chart_loss = (
        alt.Chart(df_loss)
        .mark_line(color="#1f77b4", size=2)
        .encode(
            x=alt.X("step:Q", scale=alt.Scale(domain=[0, max_updates])),
            y=alt.Y("loss:Q", scale=alt.Scale(zero=False)),
        )
        .properties(
            title="Loss Over Steps",
            width=400,
            height=250,
        )
    )

    with loss_container:
        st.altair_chart(chart_loss, use_container_width=True)

    step_counter.write(f"Step: {step} / {max_updates}")


def update_charts_part3(step):
    title_container.markdown(
        (
            "<div style='text-align:center; font-size:18px; font-weight:600; margin-bottom:10px;'>"
            "Part 3: Coming soon"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    info_container.info("Part 3 content will go here.")


### Define Headers / Update Functions By Part of the Lab ###
if st.session_state["lab_part"] == 1:
    update_fn = lambda x: None  # No animation for Part 1
elif st.session_state["lab_part"] == 2:
    update_fn = update_charts_part1
elif st.session_state["lab_part"] == 3:
    if "model_3d" not in st.session_state:
        st.info("Run the optimization above to see the loss landscape.")

    update_fn = lambda x: None  # Placeholder
elif st.session_state["lab_part"] == 4:
    st.markdown(
        """
        ## ◆ Part 4: Black Diamond - Advanced Terrain
        You've graduated from the bunny hill! This advanced slope has a complex terrain with multiple peaks and valleys. Your goal is navigate this challenging landscape using gradient descent to find the lodge at the bottom (global minimum).
        Unlike the smooth beginner slope, this terrain is tricky — you might get stuck in the wrong valley or have trouble finding your way down at all.
        """
    )

    # Initialize Part 3 state
    if "part3_data" not in st.session_state:
        st.session_state["part3_data"] = {
            "learning_rate": 0.016,
            "w1_init": 1.5,
            "w2_init": 1.0,
            "n_steps": 50,
            "path": None,
            "model": None,
        }

    part3 = st.session_state["part3_data"]

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        lr_3d = st.slider(
            "Learning Rate (α)",
            min_value=0.001,
            max_value=0.05,
            step=0.005,
            value=part3["learning_rate"],
            key="lr_dw",
            format="%.4f",
        )
        part3["learning_rate"] = lr_3d

    with col2:
        n_steps_dw = st.number_input(
            "Number of steps",
            value=part3["n_steps"],
            step=1,
            min_value=1,
            max_value=200,
            key="n_steps_dw",
        )
        part3["n_steps"] = n_steps_dw

    # Starting point
    st.write("### Starting Point")
    col1, col2 = st.columns(2)
    with col1:
        w1_start = st.number_input(
            "Start w₁", value=part3["w1_init"], step=0.5, key="w1_start_dw"
        )
        part3["w1_init"] = w1_start
    with col2:
        w2_start = st.number_input(
            "Start w₂", value=part3["w2_init"], step=0.5, key="w2_start_dw"
        )
        part3["w2_init"] = w2_start

    # Run optimization
    if st.button("Run Optimization", use_container_width=True, type="primary"):
        w1, w2 = w1_start, w2_start
        path = [(w1, w2)]
        losses = [doublewell_loss(w1, w2)]

        for step in range(n_steps_dw):
            grad = doublewell_gradient(w1, w2)
            w1 -= lr_3d * grad[0]
            w2 -= lr_3d * grad[1]
            path.append((w1, w2))
            losses.append(doublewell_loss(w1, w2))

        part3["path"] = path
        part3["losses"] = losses

    # Visualize 3D surface
    if part3["path"] is not None:
        path = part3["path"]
        losses = part3["losses"]

        # Create loss surface grid
        w1_range = np.linspace(-2, 2, 60)
        w2_range = np.linspace(-2, 2, 60)
        W1_grid, W2_grid = np.meshgrid(w1_range, w2_range)
        Z_grid = np.zeros_like(W1_grid)

        for i in range(len(w1_range)):
            for j in range(len(w2_range)):
                Z_grid[j, i] = doublewell_loss(W1_grid[j, i], W2_grid[j, i])

        # Create Plotly figure
        fig = go.Figure(
            data=[
                go.Surface(
                    x=W1_grid, y=W2_grid, z=Z_grid, colorscale="Viridis", opacity=0.8
                )
            ]
        )

        # Extract path coordinates
        path_array = np.array(path)
        w1_path = path_array[:, 0]
        w2_path = path_array[:, 1]
        z_path = np.array(losses)

        # Add optimization path
        fig.add_trace(
            go.Scatter3d(
                x=w1_path,
                y=w2_path,
                z=z_path,
                mode="markers+lines",
                marker=dict(size=4, color="red", opacity=1),
                line=dict(color="red", width=2),
                name="Path",
            )
        )

        # Add end point
        fig.add_trace(
            go.Scatter3d(
                x=[w1_path[-1]],
                y=[w2_path[-1]],
                z=[z_path[-1]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="diamond",
                    line=dict(color="red", width=2),
                ),
                name="End",
            )
        )

        # Add step labels
        num_labels = min(len(w1_path), 8)
        label_indices = np.linspace(0, len(w1_path) - 1, num_labels, dtype=int)

        for idx in label_indices:
            fig.add_trace(
                go.Scatter3d(
                    x=[w1_path[idx]],
                    y=[w2_path[idx]],
                    z=[z_path[idx]],
                    mode="text",
                    text=[f"Step {idx + 1}"],
                    textposition="top center",
                    textfont=dict(size=10, color="red"),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"Sine Landscape - Learning Rate: {lr_3d:.4f}",
            scene=dict(
                xaxis_title="w₁",
                yaxis_title="w₂",
                zaxis_title="Loss",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            width=1000,
            height=700,
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Loss", f"{losses[-1]:.4f}")
        with col2:
            st.metric("Final w₀", f"{w1_path[-1]:.3f}")
        with col3:
            st.metric("Final w₁", f"{w2_path[-1]:.3f}")

    update_fn = lambda x: None  # Placeholder
elif st.session_state["lab_part"] == 5:
    update_fn = lambda x: None  # Placeholder for bonus challenge
else:
    update_fn = lambda x: None  # Placeholder for unknown parts

# Initial chart draw
update_fn(step)

# ===== Exercises for Each Part =====
if st.session_state["lab_part"] == 2:
    st.subheader("Questions")

    if not st.session_state.get("part2_questions_submitted", False):
        restore_form_keys(["q1", "q2", "q3_text", "q4", "q5", "q6"])
    with st.form("part2_questions"):
        q1 = st.text_area(
            "1) Set the learning rate to 0. How does this affect the optimization?",
            key="q1",
            height=80,
        )

        q2 = st.text_area(
            "2) Now set the learning rate to 0.001, then 0.005. How does increasing the learning rate affect the optimization?",
            key="q2",
            height=80,
        )

        q3_text = st.text_area(
            "3) Now put the learning rate to the maximum. What happens to the optimization?",
            key="q3_text",
            height=80,
        )

        q4 = st.radio(
            "4) To speed up training, would you increase or decrease the learning rate?",
            ["Increase learning rate", "Decrease learning rate"],
            key="q4",
            index=None,
        )

        q5 = st.radio(
            "5) To speed up training, would you increase or decrease the batch size?",
            ["Increase batch size", "Decrease batch size"],
            key="q5",
            index=None,
        )

        q6 = st.text_area(
            "6) Reflection: What trade-offs would you need to consider when adjusting these hyperparameters?",
            key="q6",
            height=100,
        )

        submitted = st.form_submit_button(
            "Submit answers",
            on_click=mark_submitted,
            args=("part2_questions_submitted",),
        )

        if submitted:
            stash_form_keys(["q1", "q2", "q3_text", "q4", "q5", "q6"])
            # Check that at least some answers were provided
            if (
                q1.strip()
                and q2.strip()
                and q3_text.strip()
                and q4
                and q5
                and q6.strip()
            ):
                correct_q4 = q4 == "Increase learning rate"
                correct_q5 = q5 == "Increase batch size"

                if correct_q4 and correct_q5:
                    st.session_state["part2_completed"] = True
                    st.success("Great job! Part 3 unlocked.")
                    st.info(
                        "**Key Insights:**\n\n"
                        "- **Learning Rate = 0**: No updates occur—the parameters stay at initialization. The line doesn't move!\n"
                        "- **Increasing Learning Rate**: Larger steps toward the minimum. Higher learning rates converge faster but risk overshooting or instability.\n"
                        "- **Maximum Learning Rate**: Can cause divergence—the parameters oscillate wildly or explode instead of converging.\n"
                        "- **Batch Size Trade-offs**: Larger batches → more stable gradients but slower updates per epoch. "
                        "Smaller batches → noisier gradients but more frequent updates, which can help escape shallow local minima.\n"
                        "- **The Balance**: The best hyperparameters balance convergence speed, stability, and computational efficiency!"
                    )
                else:
                    st.session_state["part2_completed"] = False
                    error_msg = "Not quite right. "
                    if not correct_q4:
                        error_msg += "Check question 4 (learning rate). "
                    if not correct_q5:
                        error_msg += "Check question 5 (batch size). "
                    st.error(error_msg)
            else:
                st.session_state["part2_completed"] = False
                st.error("Please answer all questions before submitting.")

    st.session_state["part2_questions_submitted"] = False

    # Navigation button for Part 1
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "← Back to Part 1", use_container_width=True, key="back_to_part0_p1"
        ):
            st.session_state["lab_part"] = 1
            st.rerun()
    with col2:
        if st.session_state.get("part2_completed", False):
            if st.button(
                "Continue to Part 3 →",
                use_container_width=True,
                key="continue_to_part3",
            ):
                st.session_state["lab_part"] = 3
                st.rerun()

elif st.session_state["lab_part"] == 3:
    st.subheader("Questions")

    if not st.session_state.get("part3_questions_submitted", False):
        restore_form_keys(["q1_p2", "q2_p2", "q3_p2"])
    with st.form("part3_questions"):
        q1_p2 = st.radio(
            "Q1: As you increase the learning rate, what happens to the path on the loss surface?",
            ["Steps get larger", "Steps get smaller", "The path stays the same"],
            key="q1_p2",
            index=None,
        )
        q2_p2 = st.radio(
            "Q2: The optimization path follows which direction at each step?",
            [
                "A random direction",
                "The direction of steepest descent (negative gradient)",
                "The direction that increases loss",
                "The direction of steepest descent (positive gradient)",
            ],
            key="q2_p2",
            index=None,
        )
        q3_p2 = st.radio(
            "Q3: If the learning rate is too high, what can happen?",
            [
                "It will converge faster, but follow the same path",
                "The optimization might diverge and never reach the minimum",
                "It has no effect on convergence path",
                "The loss will plataeu",
            ],
            key="q3_p2",
            index=None,
        )

        submitted = st.form_submit_button(
            "Submit answers",
            on_click=mark_submitted,
            args=("part3_questions_submitted",),
        )

        if submitted:
            stash_form_keys(["q1_p2", "q2_p2", "q3_p2"])
            # Check if model has been run
            if "model_3d" not in st.session_state:
                st.session_state["part3_completed"] = False
                st.error("Please run the optimization first before submitting answers!")
            else:
                model = st.session_state["model_3d"]
                final_loss = model.cost_hist[-1]

                correct_answers = (
                    q1_p2 == "Steps get larger"
                    and q2_p2 == "The direction of steepest descent (negative gradient)"
                    and q3_p2
                    == "The optimization might diverge and never reach the minimum"
                )

                loss_below_3 = final_loss < 3

                if correct_answers and loss_below_3:
                    st.session_state["part3_completed"] = True
                    st.success("Great job! Part 4 unlocked.")
                elif correct_answers and not loss_below_3:
                    st.session_state["part3_completed"] = False
                    st.warning(
                        f"Your answers are correct, but your final loss is {final_loss:.4f}. "
                        "Try adjusting the learning rate or number of steps to get a loss below 3!"
                    )
                else:
                    st.session_state["part3_completed"] = False
                    st.error(
                        "Not quite. Try different learning rates and observe the patterns."
                    )

    st.session_state["part3_questions_submitted"] = False

    # Navigation buttons for Part 2
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "← Back to Part 2", use_container_width=True, key="back_to_part1_p2"
        ):
            st.session_state["lab_part"] = 2
            st.rerun()
    with col2:
        if st.session_state.get("part3_completed", False):
            if st.button(
                "Continue to Part 4 →",
                use_container_width=True,
                key="continue_to_part4",
            ):
                st.session_state["lab_part"] = 4
                st.rerun()

elif st.session_state["lab_part"] == 4:
    st.subheader("Questions")

    if not st.session_state.get("part4_questions_submitted", False):
        restore_form_keys(["q1_p4", "q2_p4"])
    with st.form("part4_questions"):
        q1_p4 = st.radio(
            "Q1: Keeping the initial position the same (1.5, 1), change the learning rate. Are you able to get to the true global minimum with the given range of learning rates?",
            [
                "Yes",
                "No",
            ],
            key="q1_p4",
            index=None,
        )

        q2_p4 = st.text_area(
            "Q2: Reflection: When can the optimizer always find the global minimum? When might it get stuck in local minima or fail to converge?",
            key="q2_p4",
            height=100,
        )

        if q2_p4.strip():
            st.info(
                "**Key insights:** Gradient descent can reliably find the global minimum on **convex** loss surfaces "
                "(like the bunny hill in Part 2). However, on **non-convex** landscapes with multiple local minima "
                "(like this sine wave terrain), the optimizer typically gets stuck in whichever local minimum is closest "
                "to the starting position. The learning rate affects convergence speed and stability, but rarely helps "
                "escape from a local minimum basin once trapped. This is why defining the loss function is so important!"
            )

        submitted = st.form_submit_button(
            "Submit answers",
            on_click=mark_submitted,
            args=("part4_questions_submitted",),
        )

        if submitted:
            stash_form_keys(["q1_p4", "q2_p4"])
            # Check if both questions are answered
            correct_q1 = q1_p4 == "No"
            correct_q2 = q2_p4.strip() != ""

            if correct_q1 and correct_q2:
                st.session_state["part4_completed"] = True
                st.success("Great work! Part 5 unlocked.")
            else:
                st.session_state["part4_completed"] = False
                if not correct_q1:
                    error_msg = "Not quite!"
                elif not correct_q2:
                    error_msg = "Question 2 requires a reflection."
                else:
                    error_msg = "Please answer all questions: "
                st.error(error_msg)

    st.session_state["part4_questions_submitted"] = False

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "← Back to Part 3", use_container_width=True, key="back_to_part2_p3"
        ):
            st.session_state["lab_part"] = 3
            st.rerun()
    with col2:
        if st.session_state.get("part4_completed", False):
            if st.button("Part 5 →", use_container_width=True, key="continue_to_part5"):
                st.session_state["lab_part"] = 5
                st.rerun()

elif st.session_state["lab_part"] == 5:
    st.markdown(
        """
    ## 🎿 Part 5: Avalanche Control - Taming Exploding Gradients

    You're training a neural network on a challenging loss landscape. Without gradient clipping,
    the gradients can explode, triggering an avalanche!

    **Your challenge:** Add one line of code needed to prevent the explosion. Below is the original training loop that is at risk for exploding gradients.
    """
    )

    st.markdown("### The Problem")
    buggy_code = """def train_without_clipping(model, x, y, learning_rate=0.01, n_steps=50):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Gradients can EXPLODE! 💥
    for step in range(n_steps):
        optimizer.zero_grad()  # Clear previous gradients
        loss = compute_loss(model, x, y)  # L = (pred - y)^4, compute loss
        loss.backward()  # Compute gradients via backpropagation
        optimizer.step()  # Update model parameters based on gradients

        if loss.item() > 1e6:
            print(f"AVALANCHE! Training exploded at step {step}!")
            break"""

    st.code(buggy_code, language="python")

    st.markdown("### The Solution Template")
    solution_code = """def train_with_clipping(model, x, y, learning_rate=0.01, n_steps=50, max_norm=5.0):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for step in range(n_steps):
        optimizer.zero_grad()
        #Location A
        loss = compute_loss(model, x, y)
        #Location B
        loss.backward()
        #Location C
        optimizer.step()
        #Location D

        if loss.item() > 1e6:
            print(f"AVALANCHE! Training exploded at step {step}!")
        """

    st.code(solution_code, language="python")

    # Simulate training with and without clipping
    clipping_results = simulate_gradient_clipping(
        n_steps_demo=10, learning_rate=0.01, max_norm=5.0
    )
    losses_no_clip_truncated = clipping_results["losses_no_clip_truncated"]
    losses_with_clip_truncated = clipping_results["losses_with_clip_truncated"]

    if not st.session_state.get("part5_code_submitted", False):
        restore_form_keys(["q5_part5"])
    with st.form("part5_code_submission"):
        q5_answer = st.text_area(
            "What ONE line of code would you add to prevent exploding gradients? (Write just the code line, no comments)",
            key="q5_part5",
            height=60,
            placeholder="Hint: It's a function in torch.nn.utils that prevents gradients from exceeding max_norm",
        )

        submitted_code = st.form_submit_button(
            "Submit Code Change",
            type="primary",
            on_click=mark_submitted,
            args=("part5_code_submitted",),
        )

        if submitted_code:
            stash_form_keys(["q5_part5"])
            if q5_answer.strip():
                # Normalize the answer for checking (remove ALL spaces/tabs/newlines, make lowercase)
                answer_normalized = (
                    q5_answer.replace(" ", "")
                    .replace("\t", "")
                    .replace("\n", "")
                    .lower()
                )

                # Check for various acceptable forms of the gradient clipping line
                # Since we strip all whitespace from user input, we define patterns without spaces
                acceptable_patterns = [
                    "torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm)",
                    "torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)",
                    "torch.nn.utils.clip_grad_norm_(model.parameters(),5)",
                    "torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)",
                    "torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)",
                    "torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_norm)",
                ]

                # Ensure patterns are also normalized (just in case)
                normalized_patterns = [
                    p.replace(" ", "").lower() for p in acceptable_patterns
                ]

                if answer_normalized in normalized_patterns:
                    st.session_state["part5_code_correct"] = True
                    st.success("🎉 You've got it!")
                else:
                    st.session_state["part5_code_correct"] = False
                    # Check what's wrong
                    if (
                        "torch.nn.utils" not in answer_normalized
                        and "torch" in answer_normalized
                    ):
                        st.error(
                            "❌ Almost! Make sure to use the full path: `torch.nn.utils....`"
                        )
                    elif "clip_grad_norm_" not in answer_normalized:
                        st.error("❌ Make sure to use the correct function name!")
                    elif "model.parameters()" not in answer_normalized:
                        st.error("❌ Make sure to use `model.parameters()`")
                    elif "max_norm" not in answer_normalized:
                        st.error("❌ Make sure to use `max_norm`!")
                    else:
                        st.info(
                            "❌ Not quite right! Double check your syntax and try again. "
                        )
            else:
                st.session_state["part5_code_correct"] = False
                st.error("Please enter your solution.")

    st.session_state["part5_code_submitted"] = False

    if not st.session_state.get("part5_location_submitted", False):
        restore_form_keys(["clipping_location"])
    with st.form("gradient_clipping_location"):
        location_q = st.radio(
            "Which location should this line be placed? Look carefully through the comments.",
            [
                "Location A",
                "Location B",
                "Location C",
                "Location D",
            ],
            key="clipping_location",
        )

        submitted_location = st.form_submit_button(
            "Submit Location",
            type="primary",
            on_click=mark_submitted,
            args=("part5_location_submitted",),
        )

        if submitted_location:
            stash_form_keys(["clipping_location"])
            correct_location = location_q == "Location C"
            if correct_location:
                st.success(
                    "✅ Correct! Gradient clipping must happen after computing gradients but before updating parameters."
                )
            else:
                st.info(
                    "❌ Not quite. Walk through the code - when are the gradients computed? When are the parameters updated? "
                )

    st.session_state["part5_location_submitted"] = False

    st.markdown("### Visualizing the Impact")
    st.markdown("See what happens with and without gradient clipping:")

    # Create side-by-side plots
    col1, col2 = st.columns(2)

    # Left plot: Without Clipping
    df_no_clip = pd.DataFrame(
        {
            "Step": list(range(len(losses_no_clip_truncated))),
            "Loss": losses_no_clip_truncated,
        }
    )

    chart_no_clip = (
        alt.Chart(df_no_clip)
        .mark_line(strokeWidth=3, color="#FF7F0E")
        .encode(
            x=alt.X("Step:Q", title="Training Step"),
            y=alt.Y("Loss:Q", title="Loss", scale=alt.Scale(type="linear")),
            tooltip=["Step:Q", "Loss:Q"],
        )
        .properties(height=300, title="❌ Without Gradient Clipping (Exploding)")
    )

    with col1:
        st.altair_chart(chart_no_clip, use_container_width=True)

    # Right plot: With Clipping
    df_with_clip = pd.DataFrame(
        {
            "Step": list(range(len(losses_with_clip_truncated))),
            "Loss": losses_with_clip_truncated,
        }
    )

    chart_with_clip = (
        alt.Chart(df_with_clip)
        .mark_line(strokeWidth=3, color="#1F77B4")
        .encode(
            x=alt.X("Step:Q", title="Training Step"),
            y=alt.Y("Loss:Q", title="Loss", scale=alt.Scale(type="linear")),
            tooltip=["Step:Q", "Loss:Q"],
        )
        .properties(height=300, title="✓ With Gradient Clipping (Stable)")
    )

    with col2:
        st.altair_chart(chart_with_clip, use_container_width=True)

    st.markdown("### Questions")

    if not st.session_state.get("part5_questions_submitted", False):
        restore_form_keys(["q5_max_norm", "q5_reflection"])
    with st.form("part5_questions"):
        q5_max_norm = st.text_input(
            "1) What is the maximum a gradient can be with this clipping?",
            key="q5_max_norm",
            placeholder="Hint: Look at the max_norm variable",
        )

        q5_reflection = st.text_area(
            "2) What do you think would happen if max_norm were set to 0.000001?",
            key="q5_reflection",
            height=80,
            placeholder="Think about how this would affect the training speed and stability...",
        )

        submitted = st.form_submit_button(
            "Check Answers",
            type="primary",
            on_click=mark_submitted,
            args=("part5_questions_submitted",),
        )

        if submitted:
            stash_form_keys(["q5_max_norm", "q5_reflection"])
            # First check if the code was submitted correctly
            if not st.session_state.get("part5_code_correct", False):
                st.error(
                    "⚠️ Please complete the gradient clipping code above before answering these questions!"
                )
            else:
                # Check q5_max_norm - should be max_norm, 1, 1.0, or 5
                answer_norm = (
                    q5_max_norm.strip().lower().replace(" ", "").replace("_", "")
                )
                correct_norm = any(variant in answer_norm for variant in ["5", "5.0"])

                # Check q5_reflection - just needs to have some content
                correct_reflection = q5_reflection.strip() != ""

                if correct_norm and correct_reflection:
                    st.balloons()
                    st.success(
                        "🎉 Excellent! You've mastered gradient clipping! Take a screenshot of the following success message and upload it to the lab campuswire post (Search 'Lab 2') "
                    )
                    st.markdown(
                        """
                    ## 🏔️ Ski Course Complete! 🏔️

                    You've successfully navigated the entire optimization adventure:
                    - ✅ **Part 1:** Understood gradients and descent direction
                    - ✅ **Part 2:** Experimented with learning rates and batch sizes
                    - ✅ **Part 3:** Explored 3D loss landscapes
                    - ✅ **Part 4:** Tackled complex terrains with local minima
                    - ✅ **Part 5:** Prevented gradient explosions with clipping!

                    **Key Lessons:**
                    1. Gradients point in the direction of steepest increase; we move opposite to minimize loss
                    2. Learning rate controls step size; too large causes divergence, too small slows convergence
                    3. Loss landscapes can have multiple local minima—starting position and hyperparameters matter
                    4. In deep networks, gradients can explode through backpropagation
                    5. **Gradient clipping** bounds gradient norms to prevent explosions

                    You're now ready to train your own neural networks! 🚀
                    """
                    )
                else:
                    if not correct_norm:
                        st.error(
                            "❌ First question hint: look at the value of max_norm in the code snippet."
                        )
                    if not correct_reflection:
                        st.error("❌ Missing reflection!")

    st.session_state["part5_questions_submitted"] = False

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Part 4", use_container_width=True, key="back_to_part4"):
            st.session_state["lab_part"] = 4
            st.rerun()
    with col2:
        st.write("")

else:
    for key in [
        "q0_1",
        "q0_2",
        "q0_3a",
        "q0_3b",
        "q0_6",
        "submit_part0",
        "q1",
        "q2",
        "q3",
        "q4",
        "submit_part1",
        "q1_p2",
        "q2_p2",
        "q3_p2",
        "submit_part2",
        "q1_p3",
        "q2_p3",
        "submit_part3",
        "q5_part5",
        "submit_part4",
    ]:
        if key in st.session_state:
            del st.session_state[key]

# ===== Animation Loop =====
while st.session_state["running"] and st.session_state["step"] < max_updates:
    st.session_state["step"] = min(st.session_state["step"] + 1, max_updates)
    step = st.session_state["step"]
    update_fn(step)

    time.sleep(st.session_state["animation_speed"])
