"""Gradient clipping simulations for understanding exploding gradients in RNNs."""

import numpy as np


def simulate_gradient_clipping(n_steps_demo=10, learning_rate=0.01, max_norm=5.0):
    """
    Simulate training with and without gradient clipping.

    This mimics an RNN scenario where gradients accumulate through backpropagation
    over multiple timesteps, causing potential explosion.

    Parameters
    ----------
    n_steps_demo : int, default=10
        Number of training steps to simulate
    learning_rate : float, default=0.01
        Learning rate for gradient updates
    max_norm : float, default=5.0
        Maximum gradient norm for clipping

    Returns
    -------
    dict
        Dictionary containing:
        - 'losses_no_clip': list of losses without clipping
        - 'losses_with_clip': list of losses with clipping
        - 'losses_no_clip_truncated': truncated losses (up to explosion point)
        - 'losses_with_clip_truncated': truncated clipped losses
        - 'explode_step': step index where explosion occurred (or None)
    """
    np.random.seed(42)

    losses_no_clip = []
    losses_with_clip = []
    explode_step = None

    # Simulate training without clipping
    # Model: h_{t+1} = 1.2 * h_t, loss = sum(h_t^2)
    # Gradient w.r.t w accumulates through time: gets multiplied by 1.2 each step
    w = 0.5  # Weight that will be multiplied through timesteps
    for step in range(n_steps_demo):
        # Simulate hidden state growing through time (like RNN backprop)
        h = w  # Initial hidden state
        loss = 0

        # Forward pass through 15 timesteps
        num_timesteps = 15
        for t in range(num_timesteps):
            h = 1.1 * h  # Each step multiplies by 1.1 (weight > 1 causes explosion)
            loss += h**2

        # Compute accumulated gradient (simplified: proportional to 1.2^timesteps)
        grad = loss * 5  # Amplify gradient to show effect

        # Update without clipping
        w_temp = w - learning_rate * grad
        if abs(w_temp) > 100:  # Exploded
            w_temp = 100
            if explode_step is None:
                explode_step = step

        w = w_temp
        losses_no_clip.append(loss)

    # Simulate training with clipping
    w = 0.5  # Start from same position
    for step in range(n_steps_demo):
        # Same forward pass
        h = w
        loss = 0

        num_timesteps = 15
        for t in range(num_timesteps):
            h = 1.2 * h
            loss += h**2

        # Compute gradient
        grad = loss * 5
        grad_norm = abs(grad)

        # Clip gradient
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)

        # Update with clipped gradient
        w -= learning_rate * grad
        losses_with_clip.append(loss)

    # Truncate no_clip at explosion point
    truncate_at = (
        min(explode_step + 3, len(losses_no_clip))
        if explode_step
        else len(losses_no_clip)
    )
    losses_no_clip_truncated = losses_no_clip[:truncate_at]
    losses_with_clip_truncated = losses_with_clip[:truncate_at]

    return {
        "losses_no_clip": losses_no_clip,
        "losses_with_clip": losses_with_clip,
        "losses_no_clip_truncated": losses_no_clip_truncated,
        "losses_with_clip_truncated": losses_with_clip_truncated,
        "explode_step": explode_step,
    }
