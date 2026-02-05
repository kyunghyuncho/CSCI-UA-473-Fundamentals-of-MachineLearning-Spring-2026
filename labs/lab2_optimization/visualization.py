"""
Altair chart utilities for Lab 2: Optimization
"""

import altair as alt
import pandas as pd


def draw_line(coefs, dom, color="lightgrey"):
    """
    Create an Altair line chart from coefficients.

    Args:
        coefs: tuple (t0, t1) where line is y = t0*x + t1
        dom: tuple (xmin, xmax) for x domain
        color: line color

    Returns:
        Altair Chart object
    """
    from labs.lab2_optimization.gradient_descent import line_fn

    f = line_fn(coefs)
    line_dom = [dom[0] - 5, dom[1] + 5]
    df = pd.DataFrame({"x": line_dom, "y": [f(x) for x in line_dom]})
    c = (
        alt.Chart(df)
        .mark_line(clip=True, color=color)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=dom)),
            y="y",
        )
    )
    return c
