import numpy as np
import plotly.graph_objects as go


def plot_surface_v2(X, y, model, path, params):
    """
    Interactive 3D plot of the cost function using Plotly.
    """

    n_steps = params["steps"]
    fig_title = params["title"]
    annotation_steps = params["annotation_steps"]
    x_lo, x_hi = -2, 2

    if model.learning_rate == 0.08:
        y_lo, y_hi = -13, 13
    else:
        y_lo, y_hi = -0.05, 0.84 * 2

    x_space = np.linspace(x_lo, x_hi, 100)
    y_space = np.linspace(y_lo, y_hi, 100)
    x_mesh, y_mesh = np.meshgrid(x_space, y_space)
    z = np.zeros(x_mesh.shape)

    all_w = np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=-1)
    y_pred = np.dot(X, all_w.T)
    loss = (y.reshape(-1, 1) - y_pred) ** 2
    z_flat = np.mean(loss, axis=0)
    z = z_flat.reshape(x_mesh.shape)

    fig = go.Figure(
        data=[go.Surface(x=x_mesh, y=y_mesh, z=z, colorscale="Viridis", opacity=0.8)]
    )

    x_path = np.array(model.w_hist)[:, 0]
    y_path = np.array(model.w_hist)[:, 1]
    z_path = np.array(model.cost_hist)

    fig.add_trace(
        go.Scatter3d(
            x=x_path[:n_steps],
            y=y_path[:n_steps],
            z=z_path[:n_steps],
            mode="markers+lines",
            marker=dict(size=6, color="red", opacity=1),
            line=dict(color="red", width=4),
            name="Optimization Path",
        )
    )

    for i in range(min(annotation_steps, n_steps)):
        fig.add_trace(
            go.Scatter3d(
                x=[x_path[i]],
                y=[y_path[i]],
                z=[z_path[i]],
                mode="text",
                text=[f"w^({i})"],
                textposition="top center",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=fig_title,
        scene=dict(
            xaxis_title="w_0",
            yaxis_title="w_1",
            zaxis_title="J(w)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        width=1000,
        height=700,
        showlegend=True,
    )

    fig.show()


"""
Author: Dmitrijs Kass.
"""


class GradientDescentLinearRegression:
    """
    Linear Regression with gradient-based optimization.
    Parameters
    ----------
    learning_rate : float
        Learning rate for the gradient descent algorithm.
    max_iterations : int
        Maximum number of iteration for the gradient descent algorithm.
    eps : float
        Tolerance level for the Euclidean norm between model parameters in two
        consequitive iterations. The algorithm is stopped when the norm becomes
        less than the tolerance level.
    """

    def __init__(self, learning_rate=0.01, max_iterations=100000, eps=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.eps = eps

    def predict(self, X):
        """Returns predictions array of shape [n_samples,1]"""
        return np.dot(X, self.w.T)

    def cost(self, X, y):
        """Returns the value of the cost function as a scalar real number"""
        y_pred = self.predict(X)
        loss = (y - y_pred) ** 2
        return np.mean(loss)

    def grad(self, X, y):
        """Returns the gradient vector"""
        y_pred = self.predict(X)
        d_intercept = -2 * sum(y - y_pred)  # dJ/d w_0.
        d_x = -2 * sum(X[:, 1:] * (y - y_pred).reshape(-1, 1))  # dJ/d w_i.
        g = np.append(np.array(d_intercept), d_x)  # Gradient.
        return g / X.shape[0]  # Average over training samples.

    def adagrad(self, g):
        self.G += g**2  # Update cache.
        step = self.learning_rate / (np.sqrt(self.G + self.eps)) * g
        return step

    def fit(self, X, y, initialization=None, method="standard", verbose=True):
        """
        Fit linear model with gradient descent.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_predictors]
            Training data
        y : numpy array of shape [n_samples,1]
            Target values.
        method : string
                 Defines the variant of gradient descent to use.
                 Possible values: "standard", "adagrad".
        verbose: boolean
                 If True, print the gradient, parameters and the cost function
                 for each iteration.

        Returns
        -------
        self : returns an instance of self.
        """
        if initialization is None:
            self.w = np.zeros(X.shape[1])  # Initialization of params.
            print("Parameters initialized to zeros.")
            print(self.w, self.w.shape)
        else:
            self.w = initialization
        if method == "adagrad":
            self.G = np.zeros(X.shape[1])  # Initialization of cache for AdaGrad.
        w_hist = [self.w]  # History of params.
        cost_hist = [self.cost(X, y)]  # History of cost.

        for it in range(self.max_iterations):
            g = self.grad(X, y)  # Calculate the gradient.
            if method == "standard":
                step = self.learning_rate * g  # Calculate standard gradient step.
            elif method == "adagrad":
                step = self.adagrad(g)  # Calculate AdaGrad step.
            else:
                raise ValueError("Method not supported.")
            self.w = self.w - step  # Update parameters.
            w_hist.append(self.w)  # Save to history.

            J = self.cost(X, y)  # Calculate the cost.
            cost_hist.append(J)  # Save to history.

            if verbose:
                print(f"Iter: {it}, gradient: {g}, params: {self.w}, cost: {J}")

            # Stop if update is small enough.
            if np.linalg.norm(w_hist[-1] - w_hist[-2]) < self.eps:
                break

        # Final updates before finishing.
        self.iterations = it + 1  # Due to zero-based indexing.
        self.w_hist = w_hist
        self.cost_hist = cost_hist
        self.method = method

        return self


# Simple alternative to adaptive alpha is self.learning_rate = 1 / (iter + self.eps)


def generate_data(n_predictors=1, n_samples=5, location=1, scale=3):
    """Generate data for the linear regression"""

    # Reproducibility.
    np.random.seed(6)
    # True parameters, +1 for the intercept.
    w_star = np.random.randn(n_predictors + 1)
    X = np.random.normal(loc=location, scale=scale, size=(n_samples, n_predictors))
    # Add a column of ones for an intercept.
    X = np.column_stack((np.ones(n_samples), X))
    noise = np.random.randn(n_samples)
    # Compute output variable.
    y = np.dot(X, w_star.T) + noise

    return X, y


if __name__ == "__main__":
    LR = 0.05
    MAX_ITER = 20
    METHOD = "standard"  # "adagrad"
    path = "./"
    a, b = -2, 0
    initialization = np.array([a, b])

    X, y = generate_data()

    model = GradientDescentLinearRegression(LR, MAX_ITER).fit(
        X, y, initialization, METHOD
    )
    print(f"Gradient descent solution in {model.iterations} iterations: {model.w}.")

    w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"Least squares solutions: {w_lstsq}.")

    if X.shape[1] == 2:
        # Generate plots only for a two-parameter problem.

        # Surface plot -------------------------------------------------------

        surf_dict = {
            ("standard", 0.0021): {
                "steps": len(model.w_hist),
                "annotation_steps": 5,
                "title": f"Path of the converging gradient descent with $\\alpha$={LR}",
            },
            ("standard", 0.0125): {
                "steps": len(model.w_hist),
                "annotation_steps": 5,
                "title": f"Path of the converging gradient descent with $\\alpha$={LR}",
            },
            ("standard", 0.025): {
                "steps": len(model.w_hist),
                "annotation_steps": 5,
                "title": f"Path of the converging gradient descent with $\\alpha$={LR}",
            },
            ("standard", 0.028): {
                "steps": 7,
                "annotation_steps": 7,
                "title": f"Path of the diverging gradient descent with $\\alpha$={LR}",
            },
            ("adagrad", 12): {
                "steps": len(model.w_hist),
                "annotation_steps": 5,
                "title": f"AdaGrad with $\\eta$={LR}",
            },
        }

        surf_dict_default = {"steps": 15, "annotation_steps": 15, "title": ""}

        plot_surface_v2(X, y, model, path, params=surf_dict_default)
