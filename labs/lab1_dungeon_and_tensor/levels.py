import streamlit as st
import torch

# Re-import checks to ensure they have access to torch/np if needed, though they are passed locals.
# But `check_level_0` uses `st.session_state`.


def check_level_0(local_vars):
    """Level 0: Map Generation (Random Variables)"""
    if "dungeon_map" not in local_vars:
        return False, "Variable 'dungeon_map' not found!"

    t = local_vars["dungeon_map"]
    if not isinstance(t, torch.Tensor):
        return False, "'dungeon_map' is not a torch.Tensor!"

    if t.shape != (10, 10):
        return False, f"WRONG SHAPE! Expected (10, 10), got {t.shape}"

    unique_vals = torch.unique(t)
    if len(unique_vals) < 2:
        return (
            False,
            f"The map looks too uniform. Did {st.session_state.player_name} use randomness?",
        )

    st.session_state.dungeon_map = t
    st.session_state.merchant_count = (t >= 98).sum().item()
    return (
        True,
        f"The World Forge rumbles. A new dungeon is born. {st.session_state.merchant_count} Merchants hidden in the shadows.",
    )


def check_level_1(local_vars):
    """Level 1: Initialization"""
    if "tensor_x" not in local_vars:
        return False, "Variable 'tensor_x' not found!"
    t = local_vars["tensor_x"]
    if not isinstance(t, torch.Tensor):
        return False, "Not a Tensor!"

    # Save for visualization
    st.session_state.tensor_x = t

    if t.shape != (5, 3):
        return False, f"WRONG SHAPE! Expected (5, 3), got {t.shape}"
    if not torch.all(t == 1):
        return False, "Values incorrect (should be all 1s)!"
    return True, f"Perfect! {st.session_state.player_name} summoned the Monolith."


def check_level_2(local_vars):
    """Level 2: Attributes"""
    if "shape_x" not in local_vars or "dtype_x" not in local_vars:
        return False, "Missing vars!"
    target_shape = (3, 224, 224)
    target_dtype = torch.float32
    if local_vars["shape_x"] != torch.Size(target_shape):
        return False, f"Wrong shape: {local_vars['shape_x']}"
    if local_vars["dtype_x"] != target_dtype:
        return False, f"Wrong dtype: {local_vars['dtype_x']}"
    return True, "Analysis complete."


def check_level_3(local_vars):
    """Level 3: Slicing"""
    if "col_1" not in local_vars:
        return False, "Variable 'col_1' missing!"
    ref_matrix = local_vars.get("matrix")
    if ref_matrix is None:
        return False, "Context Error."
    expected = ref_matrix[:, 0]
    if not torch.equal(local_vars["col_1"], expected):
        return False, "Slice incorrect!"
    return True, "Clean cut!"


def check_level_4(local_vars):
    """Level 4: Broadcasting"""
    if "tensor_broadcast" not in local_vars and "outer_product" not in local_vars:
        return (
            False,
            "Variable 'tensor_broadcast' or function 'outer_product' not found!",
        )

    # 1. Check for one-line code constraint
    code = local_vars.get("__code__", "")
    lines = [
        line.strip()
        for line in code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) > 1:
        return False, f"Code must be one line! (Found {len(lines)} lines)"

    # 2. Check for outer_product function presence and correctness
    if "outer_product" in local_vars:
        func = local_vars["outer_product"]
        if not callable(func):
            return False, "'outer_product' is not a function!"

        try:
            # Test on random inputs
            a = torch.randint(0, 10, (5,))
            b = torch.randint(0, 10, (4,))
            user_res = func(a, b)
            torch_res = torch.outer(a, b)

            if not torch.allclose(user_res, torch_res):
                return False, "Result does not match torch.outer(a, b)!"
        except Exception as e:
            return False, f"Error executed outer_product: {e}"

        return True, "Expansion complete."

    # Fallback for legacy variable check (if we want to support it, but task says write function)
    # The task explicitly asks for `outer_product(vec_a, vec_b)`, so we should enforce that.
    return False, "Function 'outer_product' not defined."


def check_level_5(local_vars):
    """Level 5: Matmul"""
    if "tensor_c" not in local_vars:
        return False, "Missing 'tensor_c'!"
    A = local_vars.get("tensor_a")
    B = local_vars.get("tensor_b")

    if A is None or B is None:
        return False, "Missing 'tensor_a' or 'tensor_b'!"

    # Inputs are (3,) and (4,). Reshape to (3, 1) and (1, 4) for proper broadcasting -> (3, 4)
    expected = A.view(3, 1) @ B.view(1, 4)

    if not torch.allclose(local_vars["tensor_c"], expected):
        return False, "Math incorrect!"
    return True, "The Golem crumbles."


def check_level_6(local_vars):
    """Level 6: Vector Space Axioms (Commutativity)"""
    if "is_commutative" not in local_vars:
        return False, "Function 'is_commutative' not found!"

    func = local_vars["is_commutative"]

    # Test 1: Standard addition (Should be True)
    try:

        def good_add(u, v):
            return u + v

        if not func(good_add):
            return False, "Failed to identify standard addition as commutative!"

        def bad_add(u, v):
            return u - v

        if func(bad_add):
            return False, "Incorrectly identified subtraction as commutative!"

    except Exception as e:
        return False, f"Error calling your function: {e}"

    return True, "Axiom Verified. The space holds."


def check_level_7(local_vars):
    """Level 7: Inner Product Axioms (Linearity)"""
    if "check_inner_product" not in local_vars:
        return False, "Function 'check_inner_product' not found!"

    func = local_vars["check_inner_product"]

    # 1. Valid Dot Product
    def valid_ip(u, v):
        return torch.dot(u, v)

    if not func(valid_ip):
        return False, "Failed to recognize valid dot product!"

    # 2. Non-linear (Affine)
    def nonlinear_ip(u, v):
        return torch.dot(u, v) + 1.0

    if func(nonlinear_ip):
        return (
            False,
            "Failed to reject non-linear function (e.g. <u,v> + 1).",
        )

    return True, "Linearity Confirmed."


def check_level_8(local_vars):
    """Level 8: L2 Norm"""
    if "my_l2_norm" not in local_vars:
        return False, "Function 'my_l2_norm' missing!"

    my_norm = local_vars["my_l2_norm"]
    x = torch.tensor([3.0, 4.0])

    try:
        res = my_norm(x)
        if hasattr(res, "item"):
            res = res.item()

        if abs(res - 5.0) > 1e-4:
            return False, f"Expected 5.0, got {res}"

        # Check if they just hardcoded
        y = torch.tensor([1.0, 1.0])
        res_y = my_norm(y)
        if hasattr(res_y, "item"):
            res_y = res_y.item()
        if abs(res_y - 1.4142) > 1e-3:
            return False, "Generalization failed."

    except Exception as e:
        return False, f"Runtime Error: {e}"

    return True, "Euclidean space mastered."


def check_level_9(local_vars):
    """Level 9: Weighted Inner Product"""
    if "weighted_ip" not in local_vars:
        return False, "Function 'weighted_ip' missing!"

    w_ip = local_vars["weighted_ip"]
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0]])  # Diagonal weights 1, 2

    # xT W y = [1, 2] @ [1, 0; 0, 2] @ [3; 4] = [1, 4] @ [3; 4] = 3 + 16 = 19
    expected = 19.0

    try:
        res = w_ip(x, y, W)
        if torch.is_tensor(res):
            res = res.item()
        if abs(res - expected) > 1e-4:
            return False, f"Expected {expected}, got {res}"
    except Exception as e:
        return False, f"Error: {e}"

    return True, "Weight balanced."


def check_level_10(local_vars):
    """Level 10: Basis Verification"""
    if "is_basis" not in local_vars:
        return False, "'is_basis' missing!"

    func = local_vars["is_basis"]

    # Basis: Identity (2D)
    B1 = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    if not func(B1):
        return False, "Rejected valid basis (Identity)."

    # Dependent
    B2 = [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])]
    if func(B2):
        return False, "Accepted linearly dependent set."

    # Not spanning (too few)
    B3 = [torch.tensor([1.0, 0.0])]
    if func(B3):
        return False, "Accepted set that doesn't span R2 (too few vectors)."

    return True, "Basis verified."


def check_level_11(local_vars):
    """Level 11: Change of Basis"""
    if "get_coordinates" not in local_vars:
        return False, "'get_coordinates' missing!"

    func = local_vars["get_coordinates"]

    # v = 2*b1 + 3*b2
    v = torch.tensor([2.0, 3.0])
    # Standard Basis
    B = torch.eye(2)

    try:
        c = func(v, B)
        if not torch.allclose(c, v):
            return False, "Failed standard basis check."

        # Scaled Basis: b1=[2,0], b2=[0,2] -> v=[2,2] should be c=[1,1]
        B2 = 2 * torch.eye(2)
        v2 = torch.tensor([2.0, 2.0])
        c2 = func(v2, B2)
        if not torch.allclose(c2, torch.tensor([1.0, 1.0])):
            return False, f"Failed scaled basis. Expected [1, 1], got {c2}"

        # Non-Orthogonal Basis Check (CRITICAL)
        # B = [[1, 1], [0, 1]]  columns are [1,0] and [1,1]
        # Target v = [2, 1]
        # c should be [1, 1] because 1*[1,0] + 1*[1,1] = [2,1]
        B3 = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        v3 = torch.tensor([2.0, 1.0])
        c3 = func(v3, B3)
        if not torch.allclose(c3, torch.tensor([1.0, 1.0]), atol=1e-4):
            return (
                False,
                f"Failed non-orthogonal basis check. Your code might assume B is orthogonal (it isn't always!). Expected [1, 1], got {c3}",
            )

    except Exception as e:
        return False, f"Error: {e}"

    return True, "Coordinate shift successful. You handled the general case!"


def get_levels():
    return {
        0: {
            "title": "Floor 0: Random Sampling",
            "desc": f"Before {st.session_state.player_name} enters, they must forge the world.",
            "task": "Create a map named `dungeon_map` of shape `(10, 10)`. The values should be sampled from a uniform distribution over the inclusive range [0, 99]. It will be converted to a map as follows: 0-70=Floor, 71-97=Wall, 98-99=Merchant.",
            "checker": check_level_0,
            "starter_code": "# Create dungeon_map using torch.randint\ndungeon_map = ",
            "hint": "torch.randint(low, high, size)",
            "context_setup": lambda: {},
            "mcq": {
                "q": "What type of distribution does `torch.randint` sample from by default?",
                "opts": ["Normal (Gaussian)", "Uniform", "Binomial", "Exponential"],
                "ans": "Uniform",
                "expl": "Correct! `randint` draws integers uniformly from the range [low, high).",
            },
            "frq": "Why is it important to set a 'seed' when generating random numbers for experiments?",
        },
        1: {
            "title": "Floor 1: Tensor Initialization",
            "desc": "A formless void blocks your path. Give it structure.",
            "task": "Create a tensor named `tensor_x` with shape `(5, 3)` filled with **ones**.",
            "checker": check_level_1,
            "starter_code": "# Create tensor_x here\ntensor_x = ",
            "context_setup": lambda: {},
            "mcq": {
                "q": "Which function creates a tensor filled with zeros?",
                "opts": [
                    "torch.empty()",
                    "torch.null()",
                    "torch.zeros()",
                    "torch.void()",
                ],
                "ans": "torch.zeros()",
                "expl": "Yes! `torch.zeros()` initializes a tensor with all 0s.",
            },
            "frq": "How do you convert a numpy array to a torch tensor?",
        },
        2: {
            "title": "Floor 2: Tensor Properties",
            "desc": f"An unseen enemy lurks. {st.session_state.player_name} must analyze its properties to reveal it.",
            "task": "Given a Tensor `image` (provided), assign its shape to `shape_x` and its data type to `dtype_x`.",
            "checker": check_level_2,
            "starter_code": "# 'image' is already defined for you\n\nshape_x = \ndtype_x = ",
            "hint": "`https://docs.pytorch.org/docs/stable/tensor_attributes.html`",
            "context_setup": lambda: {
                "image": torch.randn(3, 224, 224, dtype=torch.float32)
            },
            "mcq": {
                "q": "What property tells you where the tensor is stored (CPU vs GPU)?",
                "opts": [".place", ".device", ".cloud", ".location"],
                "ans": ".device",
                "expl": "Correct. `.device` tells you if it's on 'cpu' or 'cuda:0'.",
            },
            "frq": "You pass an image tensor into a neural network and get `Expected input of shape (N, C, H, W)`. What debugging steps would you take?",
        },
        3: {
            "title": "Floor 3: Tensor Slicing",
            "desc": "A multi-limbed beast attacks! Sever the first limb!",
            "task": "Given a Tensor `matrix` (4x4), slice out **all rows** of the **first column** (index 0) and save it to `col_1`.",
            "checker": check_level_3,
            "starter_code": "# 'matrix' is defined (4x4)\n\ncol_1 = ",
            "hint": "https://apxml.com/courses/getting-started-with-pytorch/chapter-2-advanced-tensor-manipulations/tensor-indexing-slicing",
            "context_setup": lambda: {"matrix": torch.randint(0, 10, (4, 4))},
            "mcq": {
                "q": "In Python/PyTorch, which index represents the LAST element?",
                "opts": ["0", "1", "-1", "last"],
                "ans": "-1",
                "expl": "Correct! Negative indexing allows you to access elements from the end.",
            },
            "frq": "You have a bug that slices the second column instead of the last. Sometimes it throws an index error, other times it doesn't. What could be the cause?",
        },
        4: {
            "title": "Floor 4: Broadcasting",
            "desc": "Expand your dimensions to fill the space.",
            "task": "Write a function `outer_product(vec_a, vec_b)` in one line that computes the outer product of `vec_a` (shape (m)) and `vec_b` (shape (n)).",
            "checker": check_level_4,
            "starter_code": "# vec_a (m) and vec_b (n) are defined. output shape is (m,n) \n# Write a one-line function\nouter_product = lambda vec_a, vec_b: ...",
            "context_setup": lambda: {
                "vec_a": torch.randint(0, 5, (3,)),
                "vec_b": torch.randint(0, 5, (4,)),
            },
            "hint": "https://www.geeksforgeeks.org/deep-learning/understanding-broadcasting-in-pytorch/",
            "mcq": {
                "q": "Broadcasting works if two dimensions are equal or:",
                "opts": [
                    "One of them is 1",
                    "One of them is 0",
                    "Both are negative",
                    "They are divisible",
                ],
                "ans": "One of them is 1",
                "expl": "Correct! Dimensions must either match or be 1 to broadcast.",
            },
            "frq": "If A.shape is (2,2,2) and B.shape is (2), what is the shape of A+B? What is the broadcasted shape of B?",
        },
        5: {
            "title": "Floor 5: Matrix Multiplication",
            "desc": "The Golem of Algebra blocks the exit. Combine the runes to destroy it.",
            "task": "Perform matrix multiplication: `tensor_c` = `tensor_a` matmul `tensor_b`.",
            "checker": check_level_5,
            "starter_code": "# tensor_a (3,) and tensor_b (4,) are defined. Broadcast them such that multiplication is valid (tensor_a shape (3,1) and tensor_b shape (1,4)) and output shape is (3,4).\n\ntensor_c = ",
            "context_setup": lambda: {
                "tensor_a": torch.randn(
                    3,
                ),
                "tensor_b": torch.randn(
                    4,
                ),
            },
            "hint": "1. `@` is a shorthand for `torch.matmul()`.\n2. Adding new dimension https://stackoverflow.com/questions/65470807/how-to-add-a-new-dimension-to-a-pytorch-tensor",
            "mcq": {
                "q": "To multiply matrix A (MxN) and B (PxQ), what must match?",
                "opts": ["M and P", "N and P", "M and Q", "They don't need to match"],
                "ans": "N and P",
                "expl": "Correct! The inner dimensions (columns of A and rows of B) must match.",
            },
            "frq": "Is matrix multiplication commutative (i.e., is A@B the same as B@A)?",
        },
        6: {
            "title": "Floor 6: Vector Space Axioms",
            # MOVED TO BOSS FIGHT
        },
    }
