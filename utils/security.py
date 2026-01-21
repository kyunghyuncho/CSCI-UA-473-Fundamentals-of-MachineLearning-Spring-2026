import ast


class SecurityException(Exception):
    """Exception raised for security violations in user input."""


def _check_node(node):
    """
    Recursively checks AST nodes against a whitelist of allowed operations.
    """
    # Whitelist of allowed AST nodes for tensor math puzzles
    allowed_nodes = {
        ast.Expression,
        ast.Expr,
        ast.Load,
        ast.Name,
        ast.Constant,
        ast.UnaryOp,
        ast.BinOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Tuple,
        ast.List,
        # Operators
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.Invert,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.MatMult,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.keyword,
        ast.Starred,  # For function calls
    }

    # Python < 3.9 compatibility for Index and ExtSlice if needed
    if hasattr(ast, "Index"):
        allowed_nodes.add(ast.Index)
    if hasattr(ast, "ExtSlice"):
        allowed_nodes.add(ast.ExtSlice)

    if type(node) not in allowed_nodes:
        raise SecurityException(f"Operation '{type(node).__name__}' is not allowed.")

    # Block access to private/dunder attributes and names
    if isinstance(node, ast.Name):
        if node.id.startswith("__"):
            raise SecurityException(
                f"Access to private attribute '{node.id}' is not allowed."
            )

    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            raise SecurityException(
                f"Access to private attribute '{node.attr}' is not allowed."
            )


def validate_expression(expression: str):
    """
    Parses the expression and ensures it only contains allowed operations.

    Args:
        expression (str): The python code/expression string to validate.

    Raises:
        SecurityException: If the expression contains unsafe operations.
        SyntaxError: If the expression is invalid Python.
    """
    try:
        # Parse logic
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        # If it's not a valid expression, it might be a statement (e.g. import os)
        # We try to parse it as exec to see if it's valid Python but unsafe/banned.
        try:
            tree = ast.parse(expression, mode="exec")
        except SyntaxError:
            # If it fails both, it's truly invalid syntax
            raise SyntaxError(f"Invalid syntax: {e}")

    for node in ast.walk(tree):
        _check_node(node)


def safe_eval(expression: str, global_vars=None, local_vars=None):
    """
    Safely evaluates a Python expression using AST validation and restricted scope.

    Args:
        expression (str): The expression to evaluate.
        global_vars (dict): Global variables to make available.
        local_vars (dict): Local variables to make available.

    Returns:
        The result of the evaluation.
    """
    if global_vars is None:
        global_vars = {}

    # 1. Validate Syntax and Structure
    validate_expression(expression)

    # 2. Prepare Safe Environment
    # We explicitly disable __builtins__ to prevent access to open, __import__, etc.
    safe_globals = {"__builtins__": {}}

    # Add allowed globals (like torch)
    safe_globals.update(global_vars)

    # 3. Evaluate
    return eval(expression, safe_globals, local_vars)
