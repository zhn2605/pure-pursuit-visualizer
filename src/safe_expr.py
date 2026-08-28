"""Safe evaluation of user-supplied path equations.

Players type things like ``8*cos(pi*(x-11.5)/10)`` into a web form. That string
must never reach :func:`eval`, so this module walks the AST itself and only
evaluates a small whitelist of nodes against numpy vectorised primitives.
"""

import ast
import math

import numpy as np

MAX_EXPR_LEN = 240
MAX_NODES = 160
MAX_DEPTH = 25
MAX_ABS_EXPONENT = 32


class ExpressionError(ValueError):
    """Raised when an equation is malformed or uses something not allowed."""


# Every callable a player is allowed to name, mapped to a vectorised numpy op.
FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "sign": np.sign, "hypot": np.hypot, "mod": np.mod,
    "min": np.minimum, "max": np.maximum, "clip": np.clip, "where": np.where,
}

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# One worked example per available name, shown as clickable presets in the UI.
# Each is tuned to sweep a visible range over the arena's x = 0..60. The test
# suite asserts this covers every entry in FUNCTIONS, so adding a function
# without an example is a build failure rather than a gap in the UI.
EXAMPLES = [
    ("flat",   "0"),
    ("sin",    "6*sin(x/4)"),
    ("cos",    "6*cos(x/4)"),
    ("tan",    "6*tan(x/45)"),
    ("asin",   "8*asin(sin(x/9))"),
    ("acos",   "8*acos(cos(x/9)) - 12"),
    ("atan",   "6*atan(x/10 - 3)"),
    ("atan2",  "8*atan2(x - 30, 12)"),
    ("sinh",   "6*sinh(x/25 - 1.2)"),
    ("cosh",   "14 - 3*cosh(x/18 - 1.7)"),
    ("tanh",   "10*tanh(x/10 - 3)"),
    ("exp",    "exp(x/20) - 10"),
    ("log",    "6*log(x + 1) - 12"),
    ("log10",  "12*log10(x + 1) - 10"),
    ("sqrt",   "3*sqrt(x) - 12"),
    ("abs",    "abs(x - 30)/3 - 8"),
    ("floor",  "3*floor(x/8) - 10"),
    ("ceil",   "3*ceil(x/8) - 12"),
    ("round",  "4*round(sin(x/8))"),
    ("sign",   "8*sign(sin(x/6))"),
    ("hypot",  "hypot(x - 30, 10) - 22"),
    ("mod",    "mod(x, 16) - 8"),
    ("min",    "min(x/4 - 10, 6)"),
    ("max",    "max(10 - x/3, -10)"),
    ("clip",   "clip(x/2 - 15, -10, 10)"),
    ("where",  "where(x < 30, -8, 8)"),
    ("pi",     "8*sin(2*pi*x/30)"),
    ("e",      "4*log(x + 1)/log(e) - 10"),
    ("tau",    "7*cos(tau*x/40)"),
]

# Arity limits so clip(x) or where(x) fail with a clear message rather than a
# numpy TypeError that means nothing to a student at a fair.
_ARITY = {
    "atan2": (2, 2), "hypot": (2, 2), "mod": (2, 2),
    "min": (2, 2), "max": (2, 2),
    "clip": (3, 3), "where": (3, 3),
}

_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
    ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod,
    ast.FloorDiv: np.floor_divide,
}
_UNARYOPS = {ast.UAdd: np.positive, ast.USub: np.negative}
_CMPOPS = {
    ast.Lt: np.less, ast.LtE: np.less_equal,
    ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


def _const_number(node):
    """Return the numeric value of a literal node, or None if it isn't one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        inner = _const_number(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _eval_node(node, x, depth=0):
    if depth > MAX_DEPTH:
        raise ExpressionError("Equation is nested too deeply.")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("Only plain numbers are allowed as literals.")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ExpressionError(
            f"Unknown name '{node.id}'. Use 'x', a number, or one of: "
            + ", ".join(sorted(FUNCTIONS))
        )

    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError("That operator is not allowed.")
        if isinstance(node.op, ast.Pow):
            exponent = _const_number(node.right)
            if exponent is None:
                raise ExpressionError("Exponents must be plain numbers, e.g. x**2.")
            if abs(exponent) > MAX_ABS_EXPONENT:
                raise ExpressionError(
                    f"Exponent must be between -{MAX_ABS_EXPONENT} and {MAX_ABS_EXPONENT}."
                )
        return op(_eval_node(node.left, x, depth + 1),
                  _eval_node(node.right, x, depth + 1))

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ExpressionError("That operator is not allowed.")
        return op(_eval_node(node.operand, x, depth + 1))

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ExpressionError("Chained comparisons like 0 < x < 5 are not supported.")
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise ExpressionError("That comparison is not allowed.")
        left = _eval_node(node.left, x, depth + 1)
        right = _eval_node(node.comparators[0], x, depth + 1)
        # Comparisons yield booleans; make them numeric so 2*(x<5) behaves.
        return np.asarray(op(left, right), dtype=float)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("Only direct function calls like sin(x) are allowed.")
        name = node.func.id
        fn = FUNCTIONS.get(name)
        if fn is None:
            raise ExpressionError(
                f"Unknown function '{name}'. Available: " + ", ".join(sorted(FUNCTIONS))
            )
        if node.keywords:
            raise ExpressionError("Keyword arguments are not supported.")
        lo, hi = _ARITY.get(name, (1, 1))
        if not lo <= len(node.args) <= hi:
            expected = f"{lo}" if lo == hi else f"{lo}-{hi}"
            raise ExpressionError(f"{name}() takes {expected} argument(s).")
        return fn(*[_eval_node(a, x, depth + 1) for a in node.args])

    raise ExpressionError("Unsupported syntax in equation.")


def compile_equation(expression):
    """Parse ``expression`` and return ``f(x_array) -> y_array``.

    Raises :class:`ExpressionError` with a player-readable message on anything
    malformed, unsupported, or numerically degenerate.
    """
    if not isinstance(expression, str):
        raise ExpressionError("Equation must be text.")
    expression = expression.strip()
    if not expression:
        raise ExpressionError("Equation is empty.")
    if len(expression) > MAX_EXPR_LEN:
        raise ExpressionError(f"Equation must be under {MAX_EXPR_LEN} characters.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Could not parse equation: {exc.msg}") from exc

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_NODES:
        raise ExpressionError("Equation is too complicated.")

    def f(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            result = _eval_node(tree.body, x)
        result = np.asarray(result, dtype=float)
        if result.ndim == 0:
            result = np.full(x.shape, float(result))
        elif result.shape != x.shape:
            result = np.broadcast_to(result, x.shape).astype(float, copy=True)
        return result

    # Evaluate once up front so bad equations fail at submit time, not mid-run.
    probe = f(np.linspace(-1.0, 1.0, 5))
    if not np.all(np.isfinite(probe)):
        # Non-finite over the probe range is only fatal if it is non-finite
        # everywhere; 1/x is fine away from the origin.
        if not np.any(np.isfinite(probe)):
            raise ExpressionError("Equation does not produce real numbers.")
    return f


def evaluate(expression, x):
    """Convenience one-shot: compile ``expression`` and evaluate it at ``x``."""
    return compile_equation(expression)(x)
