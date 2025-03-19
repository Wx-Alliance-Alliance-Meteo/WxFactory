import ast
import math
import operator

__all__ = ["eval_expr"]

_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    # ast.Pow: operator.pow,
    # ast.BitXor: operator.xor,
    ast.USub: operator.neg,
}

_math_constants = {"pi": math.pi}


def eval_expr(expr):
    """Evaluate the expression from the given string to get a numerical result. Only simple operations are allowed:
    +, -, *, / and some mathematical constants."""
    return _eval(ast.parse(expr, mode="eval").body)


def _eval(node):
    """Recursively evaluate the given node. Can treat integer and real literals, named constants (e.g. pi), and
    basic operators (+, -, *, /)."""
    match node:
        case ast.Constant(value) if isinstance(value, (float, int)):
            return value
        case ast.Name(value):
            return _math_constants[value.lower()]
        case ast.BinOp(left, op, right):
            return _operators[type(op)](_eval(left), _eval(right))
        case ast.UnaryOp(op, operand):
            return _operators[type(op)](_eval(operand))
        case _:
            raise TypeError(node)


if __name__ == "__main__":  # Test this module
    expressions = [
        ("1+1", 1 + 1),
        ("2*3 + 1", 2 * 3 + 1),
        ("pi", math.pi),
        ("pi/4", math.pi / 4),
        ("Pi*2", math.pi * 2),
        ("pI/2", math.pi / 2),
        ("-pi", -math.pi),
        ("2", 2),
        ("1.4", 1.4),
    ]
    for expr, expected in expressions:
        result = eval_expr(expr)
        if result != expected:
            raise ValueError(f"Expected {expected}, got {result}")
        print(f"{expr} = {result} ({type(result)})")
