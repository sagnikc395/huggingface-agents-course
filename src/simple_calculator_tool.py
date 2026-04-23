# our calculator is an instsance of the tool

from .tool import tool


@tool
def calculator(a: int, b: int) -> int:
    """multiply two integers"""
    return a * b
