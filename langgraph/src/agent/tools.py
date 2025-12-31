from langchain.tools import tool


# way1: to use tool decorator
@tool
def add(a: int, b: int) -> str:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int

    Returns:    
        The product of `a` + `b`.          
    """
    result = a + b
    return f"{a} + {b} = {result}"


from langchain_core.utils.pydantic import TypeBaseModel

class args_multiply(TypeBaseModel):
    a: int
    b: int

# way2: to use tool decorator with args_schema
@tool("multiply", description="Multiplies two integers together." , args_schema=args_multiply)
def multiply(a: int, b: int) -> int:
    return a * b


# way3: to use Tool class directly
from langchain.tools import BaseTool
def subtract(a: int, b: int) -> int:
    """Subtracts `b` from `a`.

    Args:
        a: First int
        b: Second int

    Returns:    
        The difference of `a` - `b`.          
    """
    return a - b

subtract_tool = BaseTool(
    name="subtract",
    func=subtract,
    description="Subtracts b from a."
)

# way4: 
from langchain.tools import Tool
from langchain_core.utils.pydantic import TypeBaseModel
class args_divide(TypeBaseModel):
    a: int
    b: int
def divide(a: int, b: int) -> float:
    """Divides `a` by `b`.

    Args:
        a: First int
        b: Second int

    Returns:    
        The quotient of `a` / `b`.          
    """
    return a / b    
