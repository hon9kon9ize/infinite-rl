"""
Parameter extractor for puzzle functions.

Extracts default parameter values from both Python and JavaScript function signatures.
Supports extracting inputs from sat() function signatures to use when calling sol().
"""

import re
import json
import inspect
from typing import Dict, Any, Tuple, Optional, Union


def extract_python_params(func) -> Dict[str, Any]:
    """
    Extract default parameters from a Python function using inspect.

    Args:
        func: A Python callable (function or method)

    Returns:
        Dictionary mapping parameter names to their default values.
        Only parameters with default values are included.

    Example:
        >>> def example(a, b=10, c="hello"):
        ...     pass
        >>> extract_python_params(example)
        {'b': 10, 'c': 'hello'}
    """
    try:
        spec = inspect.getfullargspec(func)
        if spec.defaults:
            # Match defaults to the last N arguments
            return dict(zip(spec.args[-len(spec.defaults) :], spec.defaults))
        return {}
    except (TypeError, ValueError):
        return {}


def extract_javascript_params(func_signature: str) -> Dict[str, Any]:
    """
    Extract default parameters from a JavaScript function signature string.

    Handles both ES6 function declarations and arrow functions.
    Supports default values for parameters like:
    - Primitives: 5, "string", true, null
    - Arrays: [1, 2, 3], []
    - Objects: { a: 1, b: 2 }

    Args:
        func_signature: JavaScript function signature string
                       e.g., "function sat (roots, coeffs = [1.0, -2.0, -1.0])"
                       or "function sol (index)"

    Returns:
        Dictionary mapping parameter names to their default values.
        Only parameters with default values are included.

    Example:
        >>> sig = 'function sat (s, counts = { a: 4, b: 17 })'
        >>> extract_javascript_params(sig)
        {'counts': {'a': 4, 'b': 17}}
    """
    params = {}

    # Extract the parameter list from function signature
    # Matches: function name (params) or (params) =>
    match = re.search(r"\(([^)]*)\)", func_signature)
    if not match:
        return params

    param_str = match.group(1)
    if not param_str.strip():
        return params

    # Parse parameters with default values
    # We need to be careful with nested braces/brackets in default values
    current_param = ""
    depth = 0  # Track nesting depth of {, [, (
    in_string = False
    string_char = None

    for char in param_str:
        # Handle string literals
        if char in ('"', "'") and (not in_string or string_char == char):
            if in_string and string_char == char:
                in_string = False
                string_char = None
            elif not in_string:
                in_string = True
                string_char = char
            current_param += char
            continue

        if in_string:
            current_param += char
            continue

        # Track nesting depth
        if char in ("{", "[", "("):
            depth += 1
            current_param += char
        elif char in ("}", "]", ")"):
            depth -= 1
            current_param += char
        elif char == "," and depth == 0:
            # End of parameter
            _process_param(current_param.strip(), params)
            current_param = ""
        else:
            current_param += char

    # Process the last parameter
    if current_param.strip():
        _process_param(current_param.strip(), params)

    return params


def _process_param(param_str: str, params_dict: Dict[str, Any]) -> None:
    """
    Process a single parameter string and extract name and default value.

    Args:
        param_str: Parameter string like "a" or "b = 10" or "arr = [1, 2, 3]"
        params_dict: Dictionary to store the extracted parameter (modified in place)
    """
    if "=" not in param_str:
        return  # No default value

    # Split on first '=' to get name and value
    parts = param_str.split("=", 1)
    if len(parts) != 2:
        return

    param_name = parts[0].strip()
    default_value_str = parts[1].strip()

    # Parse the default value
    try:
        parsed_value = _parse_javascript_value(default_value_str)
        params_dict[param_name] = parsed_value
    except (json.JSONDecodeError, ValueError):
        # If we can't parse it, store as string
        params_dict[param_name] = default_value_str


def _parse_javascript_value(value_str: str) -> Any:
    """
    Parse a JavaScript value string to Python equivalent.

    Handles:
    - Numbers: 5, 3.14, -1
    - Strings: "hello", 'world'
    - Booleans: true, false
    - Null: null
    - Arrays: [1, 2, 3]
    - Objects: { a: 1, b: "value" }

    Args:
        value_str: String representation of JavaScript value

    Returns:
        Parsed Python equivalent

    Raises:
        ValueError: If the value cannot be parsed
    """
    value_str = value_str.strip()

    # Handle null
    if value_str == "null":
        return None

    # Handle booleans
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Handle strings (single or double quoted)
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    # Handle numbers
    try:
        if "." in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass

    # Handle arrays [...]
    if value_str.startswith("[") and value_str.endswith("]"):
        return _parse_javascript_array(value_str)

    # Handle objects {...}
    if value_str.startswith("{") and value_str.endswith("}"):
        return _parse_javascript_object(value_str)

    # If all else fails, try JSON parsing (works for many cases)
    # Replace JS-style undefined, NaN, Infinity if needed
    try:
        return json.loads(value_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JavaScript value: {value_str}") from e


def _parse_javascript_array(array_str: str) -> list:
    """
    Parse a JavaScript array string to Python list.

    Args:
        array_str: String like "[1, 2, 3]" or "[]"

    Returns:
        Parsed Python list
    """
    # Remove outer brackets
    content = array_str[1:-1].strip()
    if not content:
        return []

    # Split by comma, respecting nesting
    elements = []
    current = ""
    depth = 0
    in_string = False
    string_char = None

    for char in content:
        if char in ('"', "'") and (not in_string or string_char == char):
            if in_string and string_char == char:
                in_string = False
                string_char = None
            elif not in_string:
                in_string = True
                string_char = char
            current += char
            continue

        if in_string:
            current += char
            continue

        if char in ("{", "[", "("):
            depth += 1
            current += char
        elif char in ("}", "]", ")"):
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            elements.append(_parse_javascript_value(current.strip()))
            current = ""
        else:
            current += char

    if current.strip():
        elements.append(_parse_javascript_value(current.strip()))

    return elements


def _parse_javascript_object(obj_str: str) -> dict:
    """
    Parse a JavaScript object string to Python dict.

    Args:
        obj_str: String like "{ a: 1, b: 2 }" or "{}"

    Returns:
        Parsed Python dict
    """
    # Remove outer braces
    content = obj_str[1:-1].strip()
    if not content:
        return {}

    result = {}
    current_key = ""
    current_value = ""
    in_key = True
    depth = 0
    in_string = False
    string_char = None

    for char in content:
        if char in ('"', "'") and (not in_string or string_char == char):
            if in_string and string_char == char:
                in_string = False
                string_char = None
            elif not in_string:
                in_string = True
                string_char = char
            if in_key:
                current_key += char
            else:
                current_value += char
            continue

        if in_string:
            if in_key:
                current_key += char
            else:
                current_value += char
            continue

        if in_key:
            if char == ":":
                in_key = False
                continue
            else:
                current_key += char
        else:
            if char in ("{", "[", "("):
                depth += 1
                current_value += char
            elif char in ("}", "]", ")"):
                depth -= 1
                current_value += char
            elif char == "," and depth == 0:
                # End of this key-value pair
                key = current_key.strip().strip("\"'")
                value = _parse_javascript_value(current_value.strip())
                result[key] = value
                current_key = ""
                current_value = ""
                in_key = True
            else:
                current_value += char

    # Process the last key-value pair
    if current_key.strip():
        key = current_key.strip().strip("\"'")
        value = (
            _parse_javascript_value(current_value.strip())
            if current_value.strip()
            else None
        )
        result[key] = value

    return result


def extract_puzzle_inputs(puzzle_info: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    Extract puzzle inputs from puzzle metadata.

    For Python puzzles, if a puzzle_class is available, extracts using inspect.
    For JavaScript puzzles, parses the sat function signature.

    Args:
        puzzle_info: Dictionary containing puzzle metadata from puzzles.json
                    Must contain 'sat' key with the function signature
        language: Programming language ('python' or 'javascript')

    Returns:
        Dictionary of parameter names to their default values
    """
    if language.lower() == "python":
        # For Python, try to get from the puzzle class if available
        # Otherwise, parse from sat function signature string
        sat_src = puzzle_info.get("sat", "")
        if isinstance(sat_src, str):
            # Parse Python sat signature like "def sat(n: int, year_len=365):"
            return _extract_python_params_from_string(sat_src)
        return {}
    elif language.lower() == "javascript":
        sat_src = puzzle_info.get("sat", "")
        if isinstance(sat_src, str):
            return extract_javascript_params(sat_src)
        return {}
    else:
        return {}


def _extract_python_params_from_string(sat_signature: str) -> Dict[str, Any]:
    """
    Extract default parameters from a Python function signature string.

    Args:
        sat_signature: Python function signature like "def sat(n: int, a=17, b=100):"

    Returns:
        Dictionary mapping parameter names to their default values
    """
    # Extract parameter list from function signature
    match = re.search(r"\(([^)]*)\)", sat_signature)
    if not match:
        return {}

    param_str = match.group(1)
    params = {}

    # Split parameters by comma, respecting nested parentheses
    current_param = ""
    depth = 0

    for char in param_str:
        if char in ("(", "["):
            depth += 1
            current_param += char
        elif char in (")", "]"):
            depth -= 1
            current_param += char
        elif char == "," and depth == 0:
            _process_python_param(current_param.strip(), params)
            current_param = ""
        else:
            current_param += char

    if current_param.strip():
        _process_python_param(current_param.strip(), params)

    return params


def _process_python_param(param_str: str, params_dict: Dict[str, Any]) -> None:
    """
    Process a single Python parameter string.

    Args:
        param_str: Parameter string like "a" or "b: int = 10" or "c: str = 'hello'"
        params_dict: Dictionary to store the extracted parameter (modified in place)
    """
    if "=" not in param_str:
        return  # No default value

    # Split on '=' to separate name and default value
    parts = param_str.split("=", 1)
    if len(parts) != 2:
        return

    # Extract parameter name (remove type hint)
    param_part = parts[0].strip()
    if ":" in param_part:
        param_name = param_part.split(":")[0].strip()
    else:
        param_name = param_part

    default_str = parts[1].strip()

    # Parse the default value
    try:
        # Try to evaluate it as Python literal
        parsed_value = ast_literal_eval(default_str)
        params_dict[param_name] = parsed_value
    except (ValueError, SyntaxError):
        # Store as string if we can't parse
        params_dict[param_name] = default_str


def ast_literal_eval(value_str: str) -> Any:
    """
    Safely evaluate a Python literal from string.

    Uses ast.literal_eval which safely handles Python literals.

    Args:
        value_str: String representation of a Python literal

    Returns:
        Evaluated Python value

    Raises:
        ValueError: If the string is not a valid Python literal
    """
    import ast

    return ast.literal_eval(value_str)
