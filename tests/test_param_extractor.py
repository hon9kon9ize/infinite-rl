"""
Unit tests for parameter extractor utilities.

Tests extraction of default parameters from both Python and JavaScript
function signatures.
"""

import unittest
import json
from infinite_rl.utils.param_extractor import (
    extract_python_params,
    extract_javascript_params,
    extract_puzzle_inputs,
    _parse_javascript_value,
    _parse_javascript_array,
    _parse_javascript_object,
    _extract_python_params_from_string,
)


class TestPythonParamExtraction(unittest.TestCase):
    """Test extraction of parameters from Python functions."""

    def test_no_defaults(self):
        """Test function with no default parameters."""

        def func(a, b, c):
            pass

        result = extract_python_params(func)
        self.assertEqual(result, {})

    def test_all_defaults(self):
        """Test function with all default parameters."""

        def func(a=1, b=2, c=3):
            pass

        result = extract_python_params(func)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

    def test_mixed_params(self):
        """Test function with mix of required and default parameters."""

        def func(a, b=10, c=20):
            pass

        result = extract_python_params(func)
        self.assertEqual(result, {"b": 10, "c": 20})

    def test_various_types(self):
        """Test function with default parameters of various types."""

        def func(a=1, b=3.14, c="hello", d=[1, 2, 3], e={"key": "value"}, f=True):
            pass

        result = extract_python_params(func)
        self.assertEqual(
            result,
            {
                "a": 1,
                "b": 3.14,
                "c": "hello",
                "d": [1, 2, 3],
                "e": {"key": "value"},
                "f": True,
            },
        )

    def test_none_default(self):
        """Test function with None as default."""

        def func(a, b=None):
            pass

        result = extract_python_params(func)
        self.assertEqual(result, {"b": None})


class TestJavaScriptValueParsing(unittest.TestCase):
    """Test parsing of individual JavaScript values."""

    def test_parse_integer(self):
        """Test parsing integer values."""
        self.assertEqual(_parse_javascript_value("42"), 42)
        self.assertEqual(_parse_javascript_value("-5"), -5)
        self.assertEqual(_parse_javascript_value("0"), 0)

    def test_parse_float(self):
        """Test parsing float values."""
        self.assertEqual(_parse_javascript_value("3.14"), 3.14)
        self.assertEqual(_parse_javascript_value("-0.5"), -0.5)
        self.assertEqual(_parse_javascript_value("0.0"), 0.0)

    def test_parse_string(self):
        """Test parsing string values."""
        self.assertEqual(_parse_javascript_value('"hello"'), "hello")
        self.assertEqual(_parse_javascript_value("'world'"), "world")

    def test_parse_boolean(self):
        """Test parsing boolean values."""
        self.assertEqual(_parse_javascript_value("true"), True)
        self.assertEqual(_parse_javascript_value("false"), False)

    def test_parse_null(self):
        """Test parsing null value."""
        self.assertIsNone(_parse_javascript_value("null"))

    def test_parse_simple_array(self):
        """Test parsing simple arrays."""
        self.assertEqual(_parse_javascript_value("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(_parse_javascript_value("[]"), [])

    def test_parse_array_with_floats(self):
        """Test parsing arrays with floating point numbers."""
        self.assertEqual(
            _parse_javascript_value("[1.0, -2.0, -1.0]"), [1.0, -2.0, -1.0]
        )

    def test_parse_string_array(self):
        """Test parsing arrays of strings."""
        result = _parse_javascript_value('["cat", "dog"]')
        self.assertEqual(result, ["cat", "dog"])

    def test_parse_simple_object(self):
        """Test parsing simple objects."""
        result = _parse_javascript_value("{ a: 4, b: 17 }")
        self.assertEqual(result, {"a": 4, "b": 17})

    def test_parse_object_with_string_values(self):
        """Test parsing objects with string values."""
        result = _parse_javascript_value('{ x: "hello", y: "world" }')
        self.assertEqual(result, {"x": "hello", "y": "world"})

    def test_parse_nested_array_in_object(self):
        """Test parsing objects containing arrays."""
        result = _parse_javascript_value("{ arr: [1, 2, 3] }")
        self.assertEqual(result, {"arr": [1, 2, 3]})

    def test_parse_empty_object(self):
        """Test parsing empty objects."""
        result = _parse_javascript_value("{}")
        self.assertEqual(result, {})


class TestJavaScriptArrayParsing(unittest.TestCase):
    """Test parsing of JavaScript arrays."""

    def test_simple_array(self):
        """Test simple array parsing."""
        result = _parse_javascript_array("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])

    def test_empty_array(self):
        """Test empty array parsing."""
        result = _parse_javascript_array("[]")
        self.assertEqual(result, [])

    def test_nested_array(self):
        """Test nested array parsing."""
        result = _parse_javascript_array("[[1, 2], [3, 4]]")
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_mixed_type_array(self):
        """Test array with mixed types."""
        result = _parse_javascript_array('[1, "hello", true, null]')
        self.assertEqual(result, [1, "hello", True, None])

    def test_array_with_objects(self):
        """Test array containing objects."""
        result = _parse_javascript_array("[{a: 1}, {b: 2}]")
        self.assertEqual(result, [{"a": 1}, {"b": 2}])


class TestJavaScriptObjectParsing(unittest.TestCase):
    """Test parsing of JavaScript objects."""

    def test_simple_object(self):
        """Test simple object parsing."""
        result = _parse_javascript_object("{ a: 1, b: 2 }")
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_empty_object(self):
        """Test empty object parsing."""
        result = _parse_javascript_object("{}")
        self.assertEqual(result, {})

    def test_object_with_string_keys(self):
        """Test object with quoted string keys."""
        result = _parse_javascript_object('{ "key1": 1, "key2": 2 }')
        self.assertEqual(result, {"key1": 1, "key2": 2})

    def test_object_with_various_types(self):
        """Test object with various value types."""
        result = _parse_javascript_object(
            '{ int: 42, float: 3.14, str: "hello", bool: true, null_val: null }'
        )
        self.assertEqual(
            result,
            {"int": 42, "float": 3.14, "str": "hello", "bool": True, "null_val": None},
        )

    def test_nested_object(self):
        """Test nested objects."""
        result = _parse_javascript_object("{ outer: { inner: 42 } }")
        self.assertEqual(result, {"outer": {"inner": 42}})

    def test_object_with_array_value(self):
        """Test object containing arrays."""
        result = _parse_javascript_object('{ arr: [1, 2, 3], name: "test" }')
        self.assertEqual(result, {"arr": [1, 2, 3], "name": "test"})


class TestJavaScriptParamExtraction(unittest.TestCase):
    """Test extraction of parameters from JavaScript function signatures."""

    def test_no_params(self):
        """Test function with no parameters."""
        sig = "function sol ()"
        result = extract_javascript_params(sig)
        self.assertEqual(result, {})

    def test_no_defaults(self):
        """Test function with parameters but no defaults."""
        sig = "function sol (a, b, c)"
        result = extract_javascript_params(sig)
        self.assertEqual(result, {})

    def test_single_default(self):
        """Test function with single default parameter."""
        sig = "function sat (answer, n)"
        result = extract_javascript_params(sig)
        self.assertEqual(result, {})

    def test_default_number(self):
        """Test function with numeric default."""
        sig = "function sat (answer, n = 5)"
        result = extract_javascript_params(sig)
        self.assertEqual(result, {"n": 5})

    def test_default_array(self):
        """Test function with array default."""
        sig = "function sat (roots, coeffs = [1.0, -2.0, -1.0])"
        result = extract_javascript_params(sig)
        self.assertEqual(result, {"coeffs": [1.0, -2.0, -1.0]})

    def test_default_object(self):
        """Test function with object default."""
        sig = "function sat (s, counts = { a: 4, b: 17, d: 101, e: 0, f: 12 })"
        result = extract_javascript_params(sig)
        self.assertEqual(
            result, {"counts": {"a": 4, "b": 17, "d": 101, "e": 0, "f": 12}}
        )

    def test_multiple_defaults(self):
        """Test function with multiple default parameters."""
        sig = 'function sat (s, a = "world", b = "Hello world")'
        result = extract_javascript_params(sig)
        self.assertEqual(result, {"a": "world", "b": "Hello world"})

    def test_mixed_params(self):
        """Test function with mix of required and default parameters."""
        sig = 'function sat (s, big_str = "foobar", index = 2)'
        result = extract_javascript_params(sig)
        self.assertEqual(result, {"big_str": "foobar", "index": 2})

    def test_real_world_example_1(self):
        """Test real MostUnique puzzle signature."""
        sig = 'function sat (s, pool = ["cat", "catatatatctsa", "abcdefhijklmnop", "124259239185125", "", "foo", "unique"])'
        result = extract_javascript_params(sig)
        self.assertEqual(
            result,
            {
                "pool": [
                    "cat",
                    "catatatatctsa",
                    "abcdefhijklmnop",
                    "124259239185125",
                    "",
                    "foo",
                    "unique",
                ]
            },
        )

    def test_real_world_example_2(self):
        """Test real CharCounts puzzle signature."""
        sig = "function sat (s, counts = { a: 4, b: 17, d: 101, e: 0, f: 12 })"
        result = extract_javascript_params(sig)
        self.assertEqual(
            result, {"counts": {"a": 4, "b": 17, "d": 101, "e": 0, "f": 12}}
        )

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        sig = "function sat ( s , n = 5 )"
        result = extract_javascript_params(sig)
        self.assertEqual(result, {"n": 5})

    def test_arrow_function(self):
        """Test arrow function syntax."""
        sig = "(s, n = 5) => { return n * 2; }"
        result = extract_javascript_params(sig)
        self.assertEqual(result, {"n": 5})


class TestPythonStringParamExtraction(unittest.TestCase):
    """Test extraction of parameters from Python function signature strings."""

    def test_simple_signature(self):
        """Test simple function signature."""
        sig = "def sat(n: int):"
        result = _extract_python_params_from_string(sig)
        self.assertEqual(result, {})

    def test_with_defaults(self):
        """Test function signature with defaults."""
        sig = "def sat(n: int, year_len=365):"
        result = _extract_python_params_from_string(sig)
        self.assertEqual(result, {"year_len": 365})

    def test_multiple_defaults(self):
        """Test signature with multiple defaults."""
        sig = "def sat(n: int, a=17, b=100, c=20):"
        result = _extract_python_params_from_string(sig)
        self.assertEqual(result, {"a": 17, "b": 100, "c": 20})

    def test_string_default(self):
        """Test signature with string default."""
        sig = 'def sat(s: str, big_str="foobar", index=2):'
        result = _extract_python_params_from_string(sig)
        self.assertEqual(result, {"big_str": "foobar", "index": 2})

    def test_list_default(self):
        """Test signature with list default."""
        sig = "def sat(nums: List[int], default=[1, 2, 3]):"
        result = _extract_python_params_from_string(sig)
        self.assertEqual(result, {"default": [1, 2, 3]})


class TestExtractPuzzleInputs(unittest.TestCase):
    """Test the high-level puzzle inputs extraction function."""

    def test_javascript_puzzle_with_defaults(self):
        """Test extracting inputs from JavaScript puzzle."""
        puzzle_info = {
            "sat": 'function sat (s, big_str = "foobar", index = 2)',
            "name": "StrIndex",
        }
        result = extract_puzzle_inputs(puzzle_info, "javascript")
        self.assertEqual(result, {"big_str": "foobar", "index": 2})

    def test_javascript_puzzle_with_object(self):
        """Test extracting inputs from JavaScript puzzle with object default."""
        puzzle_info = {
            "sat": "function sat (s, counts = { a: 4, b: 17 })",
            "name": "CharCounts",
        }
        result = extract_puzzle_inputs(puzzle_info, "javascript")
        self.assertEqual(result, {"counts": {"a": 4, "b": 17}})

    def test_python_puzzle_with_defaults(self):
        """Test extracting inputs from Python puzzle."""
        puzzle_info = {
            "sat": "def sat(n: int, year_len=365):",
            "name": "BirthdayParadox",
        }
        result = extract_puzzle_inputs(puzzle_info, "python")
        self.assertEqual(result, {"year_len": 365})

    def test_puzzle_no_defaults(self):
        """Test puzzle with no default parameters."""
        puzzle_info = {"sat": "function sol ()", "name": "AllPandigitalSquares"}
        result = extract_puzzle_inputs(puzzle_info, "javascript")
        self.assertEqual(result, {})

    def test_unsupported_language(self):
        """Test handling of unsupported language."""
        puzzle_info = {"sat": "some rust function", "name": "SomePuzzle"}
        result = extract_puzzle_inputs(puzzle_info, "rust")
        self.assertEqual(result, {})


class TestRealWorldPuzzles(unittest.TestCase):
    """Test with real puzzle signatures from the actual puzzles.json."""

    def test_mostunique_javascript(self):
        """Test MostUnique JavaScript puzzle."""
        sig = 'function sat (s, pool = ["cat", "catatatatctsa", "abcdefhijklmnop", "124259239185125", "", "foo", "unique"])'
        result = extract_javascript_params(sig)
        # For MostUnique, sol takes pool as parameter
        self.assertIn("pool", result)
        self.assertEqual(len(result["pool"]), 7)

    def test_charcounts_javascript(self):
        """Test CharCounts JavaScript puzzle."""
        sig = "function sat (s, counts = { a: 4, b: 17, d: 101, e: 0, f: 12 })"
        result = extract_javascript_params(sig)
        self.assertEqual(result["counts"]["a"], 4)
        self.assertEqual(result["counts"]["b"], 17)
        self.assertEqual(result["counts"]["d"], 101)

    def test_birthday_paradox_python(self):
        """Test BirthdayParadox Python puzzle."""
        sig = "def sat(n: int, year_len=365):"
        result = _extract_python_params_from_string(sig)
        self.assertEqual(result, {"year_len": 365})

    def test_strindex_python(self):
        """Test StrIndex Python puzzle."""
        sig = 'def sat(s: str, big_str="foobar", index=2):'
        result = _extract_python_params_from_string(sig)
        self.assertEqual(result, {"big_str": "foobar", "index": 2})


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_function_signature(self):
        """Test handling of empty/invalid function signature."""
        result = extract_javascript_params("not a function")
        self.assertEqual(result, {})

    def test_malformed_object(self):
        """Test handling of malformed object literals."""
        # This should attempt to parse but might fail gracefully
        sig = "function sat (s, obj = { invalid: })"
        # Should not crash, might return empty or partial result
        result = extract_javascript_params(sig)
        self.assertIsInstance(result, dict)

    def test_special_characters_in_strings(self):
        """Test handling of special characters in string values."""
        sig = 'function sat (s, msg = "hello, world!")'
        result = extract_javascript_params(sig)
        self.assertEqual(result["msg"], "hello, world!")

    def test_deeply_nested_structure(self):
        """Test deeply nested structures."""
        sig = "function sat (s, data = { a: { b: { c: 1 } } })"
        result = extract_javascript_params(sig)
        self.assertEqual(result["data"], {"a": {"b": {"c": 1}}})

    def test_array_with_empty_objects(self):
        """Test array containing empty objects."""
        result = _parse_javascript_array("[{}, [], {}]")
        self.assertEqual(result, [{}, [], {}])


class TestPuzzleInputsIntegration(unittest.TestCase):
    """Test that puzzle inputs are correctly extracted and can be used with solver functions."""

    def test_puzzle_inputs_passed_to_python_function(self):
        """Test that extracted inputs can be successfully passed to Python puzzle solver."""

        # Create a mock puzzle solver that expects parameters
        def sol(s, big_str="foobar", index=2):
            """Mock puzzle solver."""
            return s[s.find(big_str) + index] if big_str in s else None

        # Create a mock sat function signature
        sat_signature = 'def sat(s: str, big_str="foobar", index=2):'

        # Extract inputs from the sat signature
        extracted_inputs = _extract_python_params_from_string(sat_signature)

        # Verify inputs were extracted correctly
        self.assertEqual(extracted_inputs, {"big_str": "foobar", "index": 2})

        # Call the solver with extracted inputs using unpacking
        result = sol("test_foobar", **extracted_inputs)
        self.assertIsNotNone(result)
        # "test_foobar".find("foobar") = 5, + index(2) = 7, s[7] = 'o'
        self.assertEqual(result, "o")

    def test_puzzle_inputs_passed_to_python_function_with_numbers(self):
        """Test extraction and passing of numeric puzzle inputs to Python solver."""

        def sol(n, year_len=365):
            """Mock birthday paradox solver."""
            return n > year_len

        sat_signature = "def sat(n: int, year_len=365):"
        extracted_inputs = _extract_python_params_from_string(sat_signature)

        self.assertEqual(extracted_inputs, {"year_len": 365})

        # Test with year_len extracted from signature
        result = sol(100, **extracted_inputs)
        self.assertFalse(result)  # 100 is not > 365

        result = sol(400, **extracted_inputs)
        self.assertTrue(result)  # 400 is > 365

    def test_puzzle_inputs_passed_to_python_function_multiple_params(self):
        """Test extraction and passing of multiple parameters to Python solver."""

        def sol(s, a=0, b=0, c=0):
            """Mock solver with multiple parameters."""
            return sum([s, a, b, c])

        sat_signature = "def sat(s: int, a=1, b=2, c=3):"
        extracted_inputs = _extract_python_params_from_string(sat_signature)

        self.assertEqual(extracted_inputs, {"a": 1, "b": 2, "c": 3})

        # Call with extracted inputs
        result = sol(10, **extracted_inputs)
        self.assertEqual(result, 16)  # 10 + 1 + 2 + 3

    def test_puzzle_inputs_passed_to_javascript_function(self):
        """Test that extracted JavaScript inputs are correctly formatted for use."""
        # JavaScript puzzle signature with default parameters
        js_signature = "function sat(s, counts = { a: 4, b: 17 })"

        # Extract inputs from JavaScript signature
        extracted_inputs = extract_javascript_params(js_signature)

        # Verify inputs were extracted correctly
        self.assertEqual(extracted_inputs, {"counts": {"a": 4, "b": 17}})

        # Verify they can be serialized to JSON (as would be sent to JavaScript runtime)
        json_str = json.dumps(extracted_inputs)
        self.assertIn('"counts"', json_str)
        self.assertIn('"a": 4', json_str)

    def test_puzzle_inputs_with_array_values(self):
        """Test extraction of array-based puzzle inputs."""
        js_signature = "function sat(roots, coeffs = [1.0, -2.0, -1.0])"
        extracted_inputs = extract_javascript_params(js_signature)

        self.assertEqual(extracted_inputs, {"coeffs": [1.0, -2.0, -1.0]})

        # Verify they can be used in computation
        result = sum(extracted_inputs["coeffs"])
        self.assertEqual(result, -2.0)

    def test_puzzle_inputs_extraction_unpacking_order(self):
        """Test that extracted inputs can be unpacked in the correct order."""

        def sol(s, param1=10, param2=20, param3=30):
            """Mock solver expecting specific parameter order."""
            return [s, param1, param2, param3]

        sat_signature = "def sat(s: str, param1=100, param2=200, param3=300):"
        extracted_inputs = _extract_python_params_from_string(sat_signature)

        # Verify extraction
        self.assertEqual(
            extracted_inputs, {"param1": 100, "param2": 200, "param3": 300}
        )

        # Verify unpacking preserves order (Python 3.7+ dict ordering)
        result = sol("test", **extracted_inputs)
        self.assertEqual(result, ["test", 100, 200, 300])

    def test_puzzle_inputs_values_unpacking_pattern(self):
        """Test the values() unpacking pattern used in runner.py."""

        # This mimics the pattern: result = sol(*inputs.values())
        def sol_no_kwargs(arg1, arg2, arg3):
            """Solver that expects positional arguments."""
            return arg1 + arg2 + arg3

        # Extracted inputs would be a dict
        inputs = {"arg1": 10, "arg2": 20, "arg3": 30}

        # This is the pattern used in runner.py line 70
        result = sol_no_kwargs(*inputs.values())
        self.assertEqual(result, 60)

    def test_puzzle_inputs_sat_validation_call_pattern(self):
        """Test the pattern used for sat() validation: sat(result, *inputs.values())."""

        def sat(result, param1, param2):
            """Mock sat function that validates the result."""
            return result == (param1 + param2)

        inputs = {"param1": 5, "param2": 10}
        result_value = 15

        # This mimics runner.py line 75: isCorrect = puzzles[puzzle].sat(result, *inputs.values())
        is_correct = sat(result_value, *inputs.values())
        self.assertTrue(is_correct)

    def test_empty_puzzle_inputs(self):
        """Test handling of puzzles with no default parameters."""

        def sol(s):
            """Solver with no parameters."""
            return len(s)

        sat_signature = "def sat(s: str):"
        extracted_inputs = _extract_python_params_from_string(sat_signature)

        self.assertEqual(extracted_inputs, {})

        # Should still work with unpacking empty dict
        result = sol("test", **extracted_inputs)
        self.assertEqual(result, 4)

    def test_puzzle_inputs_type_consistency(self):
        """Test that extracted inputs maintain type consistency."""
        js_signature = (
            'function sat(s, config = { enabled: true, value: 42, name: "test" })'
        )
        extracted_inputs = extract_javascript_params(js_signature)

        config = extracted_inputs["config"]

        # Verify type conversions from JavaScript to Python
        self.assertIsInstance(config["enabled"], bool)
        self.assertTrue(config["enabled"])
        self.assertIsInstance(config["value"], int)
        self.assertEqual(config["value"], 42)
        self.assertIsInstance(config["name"], str)
        self.assertEqual(config["name"], "test")


if __name__ == "__main__":
    unittest.main()
