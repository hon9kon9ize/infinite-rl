"""
Validation utilities for puzzle prompt engineering.

This module provides tools to validate puzzle prompts and generated code
to ensure they meet quality standards.
"""

import re
import inspect
from typing import Tuple, List, Dict, Any


class PromptValidator:
    """Validates puzzle prompts and generated code."""

    @staticmethod
    def validate_function_signature(
        code: str, language: str = "python"
    ) -> Tuple[bool, str]:
        """
        Validate that function signature matches requirements.

        Returns:
            (is_valid, error_message)
        """
        if language == "python":
            return PromptValidator._validate_python_signature(code)
        elif language == "javascript":
            return PromptValidator._validate_javascript_signature(code)
        return False, "Unknown language"

    @staticmethod
    def _validate_python_signature(code: str) -> Tuple[bool, str]:
        """Validate Python function signature."""
        # Check function name
        if "def sol(" not in code:
            return False, "Function must be named 'sol'"

        # Extract function signature
        match = re.search(r"def sol\((.*?)\):", code)
        if not match:
            return False, "Invalid function signature syntax"

        params = match.group(1).strip()

        # Check for default values (unless no params)
        if params and "=" in params:
            return False, "Function parameters should not have default values"

        return True, "Valid Python signature"

    @staticmethod
    def _validate_javascript_signature(code: str) -> Tuple[bool, str]:
        """Validate JavaScript function signature."""
        # Check function name
        if "function sol(" not in code:
            return False, "Function must be named 'sol'"

        # Extract function signature
        match = re.search(r"function sol\((.*?)\)", code)
        if not match:
            return False, "Invalid function signature syntax"

        return True, "Valid JavaScript signature"

    @staticmethod
    def validate_code_block(
        code_str: str, language: str = "python"
    ) -> Tuple[bool, str]:
        """
        Validate code block formatting.

        Returns:
            (is_valid, error_message)
        """
        # Check for <answer> tags
        if "<answer>" not in code_str or "</answer>" not in code_str:
            return False, "Code must be wrapped in <answer> tags"

        # Check for triple backticks
        if "```" not in code_str:
            return False, "Code must be in a code block with triple backticks"

        # Check for language specifier
        lang_spec = f"```{language}"
        if lang_spec not in code_str:
            return False, f"Code block must have language specifier: {lang_spec}"

        # Extract code block
        pattern = f"```{language}\n(.*?)\n```"
        match = re.search(pattern, code_str, re.DOTALL)
        if not match:
            return False, f"No valid {language} code block found"

        return True, f"Valid {language} code block"

    @staticmethod
    def validate_prompt_structure(prompt: str) -> Tuple[bool, List[str]]:
        """
        Validate that prompt has required sections.

        Returns:
            (is_valid, missing_sections)
        """
        required_sections = ["[PROMPT]", "[ANSWER]", "[RESPONSE]"]
        missing = [s for s in required_sections if s not in prompt]

        return len(missing) == 0, missing

    @staticmethod
    def extract_parameters_from_sat(sat_signature: str) -> List[str]:
        """
        Extract parameters from sat() function signature.

        Args:
            sat_signature: String like "def sat(answer, nums, target):"

        Returns:
            List of parameter names (excluding 'answer')
        """
        # Extract parameters from sat signature
        match = re.search(r"\((.*?)\)", sat_signature)
        if not match:
            return []

        params_str = match.group(1)
        # Split by comma and clean up
        params = [p.strip().split("=")[0].strip() for p in params_str.split(",")]

        # Remove empty strings and the first parameter (answer)
        params = [p for p in params if p]

        # Return all except the first (which is the answer parameter)
        return params[1:] if params else []

    @staticmethod
    def validate_solution_signature(
        solution_code: str, expected_params: List[str], language: str = "python"
    ) -> Tuple[bool, str]:
        """
        Validate that solution function has correct parameters.

        Args:
            solution_code: The generated sol() function code
            expected_params: List of expected parameter names
            language: 'python' or 'javascript'

        Returns:
            (is_valid, error_message)
        """
        if language == "python":
            match = re.search(r"def sol\((.*?)\):", solution_code)
        else:  # javascript
            match = re.search(r"function sol\((.*?)\)", solution_code)

        if not match:
            return False, "Could not parse function signature"

        params_str = match.group(1).strip()
        actual_params = (
            [p.strip().split("=")[0].strip() for p in params_str.split(",")]
            if params_str
            else []
        )

        # Remove empty strings
        actual_params = [p for p in actual_params if p]

        # Check parameter count
        if len(actual_params) != len(expected_params):
            return (
                False,
                f"Expected {len(expected_params)} params, got {len(actual_params)}: {actual_params}",
            )

        # Check parameter names and order
        if actual_params != expected_params:
            return False, f"Expected params {expected_params}, got {actual_params}"

        return True, "Parameters match"

    @staticmethod
    def validate_no_defaults(
        function_code: str, language: str = "python"
    ) -> Tuple[bool, str]:
        """Check that function parameters don't have default values."""
        if language == "python":
            match = re.search(r"def sol\((.*?)\):", function_code)
        else:
            match = re.search(r"function sol\((.*?)\)", function_code)

        if not match:
            return False, "Could not parse function"

        params = match.group(1)

        # Check for = signs (default values)
        if "=" in params:
            return False, "Function parameters should not have default values"

        return True, "No default values found"


class CodeQualityChecker:
    """Check generated code quality."""

    @staticmethod
    def check_has_input_calls(code: str) -> Tuple[bool, str]:
        """Check if code uses input() calls (not allowed)."""
        if "input(" in code:
            return False, "Code should not use input() calls"
        return True, "No input() calls found"

    @staticmethod
    def check_has_interactive_features(code: str) -> Tuple[bool, str]:
        """Check for interactive features."""
        problematic = ["input(", "raw_input(", "readLine(", "prompt("]
        for item in problematic:
            if item in code:
                return False, f"Code contains interactive feature: {item}"
        return True, "No interactive features found"

    @staticmethod
    def check_standalone(code: str) -> Tuple[bool, str]:
        """Check if code is self-contained."""
        # Very basic check - look for imports
        if "from" in code or "import" in code:
            # Check if imports are standard library (basic check)
            if "import sys" in code or "import os" in code:
                return False, "Code imports system modules"
        return True, "Code appears self-contained"

    @staticmethod
    def check_completeness(code: str, language: str = "python") -> Tuple[bool, str]:
        """Check if code appears complete."""
        if language == "python":
            # Check for obvious incomplete markers
            if "..." in code and "def sol" not in code[: code.find("...")]:
                return False, "Code appears incomplete (contains ...)"
            if "pass" in code and code.strip().endswith("pass"):
                return False, "Function body only contains 'pass'"
        elif language == "javascript":
            if "// TODO" in code or "// FIXME" in code:
                return False, "Code contains TODO/FIXME comments"

        return True, "Code appears complete"


def validate_solution(
    code: str, expected_params: List[str], language: str = "python"
) -> Dict[str, Any]:
    """
    Comprehensive validation of generated solution code.

    Args:
        code: The generated code
        expected_params: Expected parameter list from sat()
        language: 'python' or 'javascript'

    Returns:
        Dictionary with validation results
    """
    results = {"valid": True, "errors": [], "warnings": []}

    # Check code block formatting
    valid, msg = PromptValidator.validate_code_block(code, language)
    if not valid:
        results["valid"] = False
        results["errors"].append(f"Code block: {msg}")

    # Check function signature
    valid, msg = PromptValidator.validate_function_signature(code, language)
    if not valid:
        results["valid"] = False
        results["errors"].append(f"Signature: {msg}")

    # Check parameters
    valid, msg = PromptValidator.validate_solution_signature(
        code, expected_params, language
    )
    if not valid:
        results["valid"] = False
        results["errors"].append(f"Parameters: {msg}")

    # Check for default values
    valid, msg = PromptValidator.validate_no_defaults(code, language)
    if not valid:
        results["valid"] = False
        results["errors"].append(f"Defaults: {msg}")

    # Check code quality
    valid, msg = CodeQualityChecker.check_has_input_calls(code)
    if not valid:
        results["errors"].append(f"Quality: {msg}")

    valid, msg = CodeQualityChecker.check_interactive_features(code)
    if not valid:
        results["errors"].append(f"Quality: {msg}")

    valid, msg = CodeQualityChecker.check_completeness(code, language)
    if not valid:
        results["warnings"].append(f"Completeness: {msg}")

    return results


if __name__ == "__main__":
    # Example usage
    test_sat = "def sat(answer, nums, target, k):"
    params = PromptValidator.extract_parameters_from_sat(test_sat)
    print(f"Extracted parameters: {params}")

    test_code = """
    <answer>
    ```python
    def sol(nums, target, k):
        return nums[:k]
    ```
    </answer>
    """

    result = validate_solution(test_code, params, "python")
    print(f"Validation result: {result}")
