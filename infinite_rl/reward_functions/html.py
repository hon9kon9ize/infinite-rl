import json
import re
from typing import Union, Callable, Dict, Any
from .reward_function import RewardFunction, RewardFunctionScore
from bs4 import BeautifulSoup


class HtmlRewardFunction(RewardFunction):
    """Reward function for evaluating LLM-generated HTML solutions."""

    def __init__(self, task_name: str = "html", timeout: int = 5):
        super().__init__(task_name, timeout=timeout)

    def initialize(self):
        """Initialize the HTML reward function."""
        self.initialized = True

    def compute_reward(
        self,
        model_output: str,
        expected_output: Union[str, Dict[str, Any], Callable],
    ) -> RewardFunctionScore:
        from ..parser import ExampleParser

        if not self.initialized:
            self.initialize()

        # Handle expected_output being a JSON string
        if isinstance(expected_output, str):
            # Check for <answer> tags in expected output as well
            exp_matches = ExampleParser.extract_answer_tags(expected_output)
            if exp_matches:
                expected_output = exp_matches[0]

            trimmed = expected_output.strip()
            # Try to find JSON block in the string if it's not already a pure JSON
            if not (trimmed.startswith("{") and trimmed.endswith("}")):
                json_match = re.search(r"({.*})", trimmed, re.DOTALL)
                if json_match:
                    trimmed = json_match.group(1)

            if (trimmed.startswith("{") and trimmed.endswith("}")) or (
                trimmed.startswith("[") and trimmed.endswith("]")
            ):
                try:
                    expected_output = json.loads(trimmed)
                except Exception:
                    pass

        # 1. Format Objective: Check for <answer> tags
        matches = ExampleParser.extract_answer_tags(model_output)

        if not matches:
            return RewardFunctionScore(
                format_score=0.0,
                correctness_score=0.0,
                error_msg="Missing <answer> tags in response. Ensure the HTML is wrapped in <answer> and </answer>.",
            )

        html_to_parse = matches[0] if matches else ""

        # Try to parse the HTML
        try:
            # Clean if it contains markdown code blocks inside <answer>
            code_block_match = re.search(
                r"```(?:html|json)?\s*(.*?)\s*```",
                html_to_parse,
                re.DOTALL | re.IGNORECASE,
            )
            if code_block_match:
                html_to_parse = code_block_match.group(1)
            elif "```" in html_to_parse:
                # Fallback to any code block if html one isn't found
                any_code_match = re.search(
                    r"```(?:\w+)?\s*(.*?)\s*```", html_to_parse, re.DOTALL
                )
                if any_code_match:
                    html_to_parse = any_code_match.group(1)

            # If the content looks like JSON, try to extract 'html' field
            # This handles cases where the model outputs a JSON object containing the HTML
            trimmed = html_to_parse.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    if isinstance(data, dict) and "html" in data:
                        html_to_parse = data["html"]
                except Exception:
                    pass

            soup = BeautifulSoup(html_to_parse, "html.parser")

            # Check for presence of any tags
            tags = soup.find_all()
            if not tags:
                return RewardFunctionScore(
                    format_score=1.0 if trimmed.startswith("{") else 0.0,
                    correctness_score=0.0,
                    error_msg="No HTML tags found in output",
                )

            # Check if it looks like Python code instead (common hallucination)
            if "def " in model_output and "(" in model_output and "):" in model_output:
                return RewardFunctionScore(
                    format_score=0.0,
                    correctness_score=0.0,
                    error_msg="Output looks like Python code instead of HTML",
                )

            format_score = 1.0  # Valid HTML syntax with tags
        except Exception as e:
            # Invalid HTML syntax
            return RewardFunctionScore(
                format_score=0.0, correctness_score=0.0, error_msg=str(e)
            )

        # Handle different expected_output types
        if callable(expected_output):
            # Callable: pass soup object to validator function
            try:
                result = expected_output(soup)
                if isinstance(result, bool):
                    correctness_score = 1.0 if result else 0.0
                elif isinstance(result, float):
                    correctness_score = result
                else:
                    correctness_score = 0.0
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=correctness_score
                )
            except Exception as e:
                return RewardFunctionScore(
                    format_score=format_score, correctness_score=0.0
                )

        elif isinstance(expected_output, dict):
            # Dict: check selectors or expected HTML
            selectors = expected_output.get("selectors", [])
            expected_html = expected_output.get("expected_html", None)

            all_matched = True
            total = 0
            error_msg = ""

            # Check CSS selectors - all must match
            if selectors:
                total = len(selectors)
                missing_selectors = []
                for selector in selectors:
                    try:
                        # Standard BeautifulSoup selection
                        elements = soup.select(selector)
                        if elements:
                            continue

                        # If not found via select, check special cases
                        is_found = False
                        # 1. Check if it's an at-rule (like @media)
                        if selector.startswith("@"):
                            if selector in html_to_parse:
                                is_found = True

                        if not is_found:
                            all_matched = False
                            missing_selectors.append(selector)

                    except Exception:
                        # If selector is invalid for soup.select (like @media), try string match as fallback
                        if selector in html_to_parse:
                            continue
                        all_matched = False
                        missing_selectors.append(selector)

                if missing_selectors:
                    error_msg = f"Missing CSS selectors: {', '.join(missing_selectors)}"

            # Check expected HTML substring
            if expected_html and all_matched:
                if expected_html not in model_output:
                    all_matched = False
                    error_msg = f"Expected HTML substring not found: {expected_html}"
                total += 1

            correctness_score = 1.0 if all_matched and total > 0 else 0.0
            return RewardFunctionScore(
                format_score=format_score,
                correctness_score=correctness_score,
                error_msg=error_msg if not all_matched else "",
            )

        else:
            # String: treat as CSS selector
            selector = str(expected_output).strip()

            # Simple heuristic: if it contains spaces and is long, it's probably not a selector
            if len(selector) > 100 or "\n" in selector:
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=f"Expected output (JSON) failed to parse, and it doesn't look like a valid CSS selector.",
                )

            try:
                elements = soup.select(selector)
                correctness_score = 1.0 if elements else 0.0
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=correctness_score,
                    error_msg=(
                        f"CSS selector '{selector}' not found in HTML"
                        if not elements
                        else ""
                    ),
                )
            except Exception as e:
                return RewardFunctionScore(
                    format_score=format_score,
                    correctness_score=0.0,
                    error_msg=f"Invalid CSS selector or error during selection: {e}",
                )
