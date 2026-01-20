import re
import os
from pathlib import Path


class ExampleParser:
    """Parses markdown example files or Gemini-generated text with Prompt, Answer, and Response sections."""

    @staticmethod
    def extract_answer_tags(text, tag=None):
        """Compatibility wrapper delegating to `infinite_rl.utils.parser_utils.extract_answer_tags`.

        Note: new helper accepts a single `tag` and returns a single string.
        """
        from .utils.parser_utils import extract_answer_tags as _extract

        return _extract(text, tag=tag)

    @staticmethod
    def parse_text(text):
        """Parse raw text string (useful for Gemini responses)."""
        sections = {}
        # Robust regex to split by headers:
        # 1. Bracketed style: [PROMPT]
        # 2. Markdown style: ## Prompt
        # 3. Uppercase keywords
        header_pattern = re.compile(
            r"^(?:#+\s*|\[)\s*(.+?)\s*\]?[:\s]*$",
            re.IGNORECASE,
        )

        lines = text.split("\n")
        current_header = None
        current_content = []
        in_code_block = False

        for line in lines:
            # Check for code blocks to avoid matching headers inside them
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_content.append(line)
                continue

            if in_code_block:
                current_content.append(line)
                continue

            match = header_pattern.match(line)
            if match:
                if current_header:
                    sections[current_header] = "\n".join(current_content).strip()

                header_name = match.group(1).strip().rstrip(":")
                # Normalize common headers to Proper Case
                lower_header_name = header_name.lower()

                # Use stricter matching for common headers to avoid false positives in comments
                if lower_header_name in ["prompt", "input", "question", "instruction"]:
                    current_header = "Prompt"
                elif lower_header_name in [
                    "answer",
                    "expected answer",
                    "solution",
                    "target",
                ]:
                    current_header = "Answer"
                elif lower_header_name in [
                    "response",
                    "output",
                    "model output",
                    "model response",
                ]:
                    current_header = "Response"
                else:
                    current_header = header_name

                current_content = []
            else:
                current_content.append(line)

        if current_header:
            sections[current_header] = "\n".join(current_content).strip()

        # Clean up specific sections
        prompt = sections.get("Prompt", "")

        # Extract answer from code block if present
        answer_raw = sections.get("Answer", "")
        # 1. Try finding answer in <answer> tags in the Answer section
        answer_matches = ExampleParser.extract_answer_tags(answer_raw)

        if answer_matches:
            # Helper now returns a single string (possibly containing newlines for multiple occurrences)
            answer = answer_matches
        else:
            # 2. Try finding code blocks in the Answer section
            match = re.search(
                r"```(?:json|python|javascript|rust|cpp|java|js|summary|text)?\s*(.*?)```",
                answer_raw,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                answer = match.group(1).strip()
            else:
                # 3. Try finding JSON-like structure in the Answer section
                json_only_match = re.search(r"(\{.*\})", answer_raw, re.DOTALL)
                if json_only_match:
                    answer = json_only_match.group(1).strip()
                else:
                    # 4. Fallback: Check Response section for <answer> tags
                    # This handles cases where the model puts the final answer only in the response tags
                    if "Response" in sections:
                        response_matches = ExampleParser.extract_answer_tags(
                            sections["Response"]
                        )
                        if response_matches:
                            # The utility returns a single string (may include newlines)
                            answer = response_matches
                        else:
                            answer = answer_raw.strip()
                    else:
                        answer = answer_raw.strip()

        # Response logic
        response = sections.get("Response", "")

        # Fallback: If Response header was completely missing or empty,
        # but tags are present elsewhere, use the whole text.
        if not response:
            # Check if there are <answer> tags anywhere in the full text
            if ExampleParser.extract_answer_tags(text):
                # If tags exist, the model likely just skipped the ## Response header
                # We'll treat the text after the ## Answer block (if any) or the whole text as response
                if "Answer" in sections and len(sections["Answer"]) < len(text) * 0.5:
                    # If Answer is short, maybe Response is just mixed in.
                    response = text
                else:
                    response = text
            elif "Answer" in sections and len(sections["Answer"]) > 500:
                # If Answer section is very long, maybe it contains the response
                response = sections["Answer"]

        return {"prompt": prompt, "answer": answer, "response": response}

    @staticmethod
    def parse_file(file_path):
        with open(file_path, "r") as f:
            content = f.read()
        return ExampleParser.parse_text(content)

    @staticmethod
    def get_all_examples(examples_dir):
        examples = {}
        path = Path(examples_dir)
        for file in path.glob("*.md"):
            examples[file.stem] = ExampleParser.parse_file(file)
        return examples
