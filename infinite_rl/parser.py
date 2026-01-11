import re
import os
from pathlib import Path


class ExampleParser:
    """Parses markdown example files or Gemini-generated text with Prompt, Answer, and Response sections."""

    @staticmethod
    def extract_answer_tags(text):
        """Robustly extract content from <answer> or <summary> tags, handling common LLM malformations."""
        if not text:
            return []

        # 1. Standard tags: <answer>...</answer> or <summary>...</summary>
        matches = re.findall(
            r"<(?:answer|summary)>(.*?)</(?:answer|summary)>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if matches:
            return [m.strip() for m in matches]

        # 2. Heuristic: Look for anything that looks like an opening answer/summary tag and a closing one
        # Handles: ```answer>, [answer], <answer (missing >), etc.
        pattern = r"(?:(?:```+|\[|<)\s*(?:answer|summary)\s*(?:>|\]|```+)?)(.*?)(?:(?:```+|\[|<)\s*/\s*(?:answer|summary)\s*(?:>|\]|```+)?)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return [m.strip() for m in matches]

        # 3. Only opening tag? Take everything after it
        opening_pattern = r"(?:(?:```+|\[|<)\s*(?:answer|summary)\s*(?:>|\]|```+)?)"
        match = re.search(opening_pattern, text, re.IGNORECASE)
        if match:
            # Only if it's not followed by a closing tag later in the text (which Case 2 would have caught)
            # Actually, just take everything after it.
            content = text[match.end() :].strip()
            # If there is a trailing triple backtick followed by NOTHING, it might be the end
            content = re.sub(r"```\s*$", "", content).strip()
            return [content]

        # 4. Last resort: If only closing tag exists, take everything before it (limited to a reasonable window)
        if "</answer>" in text.lower() or "</summary>" in text.lower():
            # Find the last occurrence of </answer> or </summary> (case insensitive)
            match = re.search(
                r"(.*?)</(?:answer|summary)>", text, re.DOTALL | re.IGNORECASE
            )
            if match:
                content = match.group(1).strip()
                # If there's a lot of text, try to find a natural break like a header
                if len(content) > 5000:
                    # Look for the last header before the closing tag as a better starting point
                    header_match = list(
                        re.finditer(r"^#+\s+.*$", content, re.MULTILINE)
                    )
                    if header_match:
                        content = content[header_match[-1].end() :].strip()
                return [content]

        return []

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
            # Join multiple matches with newlines instead of " | " to keep it natural
            # Especially helpful when model wraps multiple parts of a multi-step answer in separate tags
            answer = "\n".join(m.strip() for m in answer_matches)
        else:
            # 2. Try finding code blocks in the Answer section
            match = re.search(
                r"```(?:json|python|javascript|rust|cpp|java|js|ts|typescript|summary|text)?\s*(.*?)```",
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
                            answer = " | ".join(m.strip() for m in response_matches)
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
