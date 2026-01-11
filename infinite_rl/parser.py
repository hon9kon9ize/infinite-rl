import re
import os
from pathlib import Path


class ExampleParser:
    """Parses markdown example files or Gemini-generated text with Prompt, Answer, and Response sections."""

    @staticmethod
    def parse_text(text):
        """Parse raw text string (useful for Gemini responses)."""
        sections = {}
        # Simple regex to split by ## headers
        current_header = None
        lines = text.split("\n")

        current_content = []
        for line in lines:
            if line.startswith("## "):
                if current_header:
                    sections[current_header] = "\n".join(current_content).strip()
                current_header = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)

        if current_header:
            sections[current_header] = "\n".join(current_content).strip()

        # Clean up specific sections
        prompt = sections.get("Prompt", "")

        # Extract answer from code block if present
        answer_raw = sections.get("Answer", "")
        # Try finding answer in <answer> tags or code blocks
        answer_match = re.search(
            r"<(?:answer|summary)>(.*?)</(?:answer|summary)>", answer_raw, re.DOTALL
        )
        match = answer_match
        if not match:
            match = re.search(
                r"```(?:json|python|javascript|rust|cpp|java|js|ts|typescript)?\s*(.*?)```",
                answer_raw,
                re.DOTALL | re.IGNORECASE,
            )

        answer = match.group(1).strip() if match else answer_raw

        # Response logic
        response = sections.get("Response", "")

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
