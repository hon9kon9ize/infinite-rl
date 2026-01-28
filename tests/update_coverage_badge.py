#!/usr/bin/env python3
"""
Update coverage badge in README.md based on coverage.xml
"""

import xml.etree.ElementTree as ET
import re
from pathlib import Path


def get_coverage_percentage():
    """Extract coverage percentage from coverage.xml"""
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        print("coverage.xml not found")
        return None

    tree = ET.parse(coverage_file)
    root = tree.getroot()

    # Check if root is coverage element
    if root.tag == "coverage":
        line_rate = root.get("line-rate", "0")
        return round(float(line_rate) * 100)

    # Otherwise look for nested coverage element
    coverage_elem = root.find(".//coverage")
    if coverage_elem is not None:
        line_rate = coverage_elem.get("line-rate", "0")
        return round(float(line_rate) * 100)
    return None


def get_badge_color(percentage):
    """Get badge color based on coverage percentage"""
    if percentage >= 90:
        return "brightgreen"
    elif percentage >= 80:
        return "green"
    elif percentage >= 70:
        return "yellowgreen"
    elif percentage >= 60:
        return "yellow"
    else:
        return "red"


def update_readme_badge(percentage):
    """Update coverage badge in README.md"""
    readme_file = Path("README.md")
    if not readme_file.exists():
        print("README.md not found")
        return False

    content = readme_file.read_text()

    # Pattern to match existing coverage badge
    badge_pattern = r"!\[.*?\]\(https://img\.shields\.io/badge/code%20coverage-[^)]+\)"
    color = get_badge_color(percentage)
    new_badge = f"![code coverage](https://img.shields.io/badge/code%20coverage-{percentage}%25-{color})"

    if re.search(badge_pattern, content):
        # Replace existing badge
        updated_content = re.sub(badge_pattern, new_badge, content)
    else:
        # Add badge at the top if not found
        lines = content.split("\n")
        if lines and lines[0].startswith("# "):
            lines.insert(1, "")
            lines.insert(1, new_badge)
        else:
            lines.insert(0, new_badge)
        updated_content = "\n".join(lines)

    readme_file.write_text(updated_content)
    print(f"Updated coverage badge to {percentage}%")
    return True


def main():
    percentage = get_coverage_percentage()
    if percentage is None:
        print("Could not determine coverage percentage")
        return 1

    if update_readme_badge(percentage):
        print(f"Successfully updated README badge with {percentage}% coverage")
        return 0
    else:
        print("Failed to update README badge")
        return 1


if __name__ == "__main__":
    exit(main())
