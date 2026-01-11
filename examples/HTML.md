## Instruction

Create an HTML page that meets the specified requirements. Your response must contain only valid HTML code. The HTML should be properly structured with semantic tags where appropriate. Provide your answer as follows:

```html
[your code here]
```

## Prompt

Create an HTML page with a navigation bar at the top containing links for "Home", "About", and "Contact". Below the navigation, include a main content area with a heading "Welcome to My Site" and a paragraph describing the site. The page should have a footer with copyright information.

## Answer

```json
{
  "selectors": ["nav", "a[href=\"/\"]", "a[href=\"/about\"]", "a[href=\"/contact\"]", "h1", "main", "footer"]
}
```

## Response

```html
<html>
<head>
    <title>My Site</title>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
        <a href="/contact">Contact</a>
    </nav>
    <main>
        <h1>Welcome to My Site</h1>
        <p>This is a sample website demonstrating HTML structure and semantic markup.</p>
    </main>
    <footer>
        <p>&copy; 2024 My Site. All rights reserved.</p>
    </footer>
</body>
</html>
```

## Reward Function

```python
def reward_fn(model_output, expected_output):
    import re
    import json
    from bs4 import BeautifulSoup
    
    # 1. Format Objective (Part A): Extract HTML code
    code_pattern = r"```(?:html)?\s*(.*?)```"
    match = re.search(code_pattern, model_output, re.DOTALL)
    
    if not match:
        return (0.0, 0.0)
    
    html_code = match.group(1).strip()
    code_format_score = 0.5  # HTML code block found
    
    # 2. Format Objective (Part B): Validate HTML syntax and check for JSON-compatible output
    try:
        soup = BeautifulSoup(html_code, 'html.parser')
        html_format_score = 0.5  # Valid HTML
        
        # 3. Correctness Objective: Validate all selectors from expected answer
        expected_json = json.loads(expected_output.strip())
        selectors = expected_json.get('selectors', [])
        
        all_matched = True
        for selector in selectors:
            if not soup.select(selector):
                all_matched = False
                break
        
        correctness_score = 1.0 if all_matched and len(selectors) > 0 else 0.0
    except Exception:
        html_format_score = 0.0
        correctness_score = 0.0
    
    format_score = code_format_score + html_format_score
    return (format_score, correctness_score)
```
