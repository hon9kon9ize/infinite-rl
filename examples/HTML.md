## Instruction

Create an HTML page that meets the specified requirements. Your response must contain only valid HTML code. The HTML should be properly structured with semantic tags where appropriate. Provide your answer as follows:

```html
[your code here]
```

## Question

Create an HTML page with a navigation bar at the top containing links for "Home", "About", and "Contact". Below the navigation, include a main content area with a heading "Welcome to My Site" and a paragraph describing the site. The page should have a footer with copyright information.

## Answer

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
def reward_fn(model_output, reference_answer):
    from bs4 import BeautifulSoup
    
    # 1. Format Objective: Valid HTML syntax
    try:
        soup = BeautifulSoup(model_output, 'html.parser')
        format_score = 1.0
    except Exception as e:
        return (0.0, 0.0)
    
    # 2. Correctness Objective: Check required elements exist
    required_selectors = ['nav', 'a[href="/"]', 'a[href="/about"]', 'a[href="/contact"]', 'h1', 'main', 'footer']
    
    matched = 0
    for selector in required_selectors:
        try:
            if soup.select(selector):
                matched += 1
        except:
            pass
    
    # All selectors must be present for full correctness
    correctness_score = 1.0 if matched == len(required_selectors) else 0.0
    
    return (format_score, correctness_score)
```
