[PROMPT]

Create an HTML page with a navigation bar at the top containing links for "Home", "About", and "Contact". Below the navigation, include a main content area with a heading "Welcome to My Site" and a paragraph describing the site. The page should have a footer with copyright information.

[ANSWER]

```json
{
  "selectors": ["nav", "a[href=\"/\"]", "a[href=\"/about\"]", "a[href=\"/contact\"]", "h1", "main", "footer"]
}
```

[RESPONSE]

<answer>
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
</answer>
