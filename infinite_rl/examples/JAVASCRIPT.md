[PROMPT]

Write a JavaScript function that calculates the factorial of a number and outputs the result in JSON format.

[ANSWER]

```javascript
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

console.log(JSON.stringify({ factorial: factorial(5) }));
```

[RESPONSE]

Here is the JavaScript function to calculate the factorial of a number.

<answer>
```javascript
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

console.log(JSON.stringify({ factorial: factorial(5) }));
```
</answer>
