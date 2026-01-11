[PROMPT]

Write a TypeScript function that reverses a string and outputs the result in JSON format.

[ANSWER]

```typescript
function reverseString(str: string): string {
    return str.split('').reverse().join('');
}

const result = reverseString('hello');
console.log(JSON.stringify({ reversed: result }));
```

[RESPONSE]

Here is the TypeScript implementation to reverse a string and output it as JSON.

<answer>
```typescript
function reverseString(str: string): string {
    return str.split('').reverse().join('');
}

const result = reverseString('hello');
console.log(JSON.stringify({ reversed: result }));
```
</answer>
