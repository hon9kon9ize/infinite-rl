[PROMPT]

Write a Python function that filters a list to keep only even numbers.

[ANSWER]

```python
def filter_even(numbers):
    return [n for n in numbers if n % 2 == 0]

result = filter_even([1, 2, 3, 4, 5, 6, 7, 8])
import json
print(json.dumps({"result": result}))
```

[RESPONSE]

Here is the implementation of the `filter_even` function and a test case that outputs the result in JSON format.

<answer>
```python
def filter_even(numbers):
    return [n for n in numbers if n % 2 == 0]

result = filter_even([1, 2, 3, 4, 5, 6, 7, 8])
import json
print(json.dumps({"result": result}))
```
</answer>

