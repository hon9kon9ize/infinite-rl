## Instruction

Solve the following math problem. Provide your reasoning first, but you must place the final numerical or symbolic result at the very end of your response, enclosed in tags like this: <answer>[RESULT]</answer>. Do not include units or extra words inside the tags. Do not use LaTeX. Use only plain text characters (e.g., use x^2 instead of symbols).

## Prompt

What is the integral of the function f(x) = 2x^3 - 4x + 1 with respect to x?

## Answer

(1/2)x^4 - 2x^2 + x + C

## Response

The integral of the function f(x) = 2x^3 - 4x + 1 can be found by integrating each term separately:

∫(2x^3 - 4x + 1)dx = ∫2x^3dx - ∫4xdx + ∫1dx

Using the power rule (∫x^n dx = x^(n+1)/(n+1) + C):

= 2x^4/4 - 4x^2/2 + x + C
= (1/2)x^4 - 2x^2 + x + C

<answer>(1/2)x^4 - 2x^2 + x + C</answer>

## Reward Function

```python
def reward_fn(model_output, reference_answer):
    import re
    from sympy import simplify, parse_expr
    from sympy.parsing.sympy_parser import (
        standard_transformations,
        implicit_multiplication_application,
    )
    tag_pattern = r"<answer>(.*?)</answer>"
    match = re.search(tag_pattern, model_output, re.DOTALL)
    if not match:
        return (0.0, 0.0)
    format_score = 1.0 if match else 0.0
    predicted_str = match.group(1).strip()
    try:
        def to_sympy(text):
            text = re.sub(r"\+?\s*[cC]$", "", text).strip()
            if "\\" in text or "{" in text:
                return parse_latex(text)
            else:
                text = text.replace("^", "**")
                transformations = standard_transformations + (
                    implicit_multiplication_application,
                )
                return parse_expr(text, transformations=transformations)
        pred_expr = to_sympy(predicted_str)
        ref_expr = to_sympy(reference_answer)
        if simplify(pred_expr - ref_expr) == 0:
            correctness_score = 1.0
        else:
            correctness_score = 0.0
    except Exception:
        correctness_score = 0.0
    return (format_score, correctness_score)
```