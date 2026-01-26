Solve the following {language} programming puzzle. Your task is continue implement the sol function, to make it returns a value that makes the sat function return True.

# {name}

{docstring}

## Sat function

```{language}
{sat}
```

## Answer return value type

{ans_type}

## Sol header

```{language}
{sat}
```

====

Solve the following Javascript programming puzzle. Your task is continue implement the sol function, to make it returns a value that makes the sat function return True.

# AllCubicRoots

Find all 3 distinct real roots of x^3 + a x^2 + b x + c, i.e., factor into (x-r1)(x-r2)(x-r3).
coeffs = [a, b, c]. For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6,
sat(roots = [1., 2., 3.], coeffs = [-6., 11., -6.]) is True.

## Sat function

```javascript
function sat (roots, coeffs = [1.0, -2.0, -1.0]) {
      const [r1, r2, r3] = roots;
      const [a, b, c] = coeffs;
      return (
        Math.abs(r1 + r2 + r3 + a) +
        Math.abs(r1 * r2 + r1 * r3 + r2 * r3 - b) +
        Math.abs(r1 * r2 * r3 + c) <
        1e-4
      );
    }
```

## Answer return value type

List[float]

## Sol header

```javascript
function sol (coeffs)
````
