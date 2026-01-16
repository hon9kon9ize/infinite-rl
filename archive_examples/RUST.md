[PROMPT]

Write a Rust function that finds the maximum element in a vector of integers and outputs the result in JSON format.

[ANSWER]

```rust
fn find_max(nums: Vec<i32>) -> i32 {
    *nums.iter().max().unwrap()
}

fn main() {
    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    println!("{{\"max\": {}}}", find_max(numbers));
}
```

[RESPONSE]

I have implemented the `find_max` function and a `main` function to test it.

<answer>
```rust
fn find_max(nums: Vec<i32>) -> i32 {
    *nums.iter().max().unwrap()
}

fn main() {
    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    println!("{{\"max\": {}}}", find_max(numbers));
}
```
</answer>
