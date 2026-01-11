[PROMPT]

Write a C++ program that checks if a number is prime and outputs the result in JSON format.

[ANSWER]

```cpp
#include <iostream>
using namespace std;

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    cout << "{\"is_prime\": " << isPrime(17) << "}" << endl;
    return 0;
}
```

[RESPONSE]

I have implemented the primality test in C++.

<answer>
```cpp
#include <iostream>
using namespace std;

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    cout << "{\"is_prime\": " << isPrime(17) << "}" << endl;
    return 0;
}
```
</answer>
