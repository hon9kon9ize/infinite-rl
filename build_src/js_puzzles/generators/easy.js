/**
 * Easy programming puzzles - intermediate difficulty between math (level 0) and programming puzzles (level 1)
 * These puzzles introduce basic programming concepts with simpler logic than standard programming puzzles.
 */

import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

// 1. String manipulation - basic
export class ReverseString extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Reverse a given string.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello" };
  }

  static sat (x, s) {
    return x === s.split("").reverse().join("");
  }

  static sol (s) {
    return s.split("").reverse().join("");
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const s = Array.from({ length }, () =>
      String.fromCharCode(97 + Math.floor(this.random() * 26))
    ).join("");
    this.add({ s });
  }
}

export class RepeatString extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Repeat a string n times.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "abc", n: 5 };
  }

  static sat (x, s, n) {
    return x === s.repeat(n);
  }

  static sol (s, n) {
    return s.repeat(n);
  }

  genRandom () {
    const length = Math.floor(this.random() * 5) + 1;
    const s = Array.from({ length }, () =>
      String.fromCharCode(97 + Math.floor(this.random() * 26))
    ).join("");
    const n = Math.floor(this.random() * 20) + 1;
    this.add({ s, n });
  }
}

export class StringLength extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Find the length of a string.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "programming" };
  }

  static sat (x, s) {
    return x === s.length;
  }

  static sol (s) {
    return s.length;
  }

  genRandom () {
    const length = Math.floor(this.random() * 50) + 1;
    const s = Array.from({ length }, () =>
      String.fromCharCode(97 + Math.floor(this.random() * 26))
    ).join("");
    this.add({ s });
  }
}

export class Uppercase extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Convert a string to uppercase.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world" };
  }

  static sat (x, s) {
    return x === s.toUpperCase();
  }

  static sol (s) {
    return s.toUpperCase();
  }

  genRandom () {
    const length = Math.floor(this.random() * 26) + 5;
    const s = Array.from({ length }, () =>
      this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
    ).join("");
    this.add({ s });
  }
}

export class Lowercase extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Convert a string to lowercase.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "HELLO WORLD" };
  }

  static sat (x, s) {
    return x === s.toLowerCase();
  }

  static sol (s) {
    return s.toLowerCase();
  }

  genRandom () {
    const length = Math.floor(this.random() * 26) + 5;
    const s = Array.from({ length }, () =>
      this.random() > 0.2 ? String.fromCharCode(65 + Math.floor(this.random() * 26)) : " "
    ).join("");
    this.add({ s });
  }
}

// 2. List operations - basic
export class ListSum extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Find the sum of a list of integers.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5] };
  }

  static sat (x, nums) {
    return x === nums.reduce((a, b) => a + b, 0);
  }

  static sol (nums) {
    return nums.reduce((a, b) => a + b, 0);
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class ListMax extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Find the maximum value in a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [3, 7, 2, 9, 1] };
  }

  static sat (x, nums) {
    return x === Math.max(...nums);
  }

  static sol (nums) {
    return Math.max(...nums);
  }

  genRandom () {
    const length = Math.floor(this.random() * 50) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 2001) - 1000
    );
    this.add({ nums });
  }
}

export class ListMin extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Find the minimum value in a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [3, 7, 2, 9, 1] };
  }

  static sat (x, nums) {
    return x === Math.min(...nums);
  }

  static sol (nums) {
    return Math.min(...nums);
  }

  genRandom () {
    const length = Math.floor(this.random() * 50) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 2001) - 1000
    );
    this.add({ nums });
  }
}

export class ListLength extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Find the length of a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5] };
  }

  static sat (x, nums) {
    return x === nums.length;
  }

  static sol (nums) {
    return nums.length;
  }

  genRandom () {
    const length = Math.floor(this.random() * 100);
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class ReverseList extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Reverse a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === nums[nums.length - 1 - i]);
  }

  static sol (nums) {
    return [...nums].reverse();
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

// 3. Simple counting and filtering
export class CountPositive extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Count the number of positive integers in a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, -2, 3, -4, 5] };
  }

  static sat (x, nums) {
    return x === nums.filter(n => n > 0).length;
  }

  static sol (nums) {
    return nums.filter(n => n > 0).length;
  }

  genRandom () {
    const length = Math.floor(this.random() * 50) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class CountEven extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Count the number of even integers in a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5, 6] };
  }

  static sat (x, nums) {
    return x === nums.filter(n => n % 2 === 0).length;
  }

  static sol (nums) {
    return nums.filter(n => n % 2 === 0).length;
  }

  genRandom () {
    const length = Math.floor(this.random() * 50) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class CountOdd extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Count the number of odd integers in a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5, 6] };
  }

  static sat (x, nums) {
    return x === nums.filter(n => n % 2 !== 0).length;
  }

  static sol (nums) {
    return nums.filter(n => n % 2 !== 0).length;
  }

  genRandom () {
    const length = Math.floor(this.random() * 50) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class FilterPositive extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Return a list containing only positive numbers.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, -2, 3, -4, 5] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x)) return false;
    const expected = nums.filter(n => n > 0);
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (nums) {
    return nums.filter(n => n > 0);
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class FilterEven extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Return a list containing only even numbers.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5, 6] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x)) return false;
    const expected = nums.filter(n => n % 2 === 0);
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (nums) {
    return nums.filter(n => n % 2 === 0);
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

// 4. Simple arithmetic operations
export class MultiplyList extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Multiply each element in the list by k.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, 2, 3], k: 5 };
  }

  static sat (x, nums, k) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === nums[i] * k);
  }

  static sol (nums, k) {
    return nums.map(n => n * k);
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    const k = Math.floor(this.random() * 21) - 10;
    this.add({ nums, k });
  }
}

export class AddConstant extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Add k to each element in the list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, 2, 3], k: 10 };
  }

  static sat (x, nums, k) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === nums[i] + k);
  }

  static sol (nums, k) {
    return nums.map(n => n + k);
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    const k = Math.floor(this.random() * 201) - 100;
    this.add({ nums, k });
  }
}

export class SquareList extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Square each element in the list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, 2, 3, 4] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === nums[i] * nums[i]);
  }

  static sol (nums) {
    return nums.map(n => n * n);
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 41) - 20
    );
    this.add({ nums });
  }
}

export class AbsoluteValues extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Return the absolute value of each element.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [-1, 2, -3, 4] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === Math.abs(nums[i]));
  }

  static sol (nums) {
    return nums.map(n => Math.abs(n));
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

// 5. Range operations
export class CreateRange extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Create a list of integers from 0 to n-1.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { n: 10 };
  }

  static sat (x, n) {
    if (!Array.isArray(x) || x.length !== n) return false;
    return x.every((val, i) => val === i);
  }

  static sol (n) {
    return Array.from({ length: n }, (_, i) => i);
  }

  genRandom () {
    const n = Math.floor(this.random() * 100) + 1;
    this.add({ n });
  }
}

export class RangeWithStart extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Create a list of integers from start to end-1.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { start: 5, end: 15 };
  }

  static sat (x, start, end) {
    if (!Array.isArray(x) || x.length !== end - start) return false;
    return x.every((val, i) => val === start + i);
  }

  static sol (start, end) {
    return Array.from({ length: end - start }, (_, i) => start + i);
  }

  genRandom () {
    const start = Math.floor(this.random() * 101) - 50;
    const end = start + Math.floor(this.random() * 100) + 1;
    this.add({ start, end });
  }
}

export class RangeWithStep extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Create a list of integers from start to end-1 with step.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { start: 0, end: 20, step: 3 };
  }

  static sat (x, start, end, step) {
    if (!Array.isArray(x)) return false;
    const expected = [];
    for (let i = start; i < end; i += step) {
      expected.push(i);
    }
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (start, end, step) {
    const result = [];
    for (let i = start; i < end; i += step) {
      result.push(i);
    }
    return result;
  }

  genRandom () {
    const start = Math.floor(this.random() * 101) - 50;
    const step = Math.floor(this.random() * 10) + 1;
    const end = start + step + Math.floor(this.random() * 100);
    this.add({ start, end, step });
  }
}

// 6. String operations
export class JoinStrings extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Join a list of strings with a separator.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { words: ["hello", "world"], sep: " " };
  }

  static sat (x, words, sep) {
    return x === words.join(sep);
  }

  static sol (words, sep) {
    return words.join(sep);
  }

  genRandom () {
    const numWords = Math.floor(this.random() * 10) + 1;
    const words = Array.from({ length: numWords }, () => {
      const len = Math.floor(this.random() * 10) + 1;
      return Array.from({ length: len }, () =>
        String.fromCharCode(97 + Math.floor(this.random() * 26))
      ).join("");
    });
    const separators = [" ", ",", "-", ""];
    const sep = separators[Math.floor(this.random() * separators.length)];
    this.add({ words, sep });
  }
}

export class SplitString extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Split a string by a separator.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world how are you", sep: " " };
  }

  static sat (x, s, sep) {
    if (!Array.isArray(x)) return false;
    const expected = s.split(sep);
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (s, sep) {
    return s.split(sep);
  }

  genRandom () {
    const separators = [" ", ",", "-"];
    const sep = separators[Math.floor(this.random() * separators.length)];
    const numWords = Math.floor(this.random() * 10) + 1;
    const words = Array.from({ length: numWords }, () => {
      const len = Math.floor(this.random() * 10) + 1;
      return Array.from({ length: len }, () =>
        String.fromCharCode(97 + Math.floor(this.random() * 26))
      ).join("");
    });
    const s = words.join(sep);
    this.add({ s, sep });
  }
}

export class CountSubstring extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Count occurrences of a substring.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello hello world", sub: "hello" };
  }

  static sat (x, s, sub) {
    const count = (s.match(new RegExp(sub.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
    return x === count;
  }

  static sol (s, sub) {
    return (s.match(new RegExp(sub.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
  }

  genRandom () {
    const subLen = Math.floor(this.random() * 5) + 1;
    const sub = Array.from({ length: subLen }, () =>
      String.fromCharCode(97 + Math.floor(this.random() * 26))
    ).join("");

    const baseLen = Math.floor(this.random() * 21) + 10;
    const base = Array.from({ length: baseLen }, () =>
      String.fromCharCode(97 + Math.floor(this.random() * 26))
    ).join("");

    const parts = [];
    for (let i = 0; i < base.length; i += 5) {
      parts.push(base.substring(i, i + 5));
    }

    const insertCount = Math.floor(this.random() * 4);
    for (let i = 0; i < insertCount; i++) {
      const pos = Math.floor(this.random() * (parts.length + 1));
      parts.splice(pos, 0, sub);
    }

    const s = parts.join("");
    this.add({ s, sub });
  }
}

export class ReplaceString extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Replace all occurrences of old with new.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world", old: "world", new: "python" };
  }

  static sat (x, s, old, newStr) {
    return x === s.split(old).join(newStr);
  }

  static sol (s, old, newStr) {
    return s.split(old).join(newStr);
  }

  genRandom () {
    const oldLen = Math.floor(this.random() * 4) + 2;
    const old = Array.from({ length: oldLen }, () =>
      String.fromCharCode(97 + Math.floor(this.random() * 26))
    ).join("");

    const newLen = Math.floor(this.random() * 4) + 2;
    const newStr = Array.from({ length: newLen }, () =>
      String.fromCharCode(97 + Math.floor(this.random() * 26))
    ).join("");

    const baseLen = Math.floor(this.random() * 21) + 10;
    const parts = Array.from({ length: baseLen }, () =>
      this.random() > 0.3 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
    ).join("").split(" ");

    for (let i = 0; i < parts.length; i++) {
      if (this.random() < 0.3) {
        parts[i] = old;
      }
    }

    const s = parts.join(" ");
    this.add({ s, old, new: newStr });
  }
}

export class StartsWith extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if string starts with prefix.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world", prefix: "hello" };
  }

  static sat (x, s, prefix) {
    return x === s.startsWith(prefix);
  }

  static sol (s, prefix) {
    return s.startsWith(prefix);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // True case
      const prefixLen = Math.floor(this.random() * 10) + 1;
      const prefix = Array.from({ length: prefixLen }, () =>
        String.fromCharCode(97 + Math.floor(this.random() * 26))
      ).join("");

      const restLen = Math.floor(this.random() * 16) + 5;
      const rest = Array.from({ length: restLen }, () =>
        this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
      ).join("");

      const s = prefix + rest;
      this.add({ s, prefix });
    } else {
      // False case
      const sLen = Math.floor(this.random() * 21) + 10;
      const s = Array.from({ length: sLen }, () =>
        this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
      ).join("");

      const prefixLen = Math.floor(this.random() * 10) + 1;
      const prefix = Array.from({ length: prefixLen }, () =>
        String.fromCharCode(97 + Math.floor(this.random() * 26))
      ).join("");

      this.add({ s, prefix });
    }
  }
}

export class EndsWith extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if string ends with suffix.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world", suffix: "world" };
  }

  static sat (x, s, suffix) {
    return x === s.endsWith(suffix);
  }

  static sol (s, suffix) {
    return s.endsWith(suffix);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // True case
      const suffixLen = Math.floor(this.random() * 10) + 1;
      const suffix = Array.from({ length: suffixLen }, () =>
        String.fromCharCode(97 + Math.floor(this.random() * 26))
      ).join("");

      const restLen = Math.floor(this.random() * 16) + 5;
      const rest = Array.from({ length: restLen }, () =>
        this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
      ).join("");

      const s = rest + suffix;
      this.add({ s, suffix });
    } else {
      // False case
      const sLen = Math.floor(this.random() * 21) + 10;
      const s = Array.from({ length: sLen }, () =>
        this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
      ).join("");

      const suffixLen = Math.floor(this.random() * 10) + 1;
      const suffix = Array.from({ length: suffixLen }, () =>
        String.fromCharCode(97 + Math.floor(this.random() * 26))
      ).join("");

      this.add({ s, suffix });
    }
  }
}

// 7. Simple logic
export class IsEven extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Check if a number is even.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { n: 42 };
  }

  static sat (x, n) {
    return x === (n % 2 === 0);
  }

  static sol (n) {
    return n % 2 === 0;
  }

  genRandom () {
    const n = Math.floor(this.random() * 2001) - 1000;
    this.add({ n });
  }
}

export class IsPositive extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Check if a number is positive.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { n: 5 };
  }

  static sat (x, n) {
    return x === (n > 0);
  }

  static sol (n) {
    return n > 0;
  }

  genRandom () {
    const n = Math.floor(this.random() * 2001) - 1000;
    this.add({ n });
  }
}

export class InRange extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Check if n is in range [low, high).";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { n: 15, low: 10, high: 20 };
  }

  static sat (x, n, low, high) {
    return x === (low <= n && n < high);
  }

  static sol (n, low, high) {
    return low <= n && n < high;
  }

  genRandom () {
    const low = Math.floor(this.random() * 201) - 100;
    const high = low + Math.floor(this.random() * 100) + 1;
    const n = low - 10 + Math.floor(this.random() * (high - low + 20));
    this.add({ n, low, high });
  }
}

// 8. List indexing and slicing
export class FirstElement extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Get the first element of a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [5, 2, 8, 1, 9] };
  }

  static sat (x, nums) {
    return x === nums[0];
  }

  static sol (nums) {
    return nums[0];
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class LastElement extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Get the last element of a list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [5, 2, 8, 1, 9] };
  }

  static sat (x, nums) {
    return x === nums[nums.length - 1];
  }

  static sol (nums) {
    return nums[nums.length - 1];
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class GetSlice extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Get a slice of the list from start to end.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5], start: 1, end: 4 };
  }

  static sat (x, nums, start, end) {
    if (!Array.isArray(x)) return false;
    const expected = nums.slice(start, end);
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (nums, start, end) {
    return nums.slice(start, end);
  }

  genRandom () {
    const length = Math.floor(this.random() * 26) + 5;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    const start = Math.floor(this.random() * (length - 2));
    const end = start + 1 + Math.floor(this.random() * (length - start));
    this.add({ nums, start, end });
  }
}

export class EveryNth extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Get every nth element from the list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5, 6, 7, 8], n: 2 };
  }

  static sat (x, nums, n) {
    if (!Array.isArray(x)) return false;
    const expected = nums.filter((_, i) => i % n === 0);
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (nums, n) {
    return nums.filter((_, i) => i % n === 0);
  }

  genRandom () {
    const length = Math.floor(this.random() * 26) + 5;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    const n = Math.floor(this.random() * 4) + 2;
    this.add({ nums, n });
  }
}

// 9. List comprehension variations
export class DoubleEvens extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Double only the even numbers in the list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5, 6] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === (nums[i] % 2 === 0 ? nums[i] * 2 : nums[i]));
  }

  static sol (nums) {
    return nums.map(n => n % 2 === 0 ? n * 2 : n);
  }

  genRandom () {
    const length = Math.floor(this.random() * 21) + 5;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    this.add({ nums });
  }
}

export class ZeroNegatives extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Replace negative numbers with 0.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, -2, 3, -4, 5] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === (nums[i] >= 0 ? nums[i] : 0));
  }

  static sol (nums) {
    return nums.map(n => n >= 0 ? n : 0);
  }

  genRandom () {
    const length = Math.floor(this.random() * 21) + 5;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class ClampValues extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Clamp values to be within [min_val, max_val].";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 15, 8, 25, 3], min_val: 5, max_val: 20 };
  }

  static sat (x, nums, min_val, max_val) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    return x.every((val, i) => val === Math.max(min_val, Math.min(max_val, nums[i])));
  }

  static sol (nums, min_val, max_val) {
    return nums.map(n => Math.max(min_val, Math.min(max_val, n)));
  }

  genRandom () {
    const min_val = Math.floor(this.random() * 101) - 50;
    const max_val = min_val + 10 + Math.floor(this.random() * 100);
    const length = Math.floor(this.random() * 21) + 5;
    const nums = Array.from({ length }, () =>
      min_val - 50 + Math.floor(this.random() * (max_val - min_val + 100))
    );
    this.add({ nums, min_val, max_val });
  }
}

// 10. Simple set operations
export class UniqueElements extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Return list with unique elements (order preserved).";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 2, 3, 3, 3, 4] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x)) return false;
    const seen = new Set();
    const expected = [];
    for (const n of nums) {
      if (!seen.has(n)) {
        seen.add(n);
        expected.push(n);
      }
    }
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (nums) {
    const seen = new Set();
    const result = [];
    for (const n of nums) {
      if (!seen.has(n)) {
        seen.add(n);
        result.push(n);
      }
    }
    return result;
  }

  genRandom () {
    const baseLength = Math.floor(this.random() * 8) + 3;
    const base = Array.from({ length: baseLength }, () =>
      Math.floor(this.random() * 41) - 20
    );
    const numsLength = Math.floor(this.random() * 21) + 10;
    const nums = Array.from({ length: numsLength }, () =>
      base[Math.floor(this.random() * base.length)]
    );
    this.add({ nums });
  }
}

export class CountUnique extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Count the number of unique elements.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 2, 3, 3, 3, 4] };
  }

  static sat (x, nums) {
    return x === new Set(nums).size;
  }

  static sol (nums) {
    return new Set(nums).size;
  }

  genRandom () {
    const baseLength = Math.floor(this.random() * 8) + 3;
    const base = Array.from({ length: baseLength }, () =>
      Math.floor(this.random() * 41) - 20
    );
    const numsLength = Math.floor(this.random() * 21) + 10;
    const nums = Array.from({ length: numsLength }, () =>
      base[Math.floor(this.random() * base.length)]
    );
    this.add({ nums });
  }
}

export class HasDuplicates extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if the list has any duplicates.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5] };
  }

  static sat (x, nums) {
    return x === (nums.length !== new Set(nums).size);
  }

  static sol (nums) {
    return nums.length !== new Set(nums).size;
  }

  genRandom () {
    if (this.random() < 0.5) {
      // With duplicates
      const baseLength = Math.floor(this.random() * 8) + 3;
      const base = Array.from({ length: baseLength }, () =>
        Math.floor(this.random() * 41) - 20
      );
      const numsLength = Math.floor(this.random() * 21) + 10;
      const nums = Array.from({ length: numsLength }, () =>
        base[Math.floor(this.random() * base.length)]
      );
      this.add({ nums });
    } else {
      // Without duplicates
      const length = Math.floor(this.random() * 26) + 5;
      const nums = Array.from({ length }, (_, i) => i);
      // Shuffle
      for (let i = nums.length - 1; i > 0; i--) {
        const j = Math.floor(this.random() * (i + 1));
        [nums[i], nums[j]] = [nums[j], nums[i]];
      }
      this.add({ nums });
    }
  }
}

// 11. Simple tuple/list operations
export class PairElements extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Pair consecutive elements.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length - 1) return false;
    return x.every((pair, i) => {
      return Array.isArray(pair) && pair.length === 2 &&
        pair[0] === nums[i] && pair[1] === nums[i + 1];
    });
  }

  static sol (nums) {
    return nums.slice(0, -1).map((val, i) => [val, nums[i + 1]]);
  }

  genRandom () {
    const length = Math.floor(this.random() * 19) + 2;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    this.add({ nums });
  }
}

export class ZipLists extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Zip two lists together.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { a: [1, 2, 3], b: [4, 5, 6] };
  }

  static sat (x, a, b) {
    if (!Array.isArray(x) || x.length !== a.length) return false;
    return x.every((pair, i) => {
      return Array.isArray(pair) && pair.length === 2 &&
        pair[0] === a[i] && pair[1] === b[i];
    });
  }

  static sol (a, b) {
    return a.map((val, i) => [val, b[i]]);
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const a = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    const b = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    this.add({ a, b });
  }
}

export class SumPairs extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Add corresponding elements from two lists.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { a: [1, 2, 3], b: [4, 5, 6] };
  }

  static sat (x, a, b) {
    if (!Array.isArray(x) || x.length !== a.length) return false;
    return x.every((val, i) => val === a[i] + b[i]);
  }

  static sol (a, b) {
    return a.map((val, i) => val + b[i]);
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const a = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    const b = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    this.add({ a, b });
  }
}

// 12. Simple boolean operations
export class AllPositive extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if all numbers are positive.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5] };
  }

  static sat (x, nums) {
    return x === nums.every(n => n > 0);
  }

  static sol (nums) {
    return nums.every(n => n > 0);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // All positive
      const length = Math.floor(this.random() * 20) + 1;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 100) + 1
      );
      this.add({ nums });
    } else {
      // At least one non-positive
      const length = Math.floor(this.random() * 20) + 1;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 201) - 100
      );
      this.add({ nums });
    }
  }
}

export class AnyNegative extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if any number is negative.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, -3, 4, 5] };
  }

  static sat (x, nums) {
    return x === nums.some(n => n < 0);
  }

  static sol (nums) {
    return nums.some(n => n < 0);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // Has negative
      const length = Math.floor(this.random() * 20) + 1;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 201) - 100
      );
      const idx = Math.floor(this.random() * nums.length);
      nums[idx] = Math.floor(this.random() * 100) - 100;
      this.add({ nums });
    } else {
      // All non-negative
      const length = Math.floor(this.random() * 20) + 1;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 101)
      );
      this.add({ nums });
    }
  }
}

export class AllEven extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if all numbers are even.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [2, 4, 6, 8] };
  }

  static sat (x, nums) {
    return x === nums.every(n => n % 2 === 0);
  }

  static sol (nums) {
    return nums.every(n => n % 2 === 0);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // All even
      const length = Math.floor(this.random() * 20) + 1;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 101 - 50) * 2
      );
      this.add({ nums });
    } else {
      // At least one odd
      const length = Math.floor(this.random() * 20) + 1;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 201) - 100
      );
      this.add({ nums });
    }
  }
}

// 13. Character/ASCII operations
export class CharToAscii extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Get the ASCII value of a character.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { c: "A" };
  }

  static sat (x, c) {
    return x === c.charCodeAt(0);
  }

  static sol (c) {
    return c.charCodeAt(0);
  }

  genRandom () {
    const c = String.fromCharCode(32 + Math.floor(this.random() * 95));
    this.add({ c });
  }
}

export class AsciiToChar extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Convert ASCII value to character.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { n: 65 };
  }

  static sat (x, n) {
    return x === String.fromCharCode(n);
  }

  static sol (n) {
    return String.fromCharCode(n);
  }

  genRandom () {
    const n = 32 + Math.floor(this.random() * 95);
    this.add({ n });
  }
}

export class IsLetter extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if character is a letter.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { c: "a" };
  }

  static sat (x, c) {
    return x === /[a-zA-Z]/.test(c);
  }

  static sol (c) {
    return /[a-zA-Z]/.test(c);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // Letter
      const c = String.fromCharCode(97 + Math.floor(this.random() * 26));
      this.add({ c });
    } else {
      // Digit
      const c = String.fromCharCode(48 + Math.floor(this.random() * 10));
      this.add({ c });
    }
  }
}

export class IsDigit extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if character is a digit.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { c: "5" };
  }

  static sat (x, c) {
    return x === /[0-9]/.test(c);
  }

  static sol (c) {
    return /[0-9]/.test(c);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // Digit
      const c = String.fromCharCode(48 + Math.floor(this.random() * 10));
      this.add({ c });
    } else {
      // Letter
      const c = String.fromCharCode(97 + Math.floor(this.random() * 26));
      this.add({ c });
    }
  }
}

// 14. Simple sorting operations
export class SortList extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Sort a list in ascending order.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [3, 1, 4, 1, 5, 9, 2, 6] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    const sorted = [...nums].sort((a, b) => a - b);
    return x.every((val, i) => val === sorted[i]);
  }

  static sol (nums) {
    return [...nums].sort((a, b) => a - b);
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class SortDescending extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Sort a list in descending order.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [3, 1, 4, 1, 5, 9, 2, 6] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    const sorted = [...nums].sort((a, b) => b - a);
    return x.every((val, i) => val === sorted[i]);
  }

  static sol (nums) {
    return [...nums].sort((a, b) => b - a);
  }

  genRandom () {
    const length = Math.floor(this.random() * 30) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 201) - 100
    );
    this.add({ nums });
  }
}

export class IsSorted extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if list is sorted in ascending order.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5] };
  }

  static sat (x, nums) {
    const sorted = [...nums].sort((a, b) => a - b);
    return x === nums.every((val, i) => val === sorted[i]);
  }

  static sol (nums) {
    const sorted = [...nums].sort((a, b) => a - b);
    return nums.every((val, i) => val === sorted[i]);
  }

  genRandom () {
    if (this.random() < 0.5) {
      // Sorted
      const length = Math.floor(this.random() * 20) + 1;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 201) - 100
      ).sort((a, b) => a - b);
      this.add({ nums });
    } else {
      // Not sorted
      const length = Math.floor(this.random() * 19) + 2;
      const nums = Array.from({ length }, () =>
        Math.floor(this.random() * 201) - 100
      );
      this.add({ nums });
    }
  }
}

// 15. Simple find operations
export class FindIndex extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Find the index of target in the list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5], target: 3 };
  }

  static sat (x, nums, target) {
    return x === nums.indexOf(target);
  }

  static sol (nums, target) {
    return nums.indexOf(target);
  }

  genRandom () {
    const length = Math.floor(this.random() * 21) + 5;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    const target = nums[Math.floor(this.random() * nums.length)];
    this.add({ nums, target });
  }
}

export class Contains extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Check if target is in the list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5], target: 3 };
  }

  static sat (x, nums, target) {
    return x === nums.includes(target);
  }

  static sol (nums, target) {
    return nums.includes(target);
  }

  genRandom () {
    const length = Math.floor(this.random() * 21) + 5;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    let target;
    if (this.random() < 0.5) {
      target = nums[Math.floor(this.random() * nums.length)];
    } else {
      target = Math.floor(this.random() * 201) - 100;
    }
    this.add({ nums, target });
  }
}

export class CountOccurrences extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Count how many times target appears in the list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 2, 3, 2, 4], target: 2 };
  }

  static sat (x, nums, target) {
    return x === nums.filter(n => n === target).length;
  }

  static sol (nums, target) {
    return nums.filter(n => n === target).length;
  }

  genRandom () {
    const baseLength = Math.floor(this.random() * 8) + 3;
    const base = Array.from({ length: baseLength }, () =>
      Math.floor(this.random() * 41) - 20
    );
    const numsLength = Math.floor(this.random() * 21) + 10;
    const nums = Array.from({ length: numsLength }, () =>
      base[Math.floor(this.random() * base.length)]
    );
    const target = base[Math.floor(this.random() * base.length)];
    this.add({ nums, target });
  }
}

// 16. Simple number operations
export class Factorial extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Calculate factorial of n.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { n: 5 };
  }

  static sat (x, n) {
    let result = 1;
    for (let i = 1; i <= n; i++) {
      result *= i;
    }
    return x === result;
  }

  static sol (n) {
    let result = 1;
    for (let i = 1; i <= n; i++) {
      result *= i;
    }
    return result;
  }

  genRandom () {
    const n = Math.floor(this.random() * 16);
    this.add({ n });
  }
}

export class Power extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Calculate base raised to the power of exp.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { base: 2, exp: 10 };
  }

  static sat (x, base, exp) {
    return x === Math.pow(base, exp);
  }

  static sol (base, exp) {
    return Math.pow(base, exp);
  }

  genRandom () {
    const base = Math.floor(this.random() * 21) - 10;
    const exp = Math.floor(this.random() * 11);
    this.add({ base, exp });
  }
}

export class Modulo extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Calculate a modulo b.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { a: 17, b: 5 };
  }

  static sat (x, a, b) {
    // Match Python's modulo behavior (assuming b > 0 as per gen_random)
    // Python: -17 % 5 = 3 (result has same sign as divisor)
    // JavaScript: -17 % 5 = -2 (result has same sign as dividend)
    // Convert JS to Python behavior: ((a % b) + b) % b
    const pyMod = ((a % b) + b) % b;
    return x === pyMod;
  }

  static sol (a, b) {
    // Match Python's modulo behavior (for positive b)
    return ((a % b) + b) % b;
  }

  genRandom () {
    const b = Math.floor(this.random() * 100) + 1;
    const a = Math.floor(this.random() * 2001) - 1000;
    this.add({ a, b });
  }
}

export class IntegerDivision extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Calculate integer division a // b.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { a: 17, b: 5 };
  }

  static sat (x, a, b) {
    return x === Math.floor(a / b);
  }

  static sol (a, b) {
    return Math.floor(a / b);
  }

  genRandom () {
    const b = Math.floor(this.random() * 100) + 1;
    const a = Math.floor(this.random() * 2001) - 1000;
    this.add({ a, b });
  }
}

// 17. String character operations
export class CountVowels extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Count the number of vowels in a string.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world" };
  }

  static sat (x, s) {
    const vowels = "aeiouAEIOU";
    return x === s.split("").filter(c => vowels.includes(c)).length;
  }

  static sol (s) {
    const vowels = "aeiouAEIOU";
    return s.split("").filter(c => vowels.includes(c)).length;
  }

  genRandom () {
    const length = Math.floor(this.random() * 31) + 10;
    const s = Array.from({ length }, () =>
      this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
    ).join("");
    this.add({ s });
  }
}

export class CountConsonants extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Count the number of consonants in a string.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world" };
  }

  static sat (x, s) {
    const vowels = "aeiouAEIOU";
    return x === s.split("").filter(c => /[a-zA-Z]/.test(c) && !vowels.includes(c)).length;
  }

  static sol (s) {
    const vowels = "aeiouAEIOU";
    return s.split("").filter(c => /[a-zA-Z]/.test(c) && !vowels.includes(c)).length;
  }

  genRandom () {
    const length = Math.floor(this.random() * 31) + 10;
    const s = Array.from({ length }, () =>
      this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
    ).join("");
    this.add({ s });
  }
}

export class RemoveSpaces extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Remove all spaces from a string.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "hello world  test" };
  }

  static sat (x, s) {
    return x === s.replace(/ /g, "");
  }

  static sol (s) {
    return s.replace(/ /g, "");
  }

  genRandom () {
    const length = Math.floor(this.random() * 31) + 10;
    const s = Array.from({ length }, () =>
      this.random() > 0.3 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
    ).join("");
    this.add({ s });
  }
}

export class StripWhitespace extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Remove leading and trailing whitespace.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { s: "  hello world  " };
  }

  static sat (x, s) {
    return x === s.trim();
  }

  static sol (s) {
    return s.trim();
  }

  genRandom () {
    const spacesBefore = " ".repeat(Math.floor(this.random() * 11));
    const spacesAfter = " ".repeat(Math.floor(this.random() * 11));
    const middleLen = Math.floor(this.random() * 16) + 5;
    const middle = Array.from({ length: middleLen }, () =>
      this.random() > 0.2 ? String.fromCharCode(97 + Math.floor(this.random() * 26)) : " "
    ).join("");
    const s = spacesBefore + middle + spacesAfter;
    this.add({ s });
  }
}

// 18. List construction
export class RepeatElements extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Repeat each element in the list times.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nums: [1, 2, 3], times: 3 };
  }

  static sat (x, nums, times) {
    if (!Array.isArray(x)) return false;
    const expected = nums.flatMap(n => Array(times).fill(n));
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (nums, times) {
    return nums.flatMap(n => Array(times).fill(n));
  }

  genRandom () {
    const length = Math.floor(this.random() * 10) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    const times = Math.floor(this.random() * 5) + 1;
    this.add({ nums, times });
  }
}

export class FlattenOnce extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Flatten a nested list by one level.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { nested: [[1, 2], [3, 4], [5]] };
  }

  static sat (x, nested) {
    if (!Array.isArray(x)) return false;
    const expected = nested.flat();
    return x.length === expected.length && x.every((val, i) => val === expected[i]);
  }

  static sol (nested) {
    return nested.flat();
  }

  genRandom () {
    const numSublists = Math.floor(this.random() * 10) + 1;
    const nested = Array.from({ length: numSublists }, () => {
      const subLen = Math.floor(this.random() * 6);
      return Array.from({ length: subLen }, () =>
        Math.floor(this.random() * 101) - 50
      );
    });
    this.add({ nested });
  }
}

export class AlternateElements extends PuzzleGenerator {
  static tags = [Tags.trivial];
  static docstring = "Alternate elements from two lists.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.trivial];
  }

  getExample () {
    return { a: [1, 2, 3], b: [4, 5, 6] };
  }

  static sat (x, a, b) {
    if (!Array.isArray(x) || x.length !== a.length + b.length) return false;
    const expected = [];
    for (let i = 0; i < a.length; i++) {
      expected.push(a[i]);
      expected.push(b[i]);
    }
    return x.every((val, i) => val === expected[i]);
  }

  static sol (a, b) {
    const result = [];
    for (let i = 0; i < a.length; i++) {
      result.push(a[i]);
      result.push(b[i]);
    }
    return result;
  }

  genRandom () {
    const length = Math.floor(this.random() * 15) + 1;
    const a = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    const b = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    this.add({ a, b });
  }
}

// 19. Simple accumulation
export class CumulativeSum extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Create cumulative sum list.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [1, 2, 3, 4, 5] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    let total = 0;
    const expected = [];
    for (const n of nums) {
      total += n;
      expected.push(total);
    }
    return x.every((val, i) => val === expected[i]);
  }

  static sol (nums) {
    const result = [];
    let total = 0;
    for (const n of nums) {
      total += n;
      result.push(total);
    }
    return result;
  }

  genRandom () {
    const length = Math.floor(this.random() * 20) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 41) - 20
    );
    this.add({ nums });
  }
}

export class RunningMax extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Create list of running maximum values.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [3, 1, 4, 1, 5, 9, 2] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    const expected = [];
    let currentMax = nums[0];
    for (const n of nums) {
      currentMax = Math.max(currentMax, n);
      expected.push(currentMax);
    }
    return x.every((val, i) => val === expected[i]);
  }

  static sol (nums) {
    const result = [];
    let currentMax = nums[0];
    for (const n of nums) {
      currentMax = Math.max(currentMax, n);
      result.push(currentMax);
    }
    return result;
  }

  genRandom () {
    const length = Math.floor(this.random() * 25) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    this.add({ nums });
  }
}

export class RunningMin extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Create list of running minimum values.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { nums: [3, 1, 4, 1, 5, 9, 2] };
  }

  static sat (x, nums) {
    if (!Array.isArray(x) || x.length !== nums.length) return false;
    const expected = [];
    let currentMin = nums[0];
    for (const n of nums) {
      currentMin = Math.min(currentMin, n);
      expected.push(currentMin);
    }
    return x.every((val, i) => val === expected[i]);
  }

  static sol (nums) {
    const result = [];
    let currentMin = nums[0];
    for (const n of nums) {
      currentMin = Math.min(currentMin, n);
      result.push(currentMin);
    }
    return result;
  }

  genRandom () {
    const length = Math.floor(this.random() * 25) + 1;
    const nums = Array.from({ length }, () =>
      Math.floor(this.random() * 101) - 50
    );
    this.add({ nums });
  }
}

// 20. Simple comparisons
export class MaxOfTwo extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Return the maximum of two numbers.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { a: 5, b: 8 };
  }

  static sat (x, a, b) {
    return x === Math.max(a, b);
  }

  static sol (a, b) {
    return Math.max(a, b);
  }

  genRandom () {
    const a = Math.floor(this.random() * 201) - 100;
    const b = Math.floor(this.random() * 201) - 100;
    this.add({ a, b });
  }
}

export class MinOfTwo extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Return the minimum of two numbers.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { a: 5, b: 8 };
  }

  static sat (x, a, b) {
    return x === Math.min(a, b);
  }

  static sol (a, b) {
    return Math.min(a, b);
  }

  genRandom () {
    const a = Math.floor(this.random() * 201) - 100;
    const b = Math.floor(this.random() * 201) - 100;
    this.add({ a, b });
  }
}

export class ThreeWayMax extends PuzzleGenerator {
  static tags = [Tags.math];
  static docstring = "Return the maximum of three numbers.";

  constructor(seed = null) {
    super(seed);
    this.tags = [Tags.math];
  }

  getExample () {
    return { a: 5, b: 8, c: 3 };
  }

  static sat (x, a, b, c) {
    return x === Math.max(a, b, c);
  }

  static sol (a, b, c) {
    return Math.max(a, b, c);
  }

  genRandom () {
    const a = Math.floor(this.random() * 201) - 100;
    const b = Math.floor(this.random() * 201) - 100;
    const c = Math.floor(this.random() * 201) - 100;
    this.add({ a, b, c });
  }
}
