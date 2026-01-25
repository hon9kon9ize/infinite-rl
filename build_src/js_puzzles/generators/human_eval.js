import { PuzzleGenerator } from "../puzzle_generator.js";

/**
 * Problems inspired by HumanEval dataset (https://github.com/openai/human-eval)
 * described in the codex paper (https://arxiv.org/abs/2107.03374)
 *
 * Porting first 20 classes from Python human_eval.py
 */

export class FindCloseElements extends PuzzleGenerator {
	static docstring = "* Given a list of numbers, find the two closest distinct numbers in the list.\n\n        Sample Input:\n        [1.2, 5.23, 0.89, 21.0, 5.28, 1.2]\n\n        Sample Output:\n        [5.23, 5.28]";

	/** Inspired by HumanEval #0 */
	/**
	 * Given a list of numbers, find the two closest distinct numbers in the list.
	 *
	 * Sample Input:
	 * [1.2, 5.23, 0.89, 21.0, 5.28, 1.2]
	 *
	 * Sample Output:
	 * [5.23, 5.28]
	 */
	static sat (
		pair,
		nums = [0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8],
	) {
		const [a, b] = pair;
		if (a === b) return false;
		const uniqueNums = [...new Set(nums)];
		if (!uniqueNums.includes(a) || !uniqueNums.includes(b)) return false;
		const minDiff = Math.min(
			...uniqueNums.flatMap((x, i) =>
				uniqueNums.slice(i + 1).map((y) => Math.abs(x - y)),
			),
		);
		return Math.abs(a - b) === minDiff;
	}

	static sol (nums) {
		const s = [...new Set(nums)].sort((a, b) => a - b);
		let best = [s[0], s[1]];
		let bestDiff = s[1] - s[0];
		for (let i = 1; i < s.length - 1; i++) {
			const diff = s[i + 1] - s[i];
			if (diff < bestDiff) {
				bestDiff = diff;
				best = [s[i], s[i + 1]];
			}
		}
		return best;
	}

	getExample () {
		return {
			nums: [0.17, 21.3, 5.0, 9.0, 11.0, 4.99, 17.0, 17.0, 12.4, 6.8],
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(2, 10) }, () =>
			this.random.uniform(-10, 10),
		);
		nums.push(this.random.choice(nums));
		this.random.shuffle(nums);
		this.add({ nums });
	}
}

export class SeparateParenGroups extends PuzzleGenerator {
	static docstring = "* Given a string consisting of whitespace and groups of matched parentheses, split it\n        into groups of perfectly matched parentheses without any whitespace.\n\n        Sample Input:\n        '( ()) ((()()())) (()) ()'\n\n        Sample Output:\n        ['(())', '((()()()))', '(())', '()']";

	/** Inspired by HumanEval #1 */
	/**
	 * Given a string consisting of whitespace and groups of matched parentheses, split it
	 * into groups of perfectly matched parentheses without any whitespace.
	 *
	 * Sample Input:
	 * '( ()) ((()()())) (()) ()'
	 *
	 * Sample Output:
	 * ['(())', '((()()()))', '(())', '()']
	 */
	static sat (ls, combined = "() (()) ((() () ())) (() )") {
		for (const s of ls) {
			const opens = (s.match(/\(/g) || []).length;
			const closes = (s.match(/\)/g) || []).length;
			if (opens !== closes) return false;
			// Check s is not further divisible
			for (let i = 1; i < s.length; i++) {
				const prefix = s.slice(0, i);
				const prefixOpens = (prefix.match(/\(/g) || []).length;
				const prefixCloses = (prefix.match(/\)/g) || []).length;
				if (prefixOpens <= prefixCloses) return false;
			}
		}
		return ls.join("") === combined.replace(/ /g, "");
	}

	getExample () {
		return { combined: "() (()) ((() () ())) (() )" };
	}

	static sol (combined) {
		const stripped = combined.replace(/ /g, "");
		const ans = [];
		let cur = "";
		let depth = 0;
		for (const c of stripped) {
			cur += c;
			if (c === "(") {
				depth++;
			} else {
				depth--;
				if (depth === 0) {
					ans.push(cur);
					cur = "";
				}
			}
		}
		return ans;
	}

	genRandom () {
		let depth = 0;
		let combined = "";
		while (depth > 0 || combined === "" || this.random.random() > 0.2) {
			const c = this.random.choice(depth > 0 ? ["(", ")", " "] : ["(", " "]);
			if (c === "(") {
				depth++;
			} else if (c === ")") {
				depth--;
			}
			combined += c;
		}
		this.add({ combined });
	}
}

export class Frac extends PuzzleGenerator {
	static docstring = "* Given a floating point number, find its fractional part.\n\n        Sample Input:\n        4.175\n\n        Sample Output:\n        0.175";

	/** Inspired by HumanEval #2 */
	/**
	 * Given a floating point number, find its fractional part.
	 *
	 * Sample Input:
	 * 4.175
	 *
	 * Sample Output:
	 * 0.175
	 */
	static sat (x, v = 523.12892) {
		return 0 <= x && x < 1 && Number.isInteger(v - x);
	}

	static sol (v) {
		return v % 1.0;
	}

	getExample () {
		return { v: 523.12892 };
	}

	genRandom () {
		const v = this.random.uniform(-100, 100);
		this.add({ v });
	}
}

export class FirstNegCumulative extends PuzzleGenerator {
	static docstring = "* Given a list of numbers which represent bank deposits and withdrawals, find the *first* negative balance.\n\n        Sample Input:\n        [[12, -5, 3, -99, 14, 88, -99], [-1, 2, 5]]\n\n        Sample Output:\n        [-89, -1]";

	/** Inspired by HumanEval #3 */
	/**
	 * Given a list of numbers which represent bank deposits and withdrawals,
	 * find the *first* negative balance.
	 *
	 * Sample Input:
	 * [[12, -5, 3, -99, 14, 88, -99], [-1, 2, 5]]
	 *
	 * Sample Output:
	 * [-89, -1]
	 */
	static sat (
		firsts,
		balances = [
			[2, 7, -2, 4, 3, -15, 10, -45, 3],
			[3, 4, -17, -1],
			[100, -100, -101],
			[-1],
		],
	) {
		for (let i = 0; i < balances.length; i++) {
			let total = 0;
			for (const b of balances[i]) {
				total += b;
				if (total < 0) {
					if (total !== firsts[i]) return false;
					break;
				}
			}
		}
		return true;
	}

	static sol (balances) {
		const firsts = [];
		for (const bals of balances) {
			let total = 0;
			for (const b of bals) {
				total += b;
				if (total < 0) {
					firsts.push(total);
					break;
				}
			}
		}
		return firsts;
	}

	getExample () {
		return {
			balances: [
				[2, 7, -2, 4, 3, -15, 10, -45, 3],
				[3, 4, -17, -1],
				[100, -100, -101],
				[-1],
			],
		};
	}

	genRandom () {
		let balances = Array.from({ length: 10 }, () =>
			Array.from({ length: this.random.randrange(1, 11) }, () =>
				this.random.randrange(-1e10, 1e10),
			),
		);
		balances = balances.filter((bals) => {
			let sum = 0;
			for (let i = 0; i < bals.length; i++) {
				sum += bals[i];
				if (sum < 0) return true;
			}
			return false;
		});
		this.add({ balances });
	}
}

export class MinSquaredDeviation extends PuzzleGenerator {
	static docstring = "* Given a list of numbers, find x that minimizes mean squared deviation.\n\n        Sample Input:\n        [4, -5, 17, -9, 14, 108, -9]\n\n        Sample Output:\n        17.14285";

	/** Inspired by HumanEval #4 */
	static sat (x, nums = [12, -2, 14, 3, -15, 10, -45, 3, 30]) {
		/**
		 * Given a list of numbers, find x that minimizes mean squared deviation.
		 *
		 * Sample Input:
		 * [4, -5, 17, -9, 14, 108, -9]
		 *
		 * Sample Output:
		 * 17.14285
		 */
		const squaredDev = nums.reduce((sum, n) => sum + (n - x) ** 2, 0);
		const pairwiseDev = nums.flatMap((m, i) =>
			nums.slice(i + 1).map((n) => (m - n) ** 2),
		);
		const sumPairwise = pairwiseDev.reduce((a, b) => a + b, 0);
		return squaredDev * nums.length <= sumPairwise + 1e-4;
	}

	static sol (nums) {
		return nums.reduce((a, b) => a + b, 0) / nums.length;
	}

	getExample () {
		return { nums: [12, -2, 14, 3, -15, 10, -45, 3, 30] };
	}

	genRandom () {
		const length = this.random.randrange(1, 11);
		const nums = Array.from({ length }, () => this.random.randrange(-100, 100));
		this.add({ nums });
	}
}

export class Intersperse extends PuzzleGenerator {
	static docstring = "* Given a list of numbers and a number to inject, create a list containing that number in between each pair of\n        adjacent numbers.\n\n        Sample Input:\n        [8, 14, 21, 17, 9, -5], 3\n\n        Sample Output:\n        [8, 3, 14, 3, 21, 3, 17, 3, 9, 3, -5]";

	/** Inspired by HumanEval #5 */
	static sat (li, nums = [12, 23, -2, 5, 0], sep = 4) {
		/**
		 * Given a list of numbers and a number to inject, create a list containing that number
		 * in between each pair of adjacent numbers.
		 *
		 * Sample Input:
		 * [8, 14, 21, 17, 9, -5], 3
		 *
		 * Sample Output:
		 * [8, 3, 14, 3, 21, 3, 17, 3, 9, 3, -5]
		 */
		const evens = li.filter((_, i) => i % 2 === 0);
		const odds = li.filter((_, i) => i % 2 === 1);
		return (
			JSON.stringify(evens) === JSON.stringify(nums) &&
			odds.every((x) => x === sep) &&
			odds.length === nums.length - 1
		);
	}

	static sol (nums, sep) {
		const ans = Array(2 * nums.length - 1).fill(sep);
		nums.forEach((n, i) => (ans[2 * i] = n));
		return ans;
	}

	getExample () {
		return { nums: [12, 23, -2, 5, 0], sep: 4 };
	}

	genRandom () {
		const length = this.random.randrange(10);
		const nums = Array.from({ length }, () => this.random.randrange(100));
		const sep = this.random.randrange(100);
		this.add({ nums, sep });
	}
}

export class DeepestParens extends PuzzleGenerator {
	static docstring = "* Given a string consisting of groups of matched nested parentheses separated by parentheses,\n        compute the depth of each group.\n\n        Sample Input:\n        '(()) ((()()())) (()) ()'\n\n        Sample Output:\n        [2, 3, 2, 1]";

	/** Inspired by HumanEval #6 */
	static sat (depths, parens = "() (()) ((()()())) (((((((())))))))") {
		/**
		 * Given a string consisting of groups of matched nested parentheses separated by spaces,
		 * compute the depth of each group.
		 *
		 * Sample Input:
		 * '(()) ((()()())) (()) ()'
		 *
		 * Sample Output:
		 * [2, 3, 2, 1]
		 */
		const groups = parens.split(" ");
		for (let i = 0; i < groups.length; i++) {
			const depth = depths[i];
			const group = groups[i];
			let budget = depth;
			let success = false;
			for (const c of group) {
				if (c === "(") {
					budget--;
					if (budget === 0) success = true;
					if (budget < 0) return false;
				} else {
					budget++;
				}
			}
			if (!success) return false;
		}
		return groups.length === depths.length;
	}

	static sol (parens) {
		const maxDepth = (s) => {
			let m = 0;
			let depth = 0;
			for (const c of s) {
				if (c === "(") {
					depth++;
					m = Math.max(m, depth);
				} else {
					depth--;
				}
			}
			return m;
		};
		return parens.split(" ").map(maxDepth);
	}

	getExample () {
		return { parens: "() (()) ((()()())) (((((((())))))))" };
	}

	genRandom () {
		const genGroup = () => {
			let ans = "";
			let depth = 0;
			while (depth > 0 || ans === "" || this.random.random() > 0.2) {
				const c = this.random.choice(depth > 0 ? ["(", ")"] : ["("]);
				if (c === "(") depth++;
				else depth--;
				ans += c;
			}
			return ans;
		};
		const parens = Array.from(
			{ length: this.random.randrange(6) },
			genGroup,
		).join(" ");
		this.add({ parens });
	}
}

export class FindContainers extends PuzzleGenerator {
	static docstring = "* Find the strings in a list containing a given substring\n\n        Sample Input:\n        ['cat', 'dog', 'bear'], 'a'\n\n        Sample Output:\n        ['cat', 'bear']";

	/** Inspired by HumanEval #7 */
	static sat (
		containers,
		strings = ["cat", "dog", "shatter", "bear", "at", "ta"],
		substring = "at",
	) {
		/**
		 * Find the strings in a list containing a given substring
		 *
		 * Sample Input:
		 * ['cat', 'dog', 'bear'], 'a'
		 *
		 * Sample Output:
		 * ['cat', 'bear']
		 */
		let i = 0;
		for (const s of strings) {
			if (s.includes(substring)) {
				if (containers[i] !== s) return false;
				i++;
			}
		}
		return i === containers.length;
	}

	static sol (strings, substring) {
		return strings.filter((s) => s.includes(substring));
	}

	getExample () {
		return {
			strings: ["cat", "dog", "shatter", "bear", "at", "ta"],
			substring: "at",
		};
	}

	genRandom () {
		const substring = this.random.pseudoWord(0, 3);
		const gen = () => {
			const n = this.random.choice([1, 2]);
			return Array.from({ length: n }, () => this.random.pseudoWord(0, 5)).join(
				substring,
			);
		};
		const strings = Array.from({ length: this.random.randrange(6) }, gen);
		this.add({ strings, substring });
	}
}

export class SumProduct extends PuzzleGenerator {
	static docstring = "* Find a list of numbers with a given sum and a given product.\n\n        Sample Input:\n        12, 32\n\n        Sample Output:\n        [2, 8, 2]";

	/** Inspired by HumanEval #8 */
	static sat (nums, tot = 14, prod = 99) {
		/**
		 * Find a list of numbers with a given sum and a given product.
		 *
		 * Sample Input:
		 * 12, 32
		 *
		 * Sample Output:
		 * [2, 8, 2]
		 */
		if (nums.reduce((a, b) => a + b, 0) !== tot) return false;
		let p = 1;
		for (const n of nums) p *= n;
		return p === prod;
	}

	static sol (tot, prod) {
		const ans = [prod];
		while (ans.reduce((a, b) => a + b, 0) > tot) {
			ans.push(-1, -1);
		}
		const sum = ans.reduce((a, b) => a + b, 0);
		ans.push(...Array(tot - sum).fill(1));
		return ans;
	}

	getExample () {
		return { tot: 14, prod: 99 };
	}

	genRandom () {
		const tot = this.random.randrange(-100, 100);
		const prod = this.random.randrange(-100, 100);
		this.add({ tot, prod });
	}
}

export class RollingMax extends PuzzleGenerator {
	static docstring = "* Find a list whose ith element is the maximum of the first i elements of the input list.\n\n        Sample Input:\n        [2, 8, 2]\n\n        Sample Output:\n        [2, 8, 8]";

	/** Inspired by HumanEval #9 */
	static sat (maxes, nums = [1, 4, 3, -6, 19]) {
		/**
		 * Find a list whose ith element is the maximum of the first i elements of the input list.
		 *
		 * Sample Input:
		 * [2, 8, 2]
		 *
		 * Sample Output:
		 * [2, 8, 8]
		 */
		if (maxes.length !== nums.length) return false;
		for (let i = 0; i < nums.length; i++) {
			if (i > 0) {
				if (maxes[i] !== Math.max(maxes[i - 1], nums[i])) return false;
			} else {
				if (maxes[0] !== nums[0]) return false;
			}
		}
		return true;
	}

	static sol (nums) {
		const ans = [];
		let m = -Infinity;
		for (const n of nums) {
			m = Math.max(m, n);
			ans.push(m);
		}
		return ans;
	}

	getExample () {
		return { nums: [1, 4, 3, -6, 19] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ nums });
	}
}

export class PalindromeContaining extends PuzzleGenerator {
	static docstring = "* Find a palindrome of a given length containing a given string.\n\n        Sample Input:\n        \"abba\", 6\n\n        Sample Output:\n        \"cabbac\"";

	/** Inspired by HumanEval #10 */
	static sat (ans, s = "so easy", length = 20) {
		/**
		 * Find a palindrome of a given length containing a given string.
		 *
		 * Sample Input:
		 * "abba", 6
		 *
		 * Sample Output:
		 * "cabbac"
		 */
		return (
			ans === ans.split("").reverse().join("") &&
			ans.length === length &&
			ans.includes(s)
		);
	}

	static sol (s, length) {
		const ls = s.split("");
		for (let i = 0; i <= length - s.length; i++) {
			const arr = Array(length).fill("x");
			for (let j = 0; j < ls.length; j++) {
				arr[i + j] = ls[j];
			}
			const a = length - i - 1;
			const b = length - (i + s.length);
			for (let k = a; k >= b; k--) {
				arr[k] = ls[a - k];
			}
			if (arr.join("") === arr.slice().reverse().join("")) {
				const candidate = arr.join("");
				if (candidate.includes(s)) return candidate;
			}
		}
		throw new Error("shouldn't reach here");
	}

	getExample () {
		return { s: "so easy", length: 20 };
	}

	genRandom () {
		const part = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.choice(["a", "b"]),
		).join("");
		const pal =
			part +
			this.random
				.choice([part, part.slice(0, -1)])
				.split("")
				.reverse()
				.join("");
		const n = this.random.randrange(pal.length + 1);
		const m = this.random.randrange(n + 1);
		const s = pal.slice(m, n);
		this.add({ s, length: pal.length });
	}
}

export class BinaryStrXOR extends PuzzleGenerator {
	static docstring = "* Find the XOR of two given strings interpreted as binary numbers.\n\nSample Input:\n\"0001\", \"1011\"\n\nSample Output:\n\"1010\"";
	/** Inspired by HumanEval #11 */
	static sat (strNum, nums = ["100011101100001", "100101100101110"]) {
		/**
		 * Find the XOR of two given strings interpreted as binary numbers.
		 *
		 * Sample Input:
		 * "0001", "1011"
		 *
		 * Sample Output:
		 * "1010"
		 */
		const [a, b] = nums;
		return (
			Number.parseInt(strNum, 2) ===
			(Number.parseInt(a, 2) ^ Number.parseInt(b, 2))
		);
	}

	static sol (nums) {
		const [a, b] = nums;
		const ans = Number.parseInt(a, 2) ^ Number.parseInt(b, 2);
		return ans.toString(2);
	}

	getExample () {
		return { nums: ["100011101100001", "100101100101110"] };
	}

	genRandom () {
		const nums = Array.from({ length: 2 }, () =>
			this.random.randrange(1024).toString(2),
		);
		this.add({ nums });
	}
}

export class LongestStr extends PuzzleGenerator {
	static docstring = "* Find the longest of a list of strings\n\nSample Input:\n[\"cat\", \"dog\", \"sheep\", \"chimp\"]\n\nSample Output:\n\"sheep\"";
	/** Inspired by HumanEval #12 */
	static sat (ans, words = ["these", "are", "some", "pretty", "long", "words"]) {
		/**
		 * Find the longest of a list of strings
		 *
		 * Sample Input:
		 * ["cat", "dog", "sheep", "chimp"]
		 *
		 * Sample Output:
		 * "sheep"
		 */
		return words.includes(ans) && words.every((w) => ans.length >= w.length);
	}

	static sol (words) {
		return words.reduce((a, b) => (a.length >= b.length ? a : b));
	}

	getExample () {
		return { words: ["these", "are", "some", "pretty", "long", "words"] };
	}

	genRandom () {
		const words = Array.from({ length: this.random.randrange(1, 10) }, () =>
			this.random.pseudoWord(),
		);
		this.add({ words });
	}
}

export class CertifiedGCD extends PuzzleGenerator {
	static docstring = "* Find the greatest common divisor of two integers m, n and a certificate a, b such that m*a + n*b = gcd\n\nSample Input:\n20, 30\n\nSample Output:\n10, -1, 1";
	/** Inspired by HumanEval #13 */
	static sat (ans, m = 200004931, n = 66679984) {
		/**
		 * Find the greatest common divisor of two integers m, n and a certificate a, b
		 * such that m*a + n*b = gcd
		 *
		 * Sample Input:
		 * 20, 30
		 *
		 * Sample Output:
		 * [10, -1, 1]
		 */
		const [gcd, a, b] = ans;
		return m % gcd === 0 && n % gcd === 0 && a * m + b * n === gcd && gcd > 0;
	}

	static sol (m, n) {
		const gcdCert = (small, big) => {
			if (big % small === 0) return [small, 1, 0];
			const [gcd, a, b] = gcdCert(big % small, small);
			return [gcd, b - a * Math.floor(big / small), a];
		};

		if (m < n) {
			return gcdCert(m, n);
		}
		const [gcd, a, b] = gcdCert(n, m);
		return [gcd, b, a];
	}

	getExample () {
		return { m: 200004931, n: 66679984 };
	}

	genRandom () {
		const factor = 1 + this.random.randrange(10 ** this.random.randrange(10));
		const r1 = 1 + this.random.randrange(10 ** this.random.randrange(10));
		const r2 = 1 + this.random.randrange(10 ** this.random.randrange(10));
		const m = r1 * factor;
		const n = r2 * factor;
		this.add({ m, n });
	}
}

export class AllPrefixes extends PuzzleGenerator {
	static docstring = "* Find all prefixes of a given string\n\nSample Input:\n\"aabcd\"\n\nSample Output:\n[\"\", \"a\", \"aa\", \"aab\", \"aabc\", \"aabcd\"]";
	/** Inspired by HumanEval #14 */
	static sat (prefixes, s = "donesezichethofalij") {
		/**
		 * Find all prefixes of a given string
		 *
		 * Sample Input:
		 * "aabcd"
		 *
		 * Sample Output:
		 * ["", "a", "aa", "aab", "aabc", "aabcd"]
		 */
		return (
			prefixes.every((p) => s.startsWith(p)) &&
			new Set(prefixes).size > s.length
		);
	}

	static sol (s) {
		return Array.from({ length: s.length + 1 }, (_, i) => s.slice(0, i));
	}

	getExample () {
		return { s: "donesezichethofalij" };
	}

	genRandom () {
		const s = this.random.pseudoWord(0, 30);
		this.add({ s });
	}
}

export class SpaceyRange extends PuzzleGenerator {
	static docstring = "* Find a string consisting of the non-negative integers up to n inclusive\n\nSample Input:\n4\n\nSample Output:\n'0 1 2 3 4'";
	/** Inspired by HumanEval #15 */
	static sat (ans, n = 15) {
		/**
		 * Find a string consisting of the non-negative integers up to n inclusive
		 *
		 * Sample Input:
		 * 4
		 *
		 * Sample Output:
		 * '0 1 2 3 4'
		 */
		const nums = ans.split(" ").map(Number);
		return nums.length === n + 1 && nums.every((num, i) => num === i);
	}

	static sol (n) {
		return Array.from({ length: n + 1 }, (_, i) => i).join(" ");
	}

	getExample () {
		return { n: 15 };
	}

	genRandom () {
		const n = this.random.randrange(1e5);
		this.add({ n });
	}
}

export class DistinctChars extends PuzzleGenerator {
	static docstring = "* Find the set of distinct characters in a string, ignoring case\n\nSample Input:\n'HELlo', 4\n\nSample Output:\n['h', 'e', 'l', 'o']";
	/** Inspired by HumanEval #16 */
	static sat (ans, s = "The quick brown fox jumps over the lazy dog!", n = 28) {
		/**
		 * Find the set of distinct characters in a string, ignoring case
		 *
		 * Sample Input:
		 * 'HELlo', 4
		 *
		 * Sample Output:
		 * ['h', 'e', 'l', 'o']
		 */
		const lower = s.toLowerCase();
		const distinct = new Set(lower);
		return (
			ans.length === distinct.size &&
			ans.every((c) => distinct.has(c)) &&
			ans.every((c) => c === c.toLowerCase())
		);
	}

	static sol (s, n) {
		return [...new Set(s.toLowerCase())];
	}

	getExample () {
		return { s: "The quick brown fox jumps over the lazy dog!", n: 28 };
	}

	genRandom () {
		let s = this.random.string();
		s = s[0].toUpperCase() + s.slice(1);
		const n = new Set(s.toLowerCase()).size;
		this.add({ s, n });
	}
}

export class ParseMusic extends PuzzleGenerator {
	static docstring = "* Parse a string of notes to beats, 'o'=4, 'o|'=2, '.|'=1\n\nExample input:\n'o o .| o|'\n\nExample output:\n[4, 4, 1, 2]";
	/** Inspired by HumanEval #17 */
	static sat (beats, score = "o o o| o| .| .| .| o| o| o o o| .|") {
		/**
		 * Parse a string of notes to beats, 'o'=4, 'o|'=2, '.|'=1
		 *
		 * Example input:
		 * 'o o .| o|'
		 *
		 * Example output:
		 * [4, 4, 1, 2]
		 */
		const mapping = { 1: ".|", 2: "o|", 4: "o" };
		return beats.map((b) => mapping[b]).join(" ") === score;
	}

	static sol (score) {
		const mapping = { ".|": 1, "o|": 2, o: 4 };
		return score.split(" ").map((note) => mapping[note]);
	}
	getExample () {
		return { score: "o o o| o| .| .| .| o| o| o o o| .|" };
	}
	genRandom () {
		const n = this.random.randrange(12);
		const score = Array.from({ length: n }, () =>
			this.random.choice([".|", "o|", "o"]),
		).join(" ");
		this.add({ score });
	}
}

export class OverlappingCount extends PuzzleGenerator {
	static docstring = "* Find occurrences of a substring in a parent string *including overlaps*\n\nSample Input:\n'helllo', 'll'\n\nSample Output:\n[2, 3]";
	/** Inspired by HumanEval #18 */
	static sat (
		ans,
		s = "Bananannanaannanaanananananana",
		sub = "anan",
		count = 7,
	) {
		/**
		 * Find occurrences of a substring in a parent string *including overlaps*
		 *
		 * Sample Input:
		 * 'helllo', 'll'
		 *
		 * Sample Output:
		 * [2, 3]
		 */
		return (
			ans.every((i) => s.slice(i, i + sub.length) === sub && i >= 0) &&
			new Set(ans).size >= count
		);
	}

	static sol (s, sub, count) {
		const ans = [];
		for (let i = 0; i <= s.length; i++) {
			if (s.slice(i, i + sub.length) === sub) {
				ans.push(i);
			}
		}
		return ans;
	}

	getExample () {
		return { s: "Bananannanaannanaanananananana", sub: "anan", count: 7 };
	}

	genRandom () {
		const s = this.random.pseudoWord(1, 100);
		const j = this.random.randrange(1, s.length + 1);
		const i = this.random.randrange(j);
		const sub = s.slice(i, j);
		const count = this.sol(s, sub, null).length;
		this.add({ s, sub, count });
	}
}

export class SortNumbers extends PuzzleGenerator {
	static docstring = "* Sort numbers based on strings\n\nSample input\n---\n\"six one four\"\n\nSample output\n---\n\"one four six\"";
	/** Inspired by HumanEval #19 */
	static sat (ans, s = "six one four three two nine eight") {
		/**
		 * Sort numbers based on strings
		 *
		 * Sample input:
		 * "six one four"
		 *
		 * Sample output:
		 * "one four six"
		 */
		const nums = "zero one two three four five six seven eight nine".split(" ");
		const ansIndices = ans.split(" ").map((x) => nums.indexOf(x));
		const sIndices = s.split(" ").map((x) => nums.indexOf(x));
		return (
			JSON.stringify(ansIndices) ===
			JSON.stringify([...sIndices].sort((a, b) => a - b))
		);
	}

	static sol (s) {
		const nums = "zero one two three four five six seven eight nine".split(" ");
		const arr = s.split(" ").map((x) => nums.indexOf(x));
		arr.sort((a, b) => a - b);
		return arr.map((i) => nums[i]).join(" ");
	}

	getExample () {
		return { s: "six one four three two nine eight" };
	}

	genRandom () {
		const nums = "zero one two three four five six seven eight nine".split(" ");
		const n = this.random.randrange(3, 9);
		const words = Array.from({ length: n }, () => this.random.choice(nums));
		this.add({ s: words.join(" ") });
	}
}

export class FindClosePair extends PuzzleGenerator {
	static docstring = "* Given a list of numbers, find the indices of the closest pair.\n\nSample Input:\n[1.2, 5.25, 0.89, 21.0, 5.23]\n\nSample Output:\n[4, 1]";
	/** Inspired by HumanEval #20 */
	static sat (inds, nums = [0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2]) {
		/**
		 * Given a list of numbers, find the indices of the closest pair.
		 *
		 * Sample Input:
		 * [1.2, 5.25, 0.89, 21.0, 5.23]
		 *
		 * Sample Output:
		 * [4, 1]
		 */
		const [a, b] = inds;
		if (a === b || a < 0 || b < 0) return false;
		const minDist = Math.abs(nums[b] - nums[a]);
		for (let i = 0; i < nums.length; i++) {
			for (let j = 0; j < i; j++) {
				if (Math.abs(nums[i] - nums[j]) < minDist) return false;
			}
		}
		return true;
	}

	static sol (nums) {
		let best = [0, 1];
		let bestScore = Math.abs(nums[1] - nums[0]);
		for (let i = 0; i < nums.length; i++) {
			for (let j = 0; j < i; j++) {
				const score = Math.abs(nums[i] - nums[j]);
				if (score < bestScore) {
					bestScore = score;
					best = [i, j];
				}
			}
		}
		return best;
	}

	getExample () {
		return { nums: [0.31, 21.3, 5.0, 9.0, 11.0, 5.01, 17.2] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(2, 10) }, () =>
			this.random.uniform(-10, 10),
		);
		if (this.random.random() < 0.2) {
			nums.push(nums[0]);
		}
		this.random.shuffle(nums);
		this.add({ nums });
	}
}

export class Rescale extends PuzzleGenerator {
	static docstring = "* Rescale and shift numbers so that they cover the range [0, 1]\n\n        Sample input\n        ---\n        [18.5, 17.0, 18.0, 19.0, 18.0]\n\n        Sample output\n        ---\n        [0.75, 0.0, 0.5, 1.0, 0.5]";
	/** Inspired by HumanEval #21 */
	static sat (ans, nums = [13.0, 17.0, 17.0, 15.5, 2.94]) {
		/**
		 * Rescale and shift numbers so that they cover the range [0, 1]
		 *
		 * Sample input
		 * ---
		 * [18.5, 17.0, 18.0, 19.0, 18.0]
		 *
		 * Sample output
		 * ---
		 * [0.75, 0.0, 0.5, 1.0, 0.5]
		 */
		if (Math.min(...ans) !== 0.0 || Math.max(...ans) !== 1.0) return false;
		const a = Math.min(...nums);
		const b = Math.max(...nums);
		for (let i = 0; i < nums.length; i++) {
			const x = a + (b - a) * ans[i];
			if (Math.abs(nums[i] - x) >= 1e-6) return false;
		}
		return true;
	}

	static sol (nums) {
		nums = [...nums];
		const a = Math.min(...nums);
		const b = Math.max(...nums);
		if (b - a === 0) {
			return [0.0, ...Array(nums.length - 1).fill(1.0)];
		}
		for (let i = 0; i < nums.length; i++) {
			nums[i] = (nums[i] - a) / (b - a);
		}
		return nums;
	}

	getExample () {
		return { nums: [13.0, 17.0, 17.0, 15.5, 2.94] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(2, 10) }, () =>
			this.random.heavy_tail_float(),
		);
		if (this.random.random() < 0.2) {
			nums.fill(nums[0]);
		}
		this.add({ nums });
	}
}

export class FilterInts extends PuzzleGenerator {
	static docstring = "* Find a list of strings where the only valid integers are at the given indices\n\n        Sample input\n        ---\n        [2, 4, 5]\n\n        Sample output\n        ---\n        [\"cat\", \"2.7\", \"2\", \"\", \"3\", \"-17\", \"free\"]";
	/** Inspired by HumanEval #22 */
	static sat (candidates, int_indices = [2, 4, 7, 9, 101]) {
		/**
		 * Find a list of strings where the only valid integers are at the given indices
		 *
		 * Sample input
		 * ---
		 * [2, 4, 5]
		 *
		 * Sample output
		 * ---
		 * ["cat", "2.7", "2", "", "3", "-17", "free"]
		 */
		for (const i of int_indices) {
			if (
				isNaN(parseInt(candidates[i])) ||
				parseInt(candidates[i]).toString() !== candidates[i]
			) {
				return false;
			}
		}
		for (let i = 0; i < candidates.length; i++) {
			if (!int_indices.includes(i)) {
				try {
					const parsed = parseInt(candidates[i]);
					if (!isNaN(parsed) && parsed.toString() === candidates[i]) {
						return false;
					}
				} catch (e) {
					// ok
				}
			}
		}
		return true;
	}

	static sol (int_indices) {
		if (int_indices.length === 0) return [];
		const ans = Array(1 + Math.max(...int_indices.map(Math.abs))).fill("");
		for (const i of int_indices) {
			ans[i] = "17";
		}
		return ans;
	}

	getExample () {
		return { int_indices: [2, 4, 7, 9, 101] };
	}

	genRandom () {
		const int_indices = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(100),
		);
		this.add({ int_indices });
	}
}

export class StrLength extends PuzzleGenerator {
	static docstring = "* Find the lengths of a list of non-empty strings\n\n        Sample input\n        ---\n        [\"foo\", \"bars\"]\n\n        Sample output\n        ---\n        [3, 4]";
	/** Inspired by HumanEval #23 */
	static sat (
		lengths,
		strs = ["pneumonoultramicroscopicsilicovolcanoconiosis", " ", "foo", "2.5"],
	) {
		/**
		 * Find the lengths of a list of non-empty strings
		 *
		 * Sample input
		 * ---
		 * ["foo", "bars"]
		 *
		 * Sample output
		 * ---
		 * [3, 4]
		 */
		for (let i = 0; i < lengths.length; i++) {
			const length = lengths[i];
			const s = strs[i];
			if (length < s.length && s[length] !== undefined) return false;
			if (length === 0 || s[length - 1] === undefined) return false;
		}
		return lengths.length === strs.length;
	}

	static sol (strs) {
		return strs.map((s) => s.length);
	}

	getExample () {
		return {
			strs: [
				"pneumonoultramicroscopicsilicovolcanoconiosis",
				" ",
				"foo",
				"2.5",
			],
		};
	}

	genRandom () {
		const strs = Array.from({ length: 10 }, () => this.random.string(1, 50));
		this.add({ strs });
	}
}

export class LargestDivisor extends PuzzleGenerator {
	static docstring = "* Find the largest integer divisor of a number n that is less than n\n\n        Sample input\n        ---\n        1000\n\n        Sample output\n        ---\n        500";
	/** Inspired by HumanEval #24 */
	static sat (d, n = 123456) {
		/**
		 * Find the largest integer divisor of a number n that is less than n
		 *
		 * Sample input
		 * ---
		 * 1000
		 *
		 * Sample output
		 * ---
		 * 500
		 */
		if (n % d !== 0 || d >= n) return false;
		for (let e = d + 1; e < n; e++) {
			if (n % e === 0) return false;
		}
		return true;
	}

	static sol (n) {
		for (let d = n - 1; d > 0; d--) {
			if (n % d === 0) return d;
		}
	}

	getExample () {
		return { n: 123456 };
	}

	genRandom () {
		const n = this.random.randrange(1, 10 ** 5);
		this.add({ n });
	}
}

export class PrimeFactorization extends PuzzleGenerator {
	static docstring = "* Factor number n into a given number of non-trivial factors\n\n        Sample input\n        ---\n        1000, 6\n\n        Sample output\n        ---\n        [2, 2, 2, 5, 5, 5]";
	/** Inspired by HumanEval #25 */
	static sat (factors, n = 123456, num_factors = 8) {
		/**
		 * Factor number n into a given number of non-trivial factors
		 *
		 * Sample input
		 * ---
		 * 1000, 6
		 *
		 * Sample output
		 * ---
		 * [2, 2, 2, 5, 5, 5]
		 */
		if (factors.length !== num_factors) return false;
		let prod = 1;
		for (const d of factors) {
			prod *= d;
			if (d <= 1) return false;
		}
		return prod === n;
	}

	static sol (n, num_factors) {
		if (num_factors === 0) return [];
		if (num_factors === 1) return [n];
		const ans = [];
		for (let d = 2; d < n; d++) {
			while (n % d === 0) {
				n = Math.floor(n / d);
				ans.push(d);
				if (ans.length === num_factors - 1) {
					ans.push(n);
					return ans;
				}
			}
		}
		throw new Error("Should not reach here");
	}

	getExample () {
		return { n: 123456, num_factors: 8 };
	}

	genRandom () {
		let num_factors = this.random.randrange(10);
		let n = 2 ** num_factors;
		for (let i = 0; i < this.random.randrange(10); i++) {
			n *= this.random.choice([
				3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
			]);
			num_factors += 1;
		}
		this.add({ n, num_factors });
	}
}

export class Dedup extends PuzzleGenerator {
	static docstring = "* Remove duplicates from a list of integers, preserving order\n\n        Sample input\n        ---\n        [1, 3, 2, 9, 2, 1, 55]\n\n        Sample output\n        ---\n        [1, 3, 2, 9, 55]";
	/** Inspired by HumanEval #26 */
	static sat (ans, li = [2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31]) {
		/**
		 * Remove duplicates from a list of integers, preserving order
		 *
		 * Sample input
		 * ---
		 * [1, 3, 2, 9, 2, 1, 55]
		 *
		 * Sample output
		 * ---
		 * [1, 3, 2, 9, 55]
		 */
		const setAns = new Set(ans);
		const setLi = new Set(li);
		if (setAns.size !== setLi.size) return false;
		for (const item of setAns) {
			if (!setLi.has(item)) return false;
		}
		for (let i = 0; i < ans.length - 1; i++) {
			if (li.indexOf(ans[i]) >= li.indexOf(ans[i + 1])) return false;
		}
		return true;
	}

	static sol (li) {
		const seen = new Set();
		const ans = [];
		for (const n of li) {
			if (!seen.has(n)) {
				ans.push(n);
				seen.add(n);
			}
		}
		return ans;
	}

	getExample () {
		return { nums: [2, 19, 2, 53, 1, 1, 2, 44, 17, 0, 19, 31] };
	}

	genRandom () {
		const n = this.random.randrange(20);
		const nums = Array.from({ length: n }, () => this.random.randrange(10));
		this.add({ nums });
	}
}

export class FlipCase extends PuzzleGenerator {
	static docstring = "* Flip case\n\n        Sample input\n        ---\n        'cAt'\n\n        Sample output\n        ---\n        'CaT'";
	/** Inspired by HumanEval #27 */
	static sat (ans, s = "FlIp ME!") {
		/**
		 * Flip case
		 *
		 * Sample input
		 * ---
		 * 'cAt'
		 *
		 * Sample output
		 * ---
		 * 'CaT'
		 */
		if (ans.length !== s.length) return false;
		for (let i = 0; i < ans.length; i++) {
			const c = ans[i];
			const d = s[i];
			// Check that {c, d} == {d.upper(), d.lower()}
			const set1 = new Set([c, d]);
			const set2 = new Set([d.toUpperCase(), d.toLowerCase()]);
			if (set1.size !== set2.size || ![...set1].every((x) => set2.has(x))) {
				return false;
			}
		}
		return true;
	}

	static sol (s) {
		return s
			.split("")
			.map((c) => (c.toUpperCase() === c ? c.toLowerCase() : c.toUpperCase()))
			.join("");
	}

	getExample () {
		return { s: "FlIp ME!" };
	}

	genRandom () {
		const w = this.random.string();
		const s = w
			.split("")
			.map((c) =>
				this.random.choice([
					c.toUpperCase(),
					c.toLowerCase(),
					c.toUpperCase(),
					c.toLowerCase(),
					c.toUpperCase(),
					c.toLowerCase(),
					c.toUpperCase(),
					c.toLowerCase(),
					c.toUpperCase(),
					c.toLowerCase(),
					" ",
					"!",
					"3",
				]),
			)
			.join("");
		this.add({ s });
	}
}

export class CatStrings extends PuzzleGenerator {
	static docstring = "* Concatenate a list of strings\n\n        Sample input\n        ---\n        ['cat', 'dog', 'bird']\n\n        Sample output\n        ---\n        'catdogbird'";
	/** Inspired by HumanEval #28 */
	static sat (cat, strings = ["Will", "i", "am", "Now", "here"]) {
		/**
		 * Concatenate a list of strings
		 *
		 * Sample input
		 * ---
		 * ['cat', 'dog', 'bird']
		 *
		 * Sample output
		 * ---
		 * 'catdogbird'
		 */
		let i = 0;
		for (const s of strings) {
			for (const c of s) {
				if (cat[i] !== c) return false;
				i++;
			}
		}
		return i === cat.length;
	}

	static sol (strings) {
		return strings.join("");
	}

	getExample () {
		return { strings: ["Will", "i", "am", "Now", "here"] };
	}

	genRandom () {
		const strings = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.pseudo_word(),
		);
		this.add({ strings });
	}
}

export class FindExtensions extends PuzzleGenerator {
	static docstring = "* Find the strings in a list starting with a given prefix\n\n        Sample Input:\n        ['cat', 'car', 'fear', 'center'], 'ca'\n\n        Sample Output:\n        ['cat', 'car']";
	/** Inspired by HumanEval #29 */
	static sat (
		extensions,
		strings = ["cat", "dog", "shatter", "donut", "at", "todo"],
		prefix = "do",
	) {
		/**
		 * Find the strings in a list starting with a given prefix
		 *
		 * Sample Input:
		 * ['cat', 'car', 'fear', 'center'], 'ca'
		 *
		 * Sample Output:
		 * ['cat', 'car']
		 */
		let i = 0;
		for (const s of strings) {
			if (s.startsWith(prefix)) {
				if (extensions[i] !== s) return false;
				i++;
			}
		}
		return i === extensions.length;
	}

	static sol (strings, prefix) {
		return strings.filter((s) => s.startsWith(prefix));
	}

	getExample () {
		return {
			strings: ["cat", "dog", "shatter", "donut", "at", "todo"],
			prefix: "do",
		};
	}

	genRandom () {
		const prefix = this.random.pseudo_word(0, 3);
		const gen = () =>
			this.random.choice(["", prefix]) + this.random.pseudo_word(0, 5);
		const strings = Array.from({ length: this.random.randrange(6) }, gen);
		this.add({ strings, prefix });
	}
}

export class FindPositives extends PuzzleGenerator {
	static docstring = "* Find the positive integers in a list\n\n        Sample Input:\n        [-1, 3, 19, -2, 0, 44, 0, 44, 11]\n\n        Sample Output:\n        [3, 19, 44, 44, 11]";
	/** Inspired by HumanEval #30 */
	static sat (positives, nums = [2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11]) {
		/**
		 * Find the positive integers in a list
		 *
		 * Sample Input:
		 * [-1, 3, 19, -2, 0, 44, 0, 44, 11]
		 *
		 * Sample Output:
		 * [3, 19, 44, 44, 11]
		 */
		const stack = [...positives].reverse();
		for (const n of nums) {
			if (n > 0) {
				if (stack.pop() !== n) return false;
			}
		}
		return stack.length === 0;
	}

	static sol (nums) {
		return nums.filter((i) => i > 0);
	}

	getExample () {
		return { nums: [2, 2342, -2, 32, -8, -5, 2342, 0, -9, 44, 11] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ nums });
	}
}

export class FermatComposites extends PuzzleGenerator {
	static docstring = "* Find Fermat composite certificates for a list of numbers > 1\n\n        Sample Input:\n        [1469]\n\n        Sample Output:\n        [3]  # because (3 ** 1468) % 1469 != 1";
	/** Inspired by HumanEval #31 */
	static sat (certificates, nums = [1449, 14, 21, 105, 217]) {
		/**
		 * Find Fermat composite certificates for a list of numbers > 1
		 *
		 * Sample Input:
		 * [1469]
		 *
		 * Sample Output:
		 * [3]  // because (3 ** 1468) % 1469 != 1
		 */
		if (certificates.length !== nums.length) return false;
		for (let i = 0; i < certificates.length; i++) {
			const cert = certificates[i];
			const n = nums[i];
			// Compute pow(cert, n - 1, n)
			let result = 1;
			let base = cert % n;
			let exp = n - 1;
			while (exp > 0) {
				if (exp % 2 === 1) {
					result = (result * base) % n;
				}
				base = (base * base) % n;
				exp = Math.floor(exp / 2);
			}
			if (result <= 1) return false;
		}
		return true;
	}

	static sol (nums) {
		const result = [];
		for (const n of nums) {
			for (let i = 2; i < n; i++) {
				// Compute pow(i, n - 1, n)
				let res = 1;
				let base = i % n;
				let exp = n - 1;
				while (exp > 0) {
					if (exp % 2 === 1) {
						res = (res * base) % n;
					}
					base = (base * base) % n;
					exp = Math.floor(exp / 2);
				}
				if (res > 1) {
					result.push(i);
					break;
				}
			}
		}
		return result;
	}

	getExample () {
		return { nums: [1449, 14, 21, 105, 217] };
	}

	genRandom () {
		const nums = [];
		for (let i = 0; i < this.random.randrange(10); i++) {
			let a = this.random.randrange(3, 10 ** 5, 2);
			const b = this.random.randrange(3, 10 ** 5, 2);
			if (this.random.randrange(10) === 0) {
				a += 1;
			}
			nums.push(a * b);
		}
		this.add({ nums });
	}
}

export class OddDegreePolynomialRoot extends PuzzleGenerator {
	static docstring = "* Find a real root of an odd degree polynomial from its coefficients\n\n        Sample Input:\n        [1, 0, 8]\n\n        Sample Output:\n        -2.0  # 1*(-2.0)^3 + 8 == 0";
	/**
	 * Polynomials of odd degree always have a real solution.
	 *
	 * Inspired by HumanEval #32
	 */
	static sat (root, coeffs = [1, 2, 3, 17]) {
		/**
		 * Find a real root of an odd degree polynomial from its coefficients
		 *
		 * Sample Input:
		 * [1, 0, 8]
		 *
		 * Sample Output:
		 * -2.0  // 1*(-2.0)^3 + 8 == 0
		 */
		const result = coeffs.reduce((sum, coeff, i) => sum + coeff * root ** i, 0);
		return Math.abs(result) < 1e-4;
	}

	static sol (coeffs) {
		const p = (x) => coeffs.reduce((sum, coeff, i) => sum + coeff * x ** i, 0);

		for (let attempt = 0; attempt < 100; attempt++) {
			let a = -(10 ** attempt);
			let b = 10 ** attempt;
			let p_a = p(a);
			let p_b = p(b);
			while (p_a * p_b <= 0) {
				const mid = (a + b) / 2;
				const p_mid = p(mid);
				if (Math.abs(p_mid) < 1e-4) {
					return mid;
				}
				if (mid === a || mid === b) {
					throw new Error("Precision error");
				}
				if (p_mid * p_a > 0) {
					a = mid;
					p_a = p_mid;
				} else {
					b = mid;
					p_b = p_mid;
				}
			}
		}
		throw new Error("Root finder failed on 100 attempts");
	}

	getExample () {
		return { coeffs: [1, 2, 3, 17] };
	}

	genRandom () {
		const degree = this.random.randrange(1, 10, 2);
		const coeffs = Array.from({ length: degree }, () =>
			this.random.randrange(-10, 10),
		);
		coeffs.push(this.random.randrange(1, 10));
		this.add({ coeffs });
	}
}

export class TwoThirdsSorted extends PuzzleGenerator {
	static docstring = "* Start with a list of integers, keep every third element in place and otherwise sort the list\n\n        Sample Input:\n        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]\n\n        Sample Output:\n        [8, 0, 2, 2, 4, 8, 1, 8, 9, 3]";
	/** Inspired by HumanEval #33 */
	static sat (li, orig = [1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0]) {
		/**
		 * Start with a list of integers, keep every third element in place and otherwise sort the list
		 *
		 * Sample Input:
		 * [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]
		 *
		 * Sample Output:
		 * [8, 0, 2, 2, 4, 8, 1, 8, 9, 3]
		 */
		// Check that every third entry is fixed
		for (let i = 0; i < orig.length; i += 3) {
			if (orig[i] !== li[i]) return false;
		}
		// Check that it's a permutation
		const sortedLi = [...li].sort((a, b) => a - b);
		const sortedOrig = [...orig].sort((a, b) => a - b);
		if (JSON.stringify(sortedLi) !== JSON.stringify(sortedOrig)) return false;
		// Check sorting constraints
		for (let i = 1; i < li.length - 1; i += 3) {
			if (li[i] > li[i + 1]) return false;
		}
		for (let i = 2; i < li.length - 2; i += 3) {
			if (li[i] > li[i + 2]) return false;
		}
		return true;
	}

	static sol (orig) {
		const n = orig.length;
		const your_list = [];
		for (let i = 0; i < n; i += 3) {
			your_list.push(orig[i]);
		}
		const sub = [];
		for (let i = 0; i < orig.length; i++) {
			if (i % 3 !== 0) {
				sub.push(orig[i]);
			}
		}
		sub.sort((a, b) => a - b);
		const answ = [];
		for (let i = 0; i < Math.floor(n / 3); i++) {
			answ.push(your_list[i]);
			answ.push(sub[i * 2]);
			answ.push(sub[i * 2 + 1]);
		}
		if (n % 3 === 1) {
			answ.push(your_list[your_list.length - 1]);
		}
		if (n % 3 === 2) {
			answ.push(your_list[your_list.length - 1]);
			answ.push(sub[sub.length - 1]);
		}
		return answ;
	}

	getExample () {
		return { orig: [1, -2, 3, 17, 8, 4, 12, 3, 18, 5, -29, 0, 0] };
	}

	genRandom () {
		const list_length = this.random.randrange(20);
		const orig = Array.from({ length: list_length }, () =>
			this.random.randrange(-10, 10),
		);
		this.add({ orig });
	}
}

export class UniqueSorted extends PuzzleGenerator {
	/** Inspired by HumanEval #34 */
	static docstring = "* Find an increasing sequence consisting of the elements of the original list.\n\n        Sample Input:\n        [8, 0, 7, 2, 9, 4, 4, -2, 8, 3]\n\n        Sample Output:\n        [-2, 0, 2, 3, 4, 7, 8, 9]";
	static sat (li, orig = [1, 1, 3, 2, 0, 8, 32, -4, 0]) {
		/**
		 * Find an increasing sequence consisting of the elements of the original list.
		 *
		 * Sample Input:
		 * [8, 0, 7, 2, 9, 4, 4, -2, 8, 3]
		 *
		 * Sample Output:
		 * [-2, 0, 2, 3, 4, 7, 8, 9]
		 */
		for (let i = 0; i < li.length - 1; i++) {
			if (li[i] >= li[i + 1]) return false;
			if (!orig.includes(li[i])) return false;
		}
		if (li.length > 0 && !orig.includes(li[li.length - 1])) return false;
		for (const n of orig) {
			if (!li.includes(n)) return false;
		}
		return true;
	}

	static sol (orig) {
		return [...new Set(orig)].sort((a, b) => a - b);
	}

	getExample () {
		return { orig: [1, 1, 3, 2, 0, 8, 32, -4, 0] };
	}

	genRandom () {
		const list_length = this.random.randrange(20);
		const orig = Array.from({ length: list_length }, () =>
			this.random.randrange(-10, 10),
		);
		this.add({ orig });
	}
}

export class MaxInt extends PuzzleGenerator {
	/** Inspired by HumanEval #35 */
	static docstring = "* Find the largest integer in a sequence\n\n        Sample Input:\n        [8, 0, 1, 4, 9, 3, 4, -2, 8, 3]\n\n        Sample Output:\n        9";
	static sat (
		m,
		hello = [
			1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 3502145, 3502145, 21, 18, 2, 60,
		],
	) {
		/**
		 * Find the largest integer in a sequence
		 *
		 * Sample Input:
		 * [8, 0, 1, 4, 9, 3, 4, -2, 8, 3]
		 *
		 * Sample Output:
		 * 9
		 */
		if (!hello.includes(m)) return false;
		for (const i of hello) {
			if (m < i) return false;
		}
		return true;
	}

	static sol (hello) {
		return Math.max(...hello);
	}

	getExample () {
		return {
			hello: [
				1, 31, 3, 2, 0, 18, 32, -4, 2, -1000, 3502145, 3502145, 21, 18, 2, 60,
			],
		};
	}

	genRandom () {
		const list_length = this.random.randrange(1, 20);
		const hello = Array.from({ length: list_length }, () =>
			this.random.randrange(-10, 10),
		);
		this.add({ hello });
	}
}

export class SevenElevenThirteen extends PuzzleGenerator {
	/** Inspired by HumanEval #36 */
	static docstring = "* Find all 7's in integers less than n that are divisible by 11 or 13\n\n        Sample Input:\n        79, 3\n\n        Sample Output:\n        [[77, 0], [77, 1], [78, 0]]";
	static sat (li, n = 19723, lower = 1000) {
		/**
		 * Find all 7's in integers less than n that are divisible by 11 or 13
		 *
		 * Sample Input:
		 * 79, 3
		 *
		 * Sample Output:
		 * [[77, 0], [77, 1], [78, 0]]
		 */
		const unique = new Set(li.map((pair) => JSON.stringify(pair)));
		if (unique.size < lower) return false;
		for (const [i, j] of li) {
			if (
				i.toString()[j] !== "7" ||
				(i % 11 !== 0 && i % 13 !== 0) ||
				i < 0 ||
				i >= n ||
				j < 0
			) {
				return false;
			}
		}
		return true;
	}

	static sol (n, lower) {
		const result = [];
		for (let i = 0; i < n; i++) {
			if (i % 11 === 0 || i % 13 === 0) {
				const str = i.toString();
				for (let j = 0; j < str.length; j++) {
					if (str[j] === "7") {
						result.push([i, j]);
					}
				}
			}
		}
		return result;
	}

	getExample () {
		return { n: 19723, lower: 1000 };
	}

	gen (target_num_instances) {
		let lower = 0;
		let n = 0;
		while (this.numGeneratedSoFar() < target_num_instances) {
			if (n % 11 === 0 || n % 13 === 0) {
				lower += n.toString().split("7").length - 1;
			}
			n++;
			if (this.random.randrange(10) === 0) {
				this.add({ n, lower });
			}
		}
	}
}

export class HalfSorted extends PuzzleGenerator {
	/** Inspired by HumanEval #37 */
	static docstring = "* Start with a list of integers, keep every other element in place and otherwise sort the list\n\n        Sample Input:\n        [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]\n\n        Sample Output:\n        [1, 0, 2, 2, 4, 8, 8, 8, 9, 3]";
	static sat (li, orig = [1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521]) {
		/**
		 * Start with a list of integers, keep every other element in place and otherwise sort the list
		 *
		 * Sample Input:
		 * [8, 0, 7, 2, 9, 4, 1, 2, 8, 3]
		 *
		 * Sample Output:
		 * [1, 0, 2, 2, 4, 8, 8, 8, 9, 3]
		 */
		const odds_orig = [];
		const odds_li = [];
		for (let i = 1; i < orig.length; i += 2) {
			odds_orig.push(orig[i]);
			odds_li.push(li[i]);
		}
		if (JSON.stringify(odds_orig) !== JSON.stringify(odds_li)) return false;

		const evens_orig = [];
		const evens_li = [];
		for (let i = 0; i < orig.length; i += 2) {
			evens_orig.push(orig[i]);
			evens_li.push(li[i]);
		}
		const sorted_evens_orig = [...evens_orig].sort((a, b) => a - b);
		return JSON.stringify(evens_li) === JSON.stringify(sorted_evens_orig);
	}

	static sol (orig) {
		const n = orig.length;
		const odds = [];
		for (let i = 1; i < n; i += 2) {
			odds.push(orig[i]);
		}
		const evens = [];
		for (let i = 0; i < n; i += 2) {
			evens.push(orig[i]);
		}
		evens.sort((a, b) => a - b);
		const ans = [];
		for (let i = 0; i < evens.length; i++) {
			ans.push(evens[i]);
			if (i < odds.length) {
				ans.push(odds[i]);
			}
		}
		return ans;
	}

	getExample () {
		return { orig: [1, 6, 3, 41, 19, 4, 12, 3, 18, 5, -29, 0, 19521] };
	}

	genRandom () {
		const list_length = this.random.randrange(20);
		const orig = Array.from({ length: list_length }, () =>
			this.random.randrange(-10, 10),
		);
		this.add({ orig });
	}
}

export class ThreeCycle extends PuzzleGenerator {
	/** Inspired by HumanEval #38 */
	static docstring = "* Given a target string, find a string s such that when each group of three consecutive characters is cycled\n        forward one character, you achieve the target string.\n\n        Sample Input:\n        \"This is a test\"\n\n        Sample Output:\n        'hiT is aste st'";
	static sat (s, target = "Hello world") {
		/**
		 * Given a target string, find a string s such that when each group of three consecutive characters is cycled
		 * forward one character, you achieve the target string.
		 *
		 * Sample Input:
		 * "This is a test"
		 *
		 * Sample Output:
		 * 'hiT is aste st'
		 */
		const cycle3 = (trip) => {
			if (trip.length !== 3) return trip;
			return trip[2] + trip.slice(0, 2);
		};

		const result = [];
		for (let i = 0; i < s.length; i += 3) {
			result.push(cycle3(s.slice(i, i + 3)));
		}
		return target === result.join("");
	}

	static sol (target) {
		const un_cycle3 = (trip) => {
			if (trip.length !== 3) return trip;
			return trip.slice(1, 3) + trip[0];
		};

		const result = [];
		for (let i = 0; i < target.length; i += 3) {
			result.push(un_cycle3(target.slice(i, i + 3)));
		}
		return result.join("");
	}

	getExample () {
		return { target: "Hello world" };
	}

	genRandom () {
		const target = this.random.pseudo_word(0, 30);
		this.add({ target });
	}
}

export class PrimeFib extends PuzzleGenerator {
	/**
	 * Inspired by HumanEval #39.
	 * Uses Ira Gessel's test: n is a Fibonacci number if and only if
	 * 5n^2 - 4 or 5n^2 + 4 is a perfect square.
	 */
	static docstring = "* Find a prime Fibonacci number bigger than a certain threshold, using Ira Gessel's test for Fibonacci numbers.\n\n        Sample Input:\n        10\n\n        Sample Output:\n        11";
	static isPerfectSquare (n) {
		if (n < 0n) return false;
		const root = PrimeFib.sqrtBigInt(n);
		return root * root === n;
	}

	static sqrtBigInt (value) {
		if (value < 0n) return null;
		if (value < 2n) return value;
		let x = value / 2n + 1n;
		let y = (x + value / x) / 2n;
		while (y < x) {
			x = y;
			y = (x + value / x) / 2n;
		}
		return x;
	}

	static isPrime (n) {
		if (n < 2n) return false;
		if (n === 2n || n === 3n) return true;
		if (n % 2n === 0n || n % 3n === 0n) return false;
		for (let i = 5n; i * i <= n; i += 6n) {
			if (n % i === 0n || n % (i + 2n) === 0n) return false;
		}
		return true;
	}

	static sat (n, lower = 123456) {
		// Convert to BigInt for precision
		const bn = BigInt(n);
		const bLower = BigInt(lower);

		const cond1 = 5n * bn * bn - 4n;
		const cond2 = 5n * bn * bn + 4n;

		const isFib =
			PrimeFib.isPerfectSquare(cond1) || PrimeFib.isPerfectSquare(cond2);
		if (!isFib) throw new Error("n must be a Fibonacci number");

		if (!PrimeFib.isPrime(bn)) throw new Error("n must be prime");

		return bn > bLower;
	}

	static sol (lower) {
		const bLower = BigInt(lower);
		let m = 2n;
		let n = 3n;

		while (true) {
			const next = m + n;
			m = n;
			n = next;

			if (n > bLower && PrimeFib.isPrime(n)) {
				// Return as Number if it's within safe range, otherwise keep as BigInt
				return n <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(n) : n;
			}
		}
	}

	genRandom () {
		const exponent = Math.floor(Math.random() * 20);
		const limit = 2 ** exponent;
		const lower = Math.floor(Math.random() * limit);
		return { lower };
	}

	getExample () {
		return { lower: 123456 };
	}
}

export class TripleZeroSum extends PuzzleGenerator {
	static docstring = "* Find the indices of three numbers that sum to 0 in a list.\n\n        --- Example input ---\n        [1, 2, 4, -3, 5]\n\n        --- Example output ---\n        [0, 1, 3]";
	/**
	 * Inspired by HumanEval #40
	 *
	 * Similar to but harder than PairZeroSum #43.
	 *
	 * This is a version of the classic 3SUM problem (https://en.wikipedia.org/wiki/3SUM).
	 */
	static sat (inds, nums = [12, 6, 41, 15, -10452, 18242, 10440, 6, 6, 6, 6]) {
		/**
		 * Find the indices of three numbers that sum to 0 in a list.
		 *
		 * --- Example input ---
		 * [1, 2, 4, -3, 5]
		 *
		 * --- Example output ---
		 * [0, 1, 3]
		 */
		if (inds.length !== 3) return false;
		const sum = inds.reduce((acc, i) => acc + nums[i], 0);
		return sum === 0;
	}

	static sol (nums) {
		// O(n^2) algorithm
		const inv = new Map();
		for (let i = nums.length - 1; i >= 0; i--) {
			inv.set(nums[i], i);
		}
		for (let i = 0; i < nums.length; i++) {
			const n = nums[i];
			if (inv.get(n) === i) {
				inv.delete(n);
			}
			for (let j = 0; j < i; j++) {
				const m = nums[j];
				if (inv.has(-m - n)) {
					const k = inv.get(-m - n);
					return [i, j, k].sort((a, b) => a - b);
				}
			}
		}
		throw new Error("No solution found");
	}

	getExample () {
		return { nums: [12, 6, 41, 15, -10452, 18242, 10440, 6, 6, 6, 6] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(2, 10) }, () =>
			this.random.randrange(-100, 100),
		);
		nums.push(-nums[0] - nums[1]);
		this.random.shuffle(nums);
		this.add({ nums });
	}
}

// ============ Classes 41-60 ============

export class NumPasses extends PuzzleGenerator {
	static docstring = "* Given n cars traveling East and n cars traveling West on a road, how many passings will there be?\n        A passing is when one car passes another. The East-bound cars all begin further West than the West-bound cars.\n\n        --Sample input--\n        2\n\n        --Sample output--\n        4";
	/** Inspired by HumanEval #41 */
	static sat (count, n = 981) {
		/**
		 * Given n cars traveling East and n cars traveling West on a road,
		 * how many passings will there be?
		 * A passing is when one car passes another. The East-bound cars all
		 * begin further West than the West-bound cars.
		 *
		 * -- Sample input --
		 * 2
		 *
		 * -- Sample output --
		 * 4
		 */
		let temp = count;
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				temp -= 1;
			}
		}
		return temp === 0;
	}

	static sol (n) {
		return n * n;
	}

	getExample () {
		return { n: 981 };
	}

	genRandom () {
		const n = this.random.randrange(1000);
		this.add({ n });
	}
}

export class ListInc extends PuzzleGenerator {
	static docstring = "* Decrement each element of new_list by 1 and check that it's old_list\n\n        Sample Input:\n        [17, 15, 99]\n\n        Sample Output:\n        [18, 16, 100]";
	/**
	 * Increment each element of a list by 1
	 * Inspired by HumanEval #42
	 */
	static sat (newList, oldList = [321, 12, 532, 129, 9, -12, 4, 56, 90, 0]) {
		/**
		 * Decrement each element of new_list by 1 and check that it's old_list
		 *
		 * Sample Input:
		 * [17, 15, 99]
		 *
		 * Sample Output:
		 * [18, 16, 100]
		 */
		return (
			newList.length === oldList.length &&
			newList.every((x, i) => x - 1 === oldList[i])
		);
	}

	static sol (oldList) {
		return oldList.map((i) => i + 1);
	}

	getExample () {
		return { oldList: [321, 12, 532, 129, 9, -12, 4, 56, 90, 0] };
	}

	genRandom () {
		const oldList = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(100),
		);
		this.add({ oldList });
	}
}

export class PairZeroSum extends PuzzleGenerator {
	static docstring = "* Find the indices of two numbers that sum to 0 in a list.\n\n        Sample Input:\n        [1, -4, -4, 7, -3]\n\n        Sample Output:\n        [1, 2]";
	/**
	 * Inspired by HumanEval #43
	 * Similar to TripleZeroSum #40
	 */
	static sat (
		inds,
		nums = [12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20],
	) {
		/**
		 * Find the indices of two numbers that sum to 0 in a list.
		 *
		 * Sample Input:
		 * [1, -4, -4, 7, -3]
		 *
		 * Sample Output:
		 * [1, 2]
		 */
		if (inds.length !== 2) return false;
		const [a, b] = inds;
		return nums[a] + nums[b] === 0 && a >= 0 && b >= 0;
	}

	static sol (nums) {
		const s = new Set(nums);
		for (const i of s) {
			if (s.has(-i)) {
				return [nums.indexOf(i), nums.indexOf(-i)];
			}
		}
	}

	getExample () {
		return { nums: [12, -10452, 18242, 10440, 81, 241, 525, -18242, 91, 20] };
	}

	genRandom () {
		const n = this.random.randrange(1, 100);
		const nums = Array.from({ length: n }, () => this.random.randrange(-n, n));
		if (!nums.includes(0)) {
			nums.push(-nums[this.random.randrange(nums.length)]);
		}
		this.random.shuffle(nums);
		this.add({ nums });
	}
}

export class ChangeBase extends PuzzleGenerator {
	static docstring = "* Write n in the given base as a string\n\n        Sample Input:\n        n=23, base=12\n\n        Sample Output:\n        '1A'";
	/** Inspired by HumanEval #44 */
	static sat (s, n = 142, base = 7) {
		/**
		 * Write n in the given base as a string
		 *
		 * Sample Input:
		 * n=23, base=12
		 *
		 * Sample Output:
		 * '1A'
		 */
		return parseInt(s, base) === n;
	}

	static sol (n, base) {
		if (n === 0) return "0";
		let ans = "";
		while (n > 0) {
			ans = (n % base).toString() + ans;
			n = Math.floor(n / base);
		}
		return ans;
	}

	getExample () {
		return { n: 142, base: 7 };
	}

	genRandom () {
		const n = this.random.randrange(1, 10 ** 7);
		const base = this.random.randrange(2, 11);
		this.add({ n, base });
	}
}

export class TriangleArea extends PuzzleGenerator {
	static docstring = "* Find the height of a triangle given the area and base. It is guaranteed that the answer is an integer.\n\n        Sample Input:\n        area = 6, base = 3\n\n        Sample Output:\n        4";
	/** Inspired by HumanEval #45 */
	static sat (height, area = 1319098728582, base = 45126) {
		/**
		 * Find the height of a triangle given the area and base.
		 * It is guaranteed that the answer is an integer.
		 *
		 * Sample Input:
		 * area = 6, base = 3
		 *
		 * Sample Output:
		 * 4
		 */
		return base * height === 2 * area;
	}

	static sol (area, base) {
		return Math.floor((2 * area) / base);
	}

	getExample () {
		return { area: 1319098728582, base: 45126 };
	}

	genRandom () {
		const base = this.random.randrange(1, 10 ** this.random.randrange(1, 10));
		const height = this.random.randrange(1, 10 ** this.random.randrange(1, 10));
		const area = Math.floor((base * height) / 2);
		if (base * height === 2 * area) {
			this.add({ area, base });
		}
	}
}

export class Fib4 extends PuzzleGenerator {
	static docstring = "* Define a four-wise Fibonacci sequence to be a sequence such that each number is the sum of the previous\n        four. Given a target number, find an initial four numbers such that the 100th number in the sequence is the\n        given target number.\n\n        Sample Input:\n        0\n\n        Sample Output:\n        [0, 0, 0, 0]";
	/**
	 * Inspired by HumanEval #46
	 * Almost identical to problem 63
	 */
	static sat (init, target = 2021) {
		/**
		 * Define a four-wise Fibonacci sequence to be a sequence such that
		 * each number is the sum of the previous four. Given a target number,
		 * find an initial four numbers such that the 100th number in the
		 * sequence is the given target number.
		 *
		 * Sample Input:
		 * 0
		 *
		 * Sample Output:
		 * [0, 0, 0, 0]
		 */
		if (init.length !== 4) return false;
		let [a, b, c, d] = init;
		for (let i = 0; i < 99; i++) {
			[a, b, c, d] = [b, c, d, a + b + c + d];
		}
		return a === target;
	}

	static sol (target) {
		const nums = [target, 0, 0, 0];
		for (let i = 0; i < 99; i++) {
			const x = nums[3] - (nums[0] + nums[1] + nums[2]);
			nums.unshift(x);
			nums.pop();
		}
		return nums;
	}

	getExample () {
		return { target: 2021 };
	}

	genRandom () {
		const target = this.random.randrange(10 ** this.random.randrange(10));
		this.add({ target });
	}
}

export class Median extends PuzzleGenerator {
	/**
	 * One definition of the median is a number that minimizes the sum of
	 * absolute deviations. When there are an even number of items,
	 * there is an interval of valid solutions.
	 * Inspired by HumanEval #47
	 */
	static docstring = "* Find an integer that minimizes the sum of absolute deviations with respect to the given numbers.\n\n        Sample Input:\n        [3, 6, 1, 2, 5, 4, 100], upper=105\n\n        Sample Output:\n        4";
	static sat (
		x,
		nums = [132666041, 237412, 28141, -12, 11939, 912414, 17],
		upper = 133658965,
	) {
		/**
		 * Find an integer that minimizes the sum of absolute deviations
		 * with respect to the given numbers.
		 *
		 * Sample Input:
		 * [3, 6, 1, 2, 5, 4, 100], upper=105
		 *
		 * Sample Output:
		 * 4
		 */
		const dev = nums.reduce((sum, n) => sum + (n - x), 0);
		return dev <= upper;
	}

	static sol (nums, upper) {
		if (nums.length === 0) return 0;
		const sorted = [...nums].sort((a, b) => a - b);
		return sorted[Math.floor(nums.length / 2)];
	}

	getExample () {
		return {
			nums: [132666041, 237412, 28141, -12, 11939, 912414, 17],
			upper: 133658965,
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-(10 ** 10), 10 ** 10),
		);
		if (nums.length === 0) {
			this.add({ nums, upper: 0 });
			return;
		}
		const sorted = [...nums].sort((a, b) => a - b);
		const x = sorted[Math.floor(nums.length / 2)];
		const upper = nums.reduce((sum, n) => sum + Math.abs(n - x), 0);
		this.add({ nums, upper });
	}
}

export class Palindrome extends PuzzleGenerator {
	/** Inspired by HumanEval #48 */
	static docstring = "* Test whether the given strings are palindromes\n\n        Sample Input:\n        [\"aba\", \"no\"]\n\n        Sample Output:\n        [True, False]";
	static sat (
		pals,
		strs = ["palindrome", "madamimadam", "", "foo", "eyes", "(-:-)"],
	) {
		/**
		 * Test whether the given strings are palindromes
		 *
		 * Sample Input:
		 * ["aba", "no"]
		 *
		 * Sample Output:
		 * [true, false]
		 */
		return (
			pals.length === strs.length &&
			pals.every(
				(p, i) => p === (strs[i] === strs[i].split("").reverse().join("")),
			)
		);
	}

	static sol (strs) {
		return strs.map((s) => s === s.split("").reverse().join(""));
	}

	getExample () {
		return { strs: ["palindrome", "madamimadam", "", "foo", "eyes", "(-:-)"] };
	}

	genRandom () {
		const strs = [];
		for (let i = 0; i < this.random.randrange(10); i++) {
			const s = this.random.pseudoWord();
			const choice = this.random.randrange(3);
			if (choice === 0) {
				strs.push(s);
			} else if (choice === 1) {
				strs.push(s + s.split("").reverse().join(""));
			} else {
				const rev = s.slice(0, -1).split("").reverse().join("");
				strs.push(s + rev);
			}
		}
		this.add({ strs });
	}
}

export class LittleFermat extends PuzzleGenerator {
	/**
	 * Harder but loosely inspired by HumanEval #49
	 */
	static docstring = "* Fermat's little theorem implies that any polynomial can be written equivalently as a degree p-1\n        polynomial (mod p).\n        Given the p coefficients of a polynomial poly, compute a polynomial equivalent to poly^d (mod p).\n\n        Sample Input:\n        d=2, poly=[1, 0, 0, 1, 0]  # 1 + x^3\n\n        Sample Output:\n        [1, 0, 1, 2, 0]  # 1+ x^2 + 2x^3 because (1 + x^3)^2 = 1 + 2x^3 + x^6 and x^6 = x^2 (mod 5)";
	static sat (expPoly, d = 74152093423, poly = [1, 6, 3, 1, 0, 4, 4]) {
		/**
		 * Fermat's little theorem implies that any polynomial can be written
		 * equivalently as a degree p-1 polynomial (mod p).
		 * Given the p coefficients of a polynomial poly, compute a polynomial
		 * equivalent to poly^d (mod p).
		 *
		 * Sample Input:
		 * d=2, poly=[1, 0, 0, 1, 0]  # 1 + x^3
		 *
		 * Sample Output:
		 * [1, 0, 1, 2, 0]  # 1+ x^2 + 2x^3 because (1 + x^3)^2 = 1 + 2x^3 + x^6
		 * and x^6 = x^2 (mod 5)
		 */
		const p = poly.length;
		if (p <= 2) return false;
		for (let i = 2; i < p; i++) {
			if (p % i === 0) return false; // p must be prime
		}

		const modPow = (base, exp, mod) => {
			let result = 1;
			base = base % mod;
			while (exp > 0) {
				if (exp % 2 === 1) result = (result * base) % mod;
				base = (base * base) % mod;
				exp = Math.floor(exp / 2);
			}
			return result;
		};

		const val = (coeffs, n) => {
			let result = 0;
			for (let i = 0; i < coeffs.length; i++) {
				result = (result + coeffs[i] * modPow(n, i, p)) % p;
			}
			return result;
		};

		for (let n = 0; n < p; n++) {
			const left = val(expPoly, n);
			const right = modPow(val(poly, n), d, p);
			if (left !== right) return false;
		}
		return true;
	}

	static sol (d, poly) {
		const p = poly.length;

		const prod = (poly1, poly2) => {
			const ans = Array(p).fill(0);
			for (let i = 0; i < poly1.length; i++) {
				for (let j = 0; j < poly2.length; j++) {
					let e = (i + j) % (p - 1);
					if (e === 0 && i + j > 1) {
						e = p - 1;
					}
					ans[e] = (ans[e] + poly1[i] * poly2[j]) % p;
				}
			}
			return ans;
		};

		let ans = [1, ...Array(p - 1).fill(0)];
		let base = [...poly];
		let exp = d;

		while (exp > 0) {
			if (exp % 2 === 1) {
				ans = prod(ans, base);
			}
			base = prod(base, base);
			exp = Math.floor(exp / 2);
		}
		return ans;
	}

	getExample () {
		return { d: 74152093423, poly: [1, 6, 3, 1, 0, 4, 4] };
	}

	genRandom () {
		const primes = [3, 5, 7, 11];
		const p = primes[this.random.randrange(primes.length)];
		const poly = Array.from({ length: p }, () => this.random.randrange(p));
		const d = this.random.randrange(2 ** this.random.randrange(100));
		this.add({ d, poly });
	}
}

export class ShiftChars extends PuzzleGenerator {
	/** Inspired by HumanEval #50 */
	static docstring = "* Find a string which, when each character is shifted (ascii incremented) by shift, gives the result.\n\n        Sample Input:\n        result='very good', shift=-1\n\n        Sample Output:\n        'wfsz!hppe'";
	static sat (orig, result = "Hello, world!", shift = 7) {
		/**
		 * Find a string which, when each character is shifted (ascii incremented)
		 * by shift, gives the result.
		 *
		 * Sample Input:
		 * result='very good', shift=-1
		 *
		 * Sample Output:
		 * 'wfsz!hppe'
		 */
		if (orig.length !== result.length) return false;
		for (let i = 0; i < result.length; i++) {
			if (orig.charCodeAt(i) + shift !== result.charCodeAt(i)) {
				return false;
			}
		}
		return true;
	}

	static sol (result, shift) {
		return Array.from(result, (c) =>
			String.fromCharCode(c.charCodeAt(0) - shift),
		).join("");
	}

	getExample () {
		return { result: "Hello, world!", shift: 7 };
	}

	genRandom () {
		const result = this.random.pseudoWord();
		const shift = this.random.randrange(-11, 11);
		this.add({ result, shift });
	}
}

export class RemoveVowels extends PuzzleGenerator {
	/**
	 * Inspired by HumanEval #51
	 * Related to FindVowels #54
	 */
	static docstring = "* Remove the vowels from the original string.\n\n        Sample Input:\n        \"very good\"\n\n        Sample Output:\n        'vry gd'";
	static sat (txt, text = "Hello, world!") {
		/**
		 * Remove the vowels from the original string.
		 *
		 * Sample Input:
		 * "very good"
		 *
		 * Sample Output:
		 * 'vry gd'
		 */
		let n = 0;
		for (const c of text) {
			if (!"aeiouAEIOU".includes(c)) {
				if (txt[n] !== c) return false;
				n++;
			}
		}
		return n === txt.length;
	}

	static sol (text) {
		return Array.from(text, (c) => (!"aeiouAEIOU".includes(c) ? c : "")).join(
			"",
		);
	}

	getExample () {
		return { text: "Hello, world!" };
	}

	genRandom () {
		const text = this.random.pseudoWord();
		this.add({ text });
	}
}

export class BelowThreshold extends PuzzleGenerator {
	/** Inspired by HumanEval #52 */
	static docstring = "* Find the indexes of numbers below a given threshold\n\n        Sample Input:\n        nums=[4, 7, 11, 5], threshold=10\n\n        Sample Output:\n        [0, 1, 3]";
	static sat (
		indexes,
		nums = [0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55],
		thresh = 100,
	) {
		/**
		 * Find the indexes of numbers below a given threshold
		 *
		 * Sample Input:
		 * nums=[4, 7, 11, 5], threshold=10
		 *
		 * Sample Output:
		 * [0, 1, 3]
		 */
		let j = 0;
		for (let i = 0; i < nums.length; i++) {
			if (nums[i] < thresh) {
				if (indexes[j] !== i) return false;
				j++;
			}
		}
		return j === indexes.length;
	}

	static sol (nums, thresh) {
		return nums.map((n, i) => (n < thresh ? i : -1)).filter((i) => i !== -1);
	}

	getExample () {
		return { nums: [0, 2, 17, 4, 4213, 322, 102, 29, 15, 39, 55], thresh: 100 };
	}

	genRandom () {
		const thresh = this.random.randrange(-100, 100);
		const nums = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ nums, thresh });
	}
}

export class ListTotal extends PuzzleGenerator {
	/** Inspired by HumanEval #53 */
	static docstring = "* Find the number which when appended to the list makes the total 0\n\n        Sample Input:\n        [1, 2, 3]\n\n        Sample Output:\n        -6";
	static sat (n, nums = [10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755]) {
		/**
		 * Find the number which when appended to the list makes the total 0
		 *
		 * Sample Input:
		 * [1, 2, 3]
		 *
		 * Sample Output:
		 * -6
		 */
		return [...nums, -n].reduce((a, b) => a + b, 0) === 0;
	}

	static sol (nums) {
		return nums.reduce((a, b) => a + b, 0);
	}

	getExample () {
		return { nums: [10, 42, 17, 9, 1315182, 184, 102, 29, 15, 39, 755] };
	}

	genRandom () {
		const m = 10 ** this.random.randrange(10);
		const nums = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-m, m),
		);
		this.add({ nums });
	}
}

export class DiffChars extends PuzzleGenerator {
	/** Inspired by HumanEval #54 */
	static docstring = "* Find a character in one string that is not in the other.\n\n        Sample Input:\n        'Do you like green eggs and ham?', 'I do not like green eggs and ham.'\n\n        Sample Output:\n        't'  # or .?yI";
	static sat (
		c,
		a = "the quick brown fox jumped over the lazy dog",
		b = "how vexingly quick daft zebras jump",
	) {
		/**
		 * Find a character in one string that is not in the other.
		 *
		 * Sample Input:
		 * 'Do you like green eggs and ham?', 'I do not like green eggs and ham.'
		 *
		 * Sample Output:
		 * 't'  # or .?yI
		 */
		const inA = a.includes(c);
		const inB = b.includes(c);
		return inA !== inB;
	}

	static sol (a, b) {
		const setA = new Set(a);
		const setB = new Set(b);
		const diff = [...setA]
			.filter((c) => !setB.has(c))
			.concat([...setB].filter((c) => !setA.has(c)));
		return diff.sort()[0];
	}

	getExample () {
		return {
			a: "the quick brown fox jumped over the lazy dog",
			b: "how vexingly quick daft zebras jump",
		};
	}

	genRandom () {
		const a = this.random.pseudoWord();
		const b =
			this.random.randrange(2) === 0
				? this.random.pseudoWord()
				: [...a, "m"].sort().join("");

		const setA = new Set(a);
		const setB = new Set(b);
		if (setA.size !== setB.size || ![...setA].every((c) => setB.has(c))) {
			this.add({ a, b });
		}
	}
}

export class Fibonacci extends PuzzleGenerator {
	/** Inspired by HumanEval #55 */
	static docstring = "* Find the first n Fibonacci numbers\n\n        Sample Input:\n        4\n\n        Sample Output:\n        [1, 1, 2, 3]";
	static sat (nums, n = 1402) {
		/**
		 * Find the first n Fibonacci numbers
		 *
		 * Sample Input:
		 * 4
		 *
		 * Sample Output:
		 * [1, 1, 2, 3]
		 */
		if (nums.length !== n) return false;
		if (nums[0] !== 1 || nums[1] !== 1) return false;
		for (let i = 0; i < n - 2; i++) {
			if (nums[i + 2] !== nums[i + 1] + nums[i]) return false;
		}
		return true;
	}

	static sol (n) {
		const ans = [1, 1];
		while (ans.length < n) {
			ans.push(ans[ans.length - 1] + ans[ans.length - 2]);
		}
		return ans;
	}

	getExample () {
		return { n: 1402 };
	}

	genRandom () {
		const n = this.random.randrange(12000);
		this.add({ n });
	}
}

export class MatchBrackets extends PuzzleGenerator {
	/** Inspired by HumanEval #56 */
	static docstring = "* Find the index of the matching brackets for each character in the string\n\n        Sample Input:\n        \"<><>\"\n\n        Sample Output:\n        [1, 0, 3, 2]";
	static sat (matches, brackets = "<<>><<<><>><<>>>") {
		/**
		 * Find the index of the matching brackets for each character in the string
		 *
		 * Sample Input:
		 * "<><>"
		 *
		 * Sample Output:
		 * [1, 0, 3, 2]
		 */
		if (matches.length !== brackets.length) return false;
		for (let i = 0; i < brackets.length; i++) {
			const j = matches[i];
			const c = brackets[i];
			if (brackets[j] === c) return false;
			if (matches[j] !== i) return false;
			for (let k = i + 1; k < j; k++) {
				if (matches[k] <= i || matches[k] >= j) return false;
			}
		}
		return true;
	}

	static sol (brackets) {
		const matches = Array(brackets.length).fill(-1);
		const opens = [];
		for (let i = 0; i < brackets.length; i++) {
			if (brackets[i] === "<") {
				opens.push(i);
			} else {
				const j = opens.pop();
				matches[i] = j;
				matches[j] = i;
			}
		}
		return matches;
	}

	getExample () {
		return { brackets: "<<>><<<><>><<>>>" };
	}

	genRandom () {
		let depth = 0;
		let brackets = "";
		while (depth > 0 || this.random.random() > 0.2) {
			const c = depth > 0 ? this.random.choice(["<", ">", ">"]) : "<";
			if (c === "<") {
				depth++;
			} else {
				depth--;
			}
			brackets += c;
		}
		this.add({ brackets });
	}
}

export class Monotonic extends PuzzleGenerator {
	/** Inspired by HumanEval #57 */
	static docstring = "* Determine the direction ('increasing' or 'decreasing') of monotonic sequence nums\n\n        Sample Input:\n        [1, 2, 5]\n\n        Sample Output:\n        \"increasing\"";
	static sat (direction, nums = [2, 4, 17, 29, 31, 1000, 416629]) {
		/**
		 * Determine the direction ('increasing' or 'decreasing') of monotonic sequence nums
		 *
		 * Sample Input:
		 * [1, 2, 5]
		 *
		 * Sample Output:
		 * "increasing"
		 */
		if (direction === "increasing") {
			return nums.every((n, i) => i === nums.length - 1 || n < nums[i + 1]);
		}
		if (direction === "decreasing") {
			return nums.every((n, i) => i === nums.length - 1 || nums[i + 1] < n);
		}
		return false;
	}

	static sol (nums) {
		if (nums.length > 1 && nums[1] > nums[0]) {
			return "increasing";
		}
		return "decreasing";
	}

	getExample () {
		return { nums: [2, 4, 17, 29, 31, 1000, 416629] };
	}

	genRandom () {
		const reverse = this.random.randrange(2) === 0;
		const nums = [
			...new Set(
				Array.from({ length: this.random.randrange(10) }, () =>
					this.random.randrange(1000),
				),
			),
		].sort((a, b) => (reverse ? b - a : a - b));
		this.add({ nums });
	}
}

export class CommonNumbers extends PuzzleGenerator {
	/** Inspired by HumanEval #58 */
	static docstring = "* Find numbers common to a and b\n\n        Sample Input:\n        [1, 2, 3], [3, 4, 5]\n\n        Sample Output:\n        [3]";
	static sat (
		common,
		a = [2, 416629, 2, 4, 17, 29, 31, 1000],
		b = [31, 2, 4, 17, 29, 41205],
	) {
		/**
		 * Find numbers common to a and b
		 *
		 * Sample Input:
		 * [1, 2, 3], [3, 4, 5]
		 *
		 * Sample Output:
		 * [3]
		 */
		const setA = new Set(a);
		const setB = new Set(b);
		const setCommon = new Set(common);
		const all = [...new Set([...a, ...b, ...common])];
		return all.every((i) => {
			const inCommon = setCommon.has(i);
			const inAandB = setA.has(i) && setB.has(i);
			return inCommon === inAandB;
		});
	}

	static sol (a, b) {
		const setB = new Set(b);
		return [...new Set(a)].filter((x) => setB.has(x)).sort((x, y) => x - y);
	}

	getExample () {
		return {
			a: [2, 416629, 2, 4, 17, 29, 31, 1000],
			b: [31, 2, 4, 17, 29, 41205],
		};
	}

	genRandom () {
		const common = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(1000),
		);
		const a = [
			...common,
			...Array.from({ length: this.random.randrange(10) }, () =>
				this.random.randrange(1000),
			),
		];
		const b = [
			...common,
			...Array.from({ length: this.random.randrange(10) }, () =>
				this.random.randrange(1000),
			),
		];
		this.random.shuffle(a);
		this.random.shuffle(b);
		this.add({ a, b });
	}
}

export class LargestPrimeFactor extends PuzzleGenerator {
	/** Inspired by HumanEval #59 */
	static docstring = "* Find the largest prime factor of n.\n\n        Sample Input:\n        125\n\n        Sample Output:\n        5";
	static sat (p, n = 101076) {
		/**
		 * Find the largest prime factor of n.
		 *
		 * Sample Input:
		 * 125
		 *
		 * Sample Output:
		 * 5
		 */
		const isPrime = (m) => {
			if (m < 2) return false;
			for (let i = 2; i < m - 1; i++) {
				if (m % i === 0) return false;
			}
			return true;
		};

		if (!isPrime(p) || n % p !== 0 || p <= 0) return false;
		for (let i = p + 1; i < n; i++) {
			if (n % i === 0 && isPrime(i)) return false;
		}
		return true;
	}

	static sol (n) {
		const isPrime = (m) => {
			if (m < 2) return false;
			for (let i = 2; i < m - 1; i++) {
				if (m % i === 0) return false;
			}
			return true;
		};

		for (let i = 1; i < n; i++) {
			if (n % i === 0 && isPrime(n / i)) {
				return n / i;
			}
		}
	}

	getExample () {
		return { n: 101076 };
	}

	genRandom () {
		const n = this.random.randrange(2, 100 * 1000);
		this.add({ n });
	}
}

export class CumulativeSums extends PuzzleGenerator {
	static docstring = "* Find the sums of the integers from 1 to n\n\n        Sample Input:\n        3\n\n        Sample Output:\n        [0, 1, 3, 6]";
	/** Inspired by HumanEval #60 */
	static sat (sums, n = 104) {
		/**
		 * Find the sums of the integers from 1 to n
		 *
		 * Sample Input:
		 * 3
		 *
		 * Sample Output:
		 * [0, 1, 3, 6]
		 */
		if (sums.length !== n + 1) return false;
		if (sums[0] !== 0) return false;
		for (let i = 0; i < n; i++) {
			if (sums[i + 1] - sums[i] !== i) return false;
		}
		return true;
	}

	static sol (n) {
		const ans = [0];
		for (let i = 0; i < n; i++) {
			ans.push(ans[ans.length - 1] + i);
		}
		return ans;
	}

	getExample () {
		return { n: 104 };
	}

	genRandom () {
		const n = this.random.randrange(20 * 1000);
		this.add({ n });
	}
}

export class ParenDepth extends PuzzleGenerator {
	static docstring = "* Find the index of the matching parentheses for each character in the string\n\n        Sample Input:\n        \"()((()))\"\n\n        Sample Output:\n        [1, 0, 7, 6, 5, 4, 3, 2]";
	/**
	 * Inspired by HumanEval #61
	 * Note that problems 61 and 56 are essentially the same
	 */
	static sat (matches, parens = "((())()(()()))(())") {
		/**
		 * Find the index of the matching parentheses for each character in the string
		 *
		 * Sample Input:
		 * "()((()))"
		 *
		 * Sample Output:
		 * [1, 0, 7, 6, 5, 4, 3, 2]
		 */
		for (let i = 0; i < matches.length; i++) {
			const j = matches[i];
			const c = parens[i];
			if (parens[j] === c || matches[j] !== i) return false;
			for (let k = i + 1; k < j; k++) {
				if (!(i < matches[k] && matches[k] < j)) return false;
			}
		}
		return matches.length === parens.length;
	}

	static sol (parens) {
		const matches = Array(parens.length).fill(-1);
		const opens = [];
		for (let i = 0; i < parens.length; i++) {
			if (parens[i] === "(") {
				opens.push(i);
			} else {
				const j = opens.pop();
				matches[i] = j;
				matches[j] = i;
			}
		}
		return matches;
	}

	getExample () {
		return { parens: "((())()(()()))(())" };
	}

	genRandom () {
		let depth = 0;
		let parens = "";
		while (depth > 0 || this.random.random() > 0.3) {
			const c = this.random.choice(depth > 0 ? "())" : "(");
			if (c === "(") depth++;
			else if (c === ")") depth--;
			parens += c;
		}
		this.add({ parens });
	}
}

export class Derivative extends PuzzleGenerator {
	static docstring = "* Find the derivative of the given polynomial, with coefficients in order of increasing degree\n\n        Sample Input:\n        [3, 4, 1] # 3 + 4x + x^2\n\n        Sample Output:\n        [2, 4]   # 4 + 2x^2";
	/**
	 * Inspired by HumanEval #62
	 * This puzzle gives the raw definition of a derivative in terms of small changes in x.
	 */
	static sat (derivative, poly = [2, 1, 0, 4, 19, 231, 0, 5]) {
		/**
		 * Find the derivative of the given polynomial, with coefficients in order of increasing degree
		 *
		 * Sample Input:
		 * [3, 4, 1] # 3 + 4x + x^2
		 *
		 * Sample Output:
		 * [2, 4]   # 4 + 2x^2
		 */
		const val = (p, x) => p.reduce((sum, coeff, i) => sum + coeff * x ** i, 0);
		for (let x = 0; x < poly.length; x++) {
			const delta = 1e-8;
			const actual = val(poly, x + delta) - val(poly, x);
			const estimated = delta * val(derivative, x);
			if (Math.abs(actual - estimated) >= 1e-4) return false;
		}
		return true;
	}

	static sol (poly) {
		return Array.from(
			{ length: poly.length - 1 },
			(_, i) => (i + 1) * poly[i + 1],
		);
	}

	getExample () {
		return { poly: [2, 1, 0, 4, 19, 231, 0, 5] };
	}

	genRandom () {
		const poly = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-10, 10),
		);
		this.add({ poly });
	}
}

export class Fib3 extends PuzzleGenerator {
	/**
	 * Inspired by HumanEval #63
	 * Almost identical to problem 46
	 */
	static docstring = "* Define a triple-Fibonacci sequence to be a sequence such that each number is the sum of the previous\n        three. Given a target number, find an initial triple such that the 17th number in the sequence is the\n        given target number.\n\n        Sample Input:\n        0\n\n        Sample Output:\n        [0, 0, 0]";
	static sat (init, target = 124156) {
		/**
		 * Define a triple-Fibonacci sequence to be a sequence such that each number is the sum of the previous
		 * three. Given a target number, find an initial triple such that the 17th number in the sequence is the
		 * given target number.
		 *
		 * Sample Input:
		 * 0
		 *
		 * Sample Output:
		 * [0, 0, 0]
		 */
		let a = init[0],
			b = init[1],
			c = init[2];
		for (let i = 0; i < 16; i++) {
			[a, b, c] = [b, c, a + b + c];
		}
		return a === target;
	}

	static sol (target) {
		const nums = [target, 0, 0];
		for (let i = 0; i < 16; i++) {
			const x =
				nums[nums.length - 1] - nums.slice(0, -1).reduce((a, b) => a + b, 0);
			nums.unshift(x);
			nums.pop();
		}
		return nums;
	}

	getExample () {
		return { target: 124156 };
	}

	genRandom () {
		const target = this.random.randrange(2 ** this.random.randrange(15));
		this.add({ target });
	}
}

export class FindVowels extends PuzzleGenerator {
	/**
	 * Inspired by HumanEval #64
	 * Very similar to RemoveVowels #51
	 */
	static docstring = "* Find the vowels from each of the original texts (y counts as a vowel at the end of the word)\n\n        Sample Input:\n        [\"You can do it!\", \"CAT\"]\n\n        Sample Output:\n        [\"ouaoi\", \"A\"]";
	static sat (vowels, texts = ["Hello, world!", "Goodbye, world!"]) {
		/**
		 * Find the vowels from each of the original texts (y counts as a vowel at the end of the word)
		 *
		 * Sample Input:
		 * ["You can do it!", "CAT"]
		 *
		 * Sample Output:
		 * ["ouaoi", "A"]
		 */
		for (let idx = 0; idx < vowels.length; idx++) {
			const v = vowels[idx];
			const t = texts[idx];
			let i = 0;
			for (let j = 0; j < t.length; j++) {
				const c = t[j];
				const lower = c.toLowerCase();
				if ("aeiou".includes(lower) || (lower === "y" && j === t.length - 1)) {
					if (v[i] !== c) return false;
					i++;
				}
			}
			if (i !== v.length) return false;
		}
		return vowels.length === texts.length;
	}

	static sol (texts) {
		return texts.map((text) => {
			let result = "";
			for (let i = 0; i < text.length; i++) {
				const c = text[i];
				const lower = c.toLowerCase();
				if ("aeiou".includes(lower)) result += c;
				else if (lower === "y" && i === text.length - 1) result += c;
			}
			return result;
		});
	}

	getExample () {
		return { texts: ["Hello, world!", "Goodbye, world!"] };
	}

	genRandom () {
		const texts = Array.from({ length: this.random.randrange(10) }, () => {
			const word = this.random.pseudoWord();
			return Array.from(word)
				.map((c) => this.random.choice([c.toLowerCase(), c.toUpperCase()]))
				.join("");
		});
		this.add({ texts });
	}
}

export class CircularShiftNum extends PuzzleGenerator {
	/** Inspired by HumanEval #65 */
	static docstring = "* Shift the decimal digits n places to the left, wrapping the extra digits around. If shift > the number of\n        digits of n, reverse the string.\n\n        n=12345 shift=2 => '34512'";
	static sat (shifted, n = 124582369835, shift = 3) {
		/**
		 * Shift the decimal digits n places to the left, wrapping the extra digits around. If shift > the number of
		 * digits of n, reverse the string.
		 *
		 * n=12345 shift=2 => '34512'
		 */
		const s = String(n);
		if (shift > s.length) {
			return n === parseInt(shifted.split("").reverse().join(""));
		}
		return n === parseInt(shifted.slice(-shift) + shifted.slice(0, -shift));
	}

	static sol (n, shift) {
		const s = String(n);
		if (shift > s.length) {
			return s.split("").reverse().join("");
		}
		return s.slice(shift) + s.slice(0, shift);
	}

	getExample () {
		return { n: 124582369835, shift: 3 };
	}

	genRandom () {
		const n = this.random.randrange(10 ** this.random.randrange(30));
		const shift = this.random.randrange(31);
		this.add({ n, shift });
	}
}

export class CharSum extends PuzzleGenerator {
	/** Inspired by HumanEval #66 */
	static docstring = "* Compute the sum of the ASCII values of the upper-case characters in the string.\n\n        Sample Input:\n        ARt\n\n        Sample Output:\n        147 # = 65 + 82";
	static sat (tot, s = "Add ME uP AND YOU WILL GET A BIG NUMBER!") {
		/**
		 * Compute the sum of the ASCII values of the upper-case characters in the string.
		 *
		 * Sample Input:
		 * ARt
		 *
		 * Sample Output:
		 * 147 # = 65 + 82
		 */
		let sum = 0;
		for (const c of s) {
			if (c === c.toUpperCase() && c !== c.toLowerCase()) {
				sum += c.charCodeAt(0);
			}
		}
		return tot === sum;
	}

	static sol (s) {
		let sum = 0;
		for (const c of s) {
			if (c === c.toUpperCase() && c !== c.toLowerCase()) {
				sum += c.charCodeAt(0);
			}
		}
		return sum;
	}

	getExample () {
		return { s: "Add ME uP AND YOU WILL GET A BIG NUMBER!" };
	}

	genRandom () {
		const s = this.random.string({ minLen: 0 });
		this.add({ s });
	}
}

export class MissingBananas extends PuzzleGenerator {
	/** Inspired by HumanEval #67 */
	static docstring = "* Determine how many bananas are necessary to reach a certain total amount of fruit\n\n        bowl=\"3 apples and 4 oranges\", total=12 => 5";
	static sat (
		bananas,
		bowl = "5024 apples and 12189 oranges",
		total = 12491241,
	) {
		/**
		 * Determine how many bananas are necessary to reach a certain total amount of fruit
		 *
		 * bowl="3 apples and 4 oranges", total=12 => 5
		 */
		const fullBowl = bowl + ` and ${bananas} bananas`;
		const nums = fullBowl
			.split(" ")
			.filter((s) => /^\d+$/.test(s))
			.map(Number);
		return nums.reduce((a, b) => a + b, 0) === total;
	}

	static sol (bowl, total) {
		const nums = bowl
			.split(" ")
			.filter((s) => /^\d+$/.test(s))
			.map(Number);
		const sum = nums.reduce((a, b) => a + b, 0);
		return total - sum;
	}

	getExample () {
		return { bowl: "5024 apples and 12189 oranges", total: 12491241 };
	}

	genRandom () {
		const digits = this.random.choice([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
		const apples = this.random.randrange(10 ** digits);
		const oranges = this.random.randrange(10 ** digits);
		const bananas = this.random.randrange(10 ** digits);
		const bowl = `${apples} apples and ${oranges} oranges`;
		const total = apples + oranges + bananas;
		this.add({ bowl, total });
	}
}

export class SmallestEven extends PuzzleGenerator {
	/** Inspired by HumanEval #68 */
	static docstring = "* Given an array of nums representing a branch on a binary tree, find the minimum even value and its index.\n        In the case of a tie, return the smallest index. If there are no even numbers, the answer is [].\n\n        Sample Input:\n        [1, 7, 4, 6, 10, 11, 14]\n\n        Sample Output:\n        [4, 2]";
	static sat (
		val_index,
		nums = [125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152],
	) {
		/**
		 * Given an array of nums representing a branch on a binary tree, find the minimum even value and its index.
		 * In the case of a tie, return the smallest index. If there are no even numbers, the answer is [].
		 *
		 * Sample Input:
		 * [1, 7, 4, 6, 10, 11, 14]
		 *
		 * Sample Output:
		 * [4, 2]
		 */
		if (val_index.length === 0) {
			return nums.every((n) => n % 2 === 1);
		}
		const [v, i] = val_index;
		if (v % 2 !== 0 || nums[i] !== v) return false;
		for (let k = 0; k < i; k++) {
			if (nums[k] <= v && nums[k] % 2 === 0) return false;
		}
		for (let k = i; k < nums.length; k++) {
			if (nums[k] < v && nums[k] % 2 === 0) return false;
		}
		return true;
	}

	static sol (nums) {
		const evens = nums
			.map((v, i) => (v % 2 === 0 ? [v, i] : null))
			.filter((x) => x !== null);
		if (evens.length === 0) return [];
		return evens.reduce((min, cur) => (cur[0] < min[0] ? cur : min));
	}

	getExample () {
		return {
			nums: [125123, 422323, 141, 5325, 812152, 9, 42145, 5313, 421, 812152],
		};
	}

	genRandom () {
		const digits = this.random.randrange(10);
		const nums = Array.from({ length: this.random.randrange(5) }, () =>
			this.random.randrange(10 ** digits),
		);
		this.add({ nums });
	}
}

export class GreatestHIndex extends PuzzleGenerator {
	/** Inspired by HumanEval #69 */
	static docstring = "* Find the h-index, the largest positive number h such that that h occurs in the sequence at least h times.\n        h = -1 if there is no such positive number.\n\n        Sample Input:\n        [1, 2, 2, 3, 3, 3, 4, 4]\n\n        Sample Output:\n        3";
	static sat (h, seq = [3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5]) {
		/**
		 * Find the h-index, the largest positive number h such that that h occurs in the sequence at least h times.
		 * h = -1 if there is no such positive number.
		 *
		 * Sample Input:
		 * [1, 2, 2, 3, 3, 3, 4, 4]
		 *
		 * Sample Output:
		 * 3
		 */
		for (const i of seq) {
			if (i > 0 && i > h) {
				const count = seq.filter((x) => x === i).length;
				if (count >= i) return false;
			}
		}
		return h === -1 || (seq.filter((x) => x === h).length >= h && h > 0);
	}

	static sol (seq) {
		const candidates = [-1];
		for (const i of seq) {
			if (i > 0) {
				const count = seq.filter((x) => x === i).length;
				if (count >= i) candidates.push(i);
			}
		}
		return Math.max(...candidates);
	}

	getExample () {
		return { seq: [3, 1, 4, 17, 5, 17, 2, 1, 41, 32, 2, 5, 5, 5, 5] };
	}

	genRandom () {
		const seq = Array.from({ length: this.random.randrange(100) }, () =>
			this.random.randrange(10),
		);
		this.add({ seq });
	}
}

export class WildSort extends PuzzleGenerator {
	/** Inspired by HumanEval #70 */
	static docstring = "* Find the following strange sort of li: the first element is the smallest, the second is the largest of the\n        remaining, the third is the smallest of the remaining, the fourth is the smallest of the remaining, etc.\n\n        Sample Input:\n        [1, 2, 7, 3, 4, 5, 6]\n\n        Sample Output:\n        [1, 7, 2, 6, 3, 5, 4]";
	static sat (strange, li = [30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15]) {
		/**
		 * Find the following strange sort of li: the first element is the smallest, the second is the largest of the
		 * remaining, the third is the smallest of the remaining, the fourth is the smallest of the remaining, etc.
		 *
		 * Sample Input:
		 * [1, 2, 7, 3, 4, 5, 6]
		 *
		 * Sample Output:
		 * [1, 7, 2, 6, 3, 5, 4]
		 */
		if (
			JSON.stringify([...strange].sort((a, b) => a - b)) !==
			JSON.stringify([...li].sort((a, b) => a - b))
		) {
			return false;
		}
		for (let i = 0; i < strange.length; i++) {
			const n = strange[i];
			const subarray = strange.slice(i);
			if (i % 2 === 0) {
				if (n !== Math.min(...subarray)) return false;
			} else {
				if (n !== Math.max(...subarray)) return false;
			}
		}
		return true;
	}

	static sol (li) {
		const s = [...li].sort((a, b) => a - b);
		let i = 0;
		let j = s.length - 1;
		const ans = [];
		while (i <= j) {
			if (ans.length % 2 === 1) {
				ans.push(s[j]);
				j--;
			} else {
				ans.push(s[i]);
				i++;
			}
		}
		return ans;
	}

	getExample () {
		return { li: [30, 12, 42, 717, 45, 317, 200, -1, 491, 32, 15] };
	}

	genRandom () {
		const li = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.randrange(10),
		);
		this.add({ li });
	}
}

export class HeronTriangle extends PuzzleGenerator {
	/**
	 * Inspired by HumanEval #71
	 * That problem essentially asks for Heron's formula for the area of a triangle in terms of its three sides.
	 * In our version, we consider the related problem (also solved by Heron's formula) of finding 2d coordinates
	 * of a triangle with the given sides. If one knows the area, this is a straightforward calculation.
	 */
	static docstring = "* Find the coordinates of a triangle with the given side lengths\n\n        Sample Input:\n        [3.0, 4.0, 5.0\n\n        Sample Output:\n        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]";
	static sat (coords, sides = [8.9, 10.8, 17.0]) {
		/**
		 * Find the coordinates of a triangle with the given side lengths
		 *
		 * Sample Input:
		 * [3.0, 4.0, 5.0]
		 *
		 * Sample Output:
		 * [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]
		 */
		if (coords.length !== 3) return false;
		const sides2 = [];
		for (let i = 0; i < coords.length; i++) {
			const [x, y] = coords[i];
			for (let j = 0; j < i; j++) {
				const [x2, y2] = coords[j];
				const dist = Math.sqrt((x - x2) ** 2 + (y - y2) ** 2);
				sides2.push(dist);
			}
		}
		const sortedSides = [...sides].sort((a, b) => a - b);
		const sortedSides2 = sides2.sort((a, b) => a - b);
		return sortedSides.every((a, i) => Math.abs(a - sortedSides2[i]) < 1e-6);
	}

	static sol (sides) {
		const sorted = [...sides].sort((a, b) => a - b);
		const [a, b, c] = sorted;
		const s = (a + b + c) / 2;
		const area = Math.sqrt(s * (s - a) * (s - b) * (s - c));
		const y = (2 * area) / a;
		const x = Math.sqrt(c ** 2 - y ** 2);
		return [
			[0.0, 0.0],
			[a, 0.0],
			[x, y],
		];
	}

	getExample () {
		return { sides: [8.9, 10.8, 17.0] };
	}

	genRandom () {
		const sides = Array.from(
			{ length: 3 },
			() => this.random.random() * 100,
		).sort((a, b) => a - b);
		if (sides[0] + sides[1] > sides[2]) {
			this.add({ sides });
		}
	}
}

export class InvestigateCrash extends PuzzleGenerator {
	/** Inspired by HumanEval #72 */
	static docstring = "* An object will \"fly\" if its weights are a palindrome and sum to <= max_weight. The given object won't fly.\n        You have to determine why. Find index where the weights aren't a palindrome or -1 if weights are too big.\n\n        weights=[77, 40], max_weight=100 => -1\n\n        weights=[1,2,3], max_weight=50   => 0 # because 1 != 3";
	static sat (problem, weights = [1, 2, 5, 2, 1, 17], max_weight = 100) {
		/**
		 * An object will "fly" if its weights are a palindrome and sum to <= max_weight. The given object won't fly.
		 * You have to determine why. Find index where the weights aren't a palindrome or -1 if weights are too big.
		 *
		 * weights=[77, 40], max_weight=100 => -1
		 * weights=[1,2,3], max_weight=50   => 0 # because 1 != 3
		 */
		if (problem === -1) {
			return weights.reduce((a, b) => a + b, 0) > max_weight;
		}
		return weights[problem] !== weights[weights.length - 1 - problem];
	}

	static sol (weights, max_weight) {
		if (weights.reduce((a, b) => a + b, 0) > max_weight) {
			return -1;
		}
		for (let i = 0; i < weights.length; i++) {
			if (weights[i] !== weights[weights.length - 1 - i]) {
				return i;
			}
		}
	}

	getExample () {
		return { weights: [1, 2, 5, 2, 1, 17], max_weight: 100 };
	}

	genRandom () {
		const weights = Array.from({ length: this.random.randrange(1, 10) }, () =>
			this.random.randrange(100),
		);
		const shouldReverse = this.random.choice([true, false]);
		const toAppend = shouldReverse ? [...weights] : weights.slice(0, -1);
		weights.push(...toAppend.reverse());
		if (this.random.random() < 0.8) {
			weights[this.random.randrange(weights.length)] =
				this.random.randrange(100);
		}
		const max_weight =
			weights.reduce((a, b) => a + b, 0) + this.random.randrange(-10, 100);
		const sum = weights.reduce((a, b) => a + b, 0);
		const isReversed =
			JSON.stringify(weights) === JSON.stringify([...weights].reverse());
		if (sum > max_weight || !isReversed) {
			this.add({ weights, max_weight });
		}
	}
}

export class ClosestPalindrome extends PuzzleGenerator {
	/** Inspired by HumanEval #73 */
	static docstring = "* Find the closest palindrome\n\n        Sample Input:\n        \"cat\"\n\n        Sample Output:\n        \"tat\"";
	static sat (pal, s = "palindromordinals") {
		/**
		 * Find the closest palindrome
		 *
		 * Sample Input:
		 * "cat"
		 *
		 * Sample Output:
		 * "tat"
		 */
		if (pal !== pal.split("").reverse().join("")) return false;
		if (pal.length !== s.length) return false;
		const palDiff = Array.from(pal).filter((a, i) => a !== s[i]).length;
		const sDiff = Array.from(s).filter(
			(a, i) => a !== s[s.length - 1 - i],
		).length;
		return palDiff === Math.floor(sDiff / 2);
	}

	static sol (s) {
		const n = s.length;
		return (
			s.slice(0, Math.ceil(n / 2)) +
			s
				.slice(0, Math.floor(n / 2))
				.split("")
				.reverse()
				.join("")
		);
	}

	getExample () {
		return { s: "palindromordinals" };
	}

	genRandom () {
		const w = this.random.pseudoWord();
		const n = w.length;
		const half = w.slice(0, Math.ceil(n / 2));
		const rest = Array.from(w.slice(0, Math.floor(n / 2)))
			.map((c) => this.random.choice([c, this.random.char()]))
			.join("");
		const s = half + rest;
		this.add({ s });
	}
}

export class NarrowerList extends PuzzleGenerator {
	/** Inspired by HumanEval #74 */
	static docstring = "* Find the list that has fewer total characters (including repetitions)\n\n        Sample Input:\n        [[\"sh\", \"ort\"], [\"longest\"]]\n\n        Sample Output:\n        [[\"sh\", \"ort\"]";
	static sat (
		li,
		lists = [
			["this", "list", "is", "narrow"],
			["I", "am", "shorter but wider"],
		],
	) {
		/**
		 * Find the list that has fewer total characters (including repetitions)
		 *
		 * Sample Input:
		 * [["sh", "ort"], ["longest"]]
		 *
		 * Sample Output:
		 * [["sh", "ort"]]
		 */
		const width = li.reduce((sum, s) => sum + s.length, 0);
		for (const li2 of lists) {
			const width2 = li2.reduce((sum, s) => sum + s.length, 0);
			if (width > width2) return false;
		}
		return lists.some((l) => JSON.stringify(l) === JSON.stringify(li));
	}

	static sol (lists) {
		return lists.reduce((min, cur) => {
			const minWidth = min.reduce((sum, s) => sum + s.length, 0);
			const curWidth = cur.reduce((sum, s) => sum + s.length, 0);
			return curWidth < minWidth ? cur : min;
		});
	}

	getExample () {
		return {
			lists: [
				["this", "list", "is", "narrow"],
				["I", "am", "shorter but wider"],
			],
		};
	}

	genRandom () {
		const num_lists = this.random.randrange(1, 5);
		const lists = Array.from({ length: num_lists }, () =>
			Array.from({ length: this.random.randrange(2, 5) }, () =>
				this.random.pseudoWord(),
			),
		);
		this.add({ lists });
	}
}

export class ThreePrimes extends PuzzleGenerator {
	/** Inspired by HumanEval #75 */
	static docstring = "* Find all 247 integers <= 1000 that are the product of exactly three primes.\n        Each integer should represented as the list of its three prime factors.\n        [[2, 2, 2], [2, 2, 3],  [2, 2, 5], ...";
	static sat (factors) {
		/**
		 * Find all 247 integers <= 1000 that are the product of exactly three primes.
		 * Each integer should represented as the list of its three prime factors.
		 * [[2, 2, 2], [2, 2, 3],  [2, 2, 5], ...
		 */
		const primes = new Set(Array.from({ length: 998 }, (_, i) => i + 2));
		for (let n = 2; n < 1000; n++) {
			if (primes.has(n)) {
				for (let k = 2 * n; k < 1000; k += n) {
					primes.delete(k);
				}
			}
		}
		for (const f of factors) {
			for (const p of f) {
				if (!primes.has(p)) return false;
			}
		}
		const nums = new Set(factors.map((f) => f[0] * f[1] * f[2]));
		return Math.max(...nums) < 1000 && nums.size === 247;
	}

	static sol () {
		const primes = new Set(Array.from({ length: 998 }, (_, i) => i + 2));
		for (let n = 2; n < 1000; n++) {
			if (primes.has(n)) {
				for (let k = 2 * n; k < 1000; k += n) {
					primes.delete(k);
				}
			}
		}
		const primesArr = Array.from(primes);
		const result = [];
		for (let i = 0; i < primesArr.length; i++) {
			const p = primesArr[i];
			for (let j = i; j < primesArr.length; j++) {
				const q = primesArr[j];
				for (let k = j; k < primesArr.length; k++) {
					const r = primesArr[k];
					if (p * q * r < 1000) {
						result.push([p, q, r]);
					}
				}
			}
		}
		return result;
	}

	getExample () {
		return {};
	}

	gen (targetNumInstances) {
		this.add({});
	}
}

// COMMENT OUT: stuck
// export class IntegerLog extends PuzzleGenerator {
// 	//
// 	// Inspired by HumanEval #76
// 	//
// 	static docstring = "* Find an integer exponent x such that a^x = n\n        Sample Input:\n        a=2, n=1024\n\n        Sample Output:\n        x = 10";
// 	static sat (x, a = 3n, n = 1290070078170102666248196035845070394933441741644993085810116441344597492642263849n) {
// 		// Ensure all inputs are BigInts for exact comparison
// 		const bigX = BigInt(x);
// 		const bigA = BigInt(a);
// 		const bigN = BigInt(n);

// 		// a^x == n
// 		return bigA ** bigX === bigN;
// 	}

// 	static sol (a, n) {
// 		let bigA = BigInt(a);
// 		let bigN = BigInt(n);

// 		let m = 1n;
// 		let x = 0;

// 		while (m !== bigN) {
// 			x++;
// 			m *= bigA;

// 			// Safety break to prevent infinite loops if n is not a power of a
// 			if (m > bigN && bigA > 1n) {
// 				throw new Error("n is not a power of a");
// 			}
// 		}
// 		return x;
// 	}

// 	genRandom () {
// 		// Random base a between 1 and 9
// 		const a = BigInt(Math.floor(this.random() * 9) + 1);
// 		// Random exponent between 0 and 254
// 		const exponent = BigInt(Math.floor(this.random() * 255));

// 		const n = a ** exponent;
// 		this.add({ a: a, n: n });
// 	}
// }

export class CubeRoot extends PuzzleGenerator {
	/**
	 * Inspired by HumanEval #77
	 * We made it harder by giving very large n for which `round(n ** (1/3))`
	 */
	static docstring = "* Find an integer that when cubed is n\n\n        Sample Input:\n        21\n\n        Sample Output:\n        3";
	static sat (x, n) {
		/**
		 * Find an integer that when cubed is n
		 *
		 * Sample Input:
		 * 21
		 *
		 * Sample Output:
		 * 3
		 */
		if (n === undefined) {
			n = 42714774173606970182754018064350848294149432972747296768n;
		}
		let result = 1n;
		for (let i = 0; i < 3; i++) {
			result *= BigInt(x);
		}
		return result === BigInt(n);
	}

	static sol (n) {
		const m = BigInt(n) < 0n ? -BigInt(n) : BigInt(n);
		let x = BigInt(Math.round(Number(m) ** (1 / 3)));
		while (x * x * x !== m) {
			const next = x + (m - x * x * x) / (3n * x * x);
			if (next === x) break;
			x = next;
		}
		return BigInt(n) < 0n ? -x : x;
	}

	getExample () {
		return { n: "42714774173606970182754018064350848294149432972747296768" };
	}

	genRandom () {
		const base = this.random.randrange(-10, 10);
		const exp = this.random.randrange(1, 30);
		let n = 1n;
		for (let i = 0; i < exp; i++) {
			n *= BigInt(base);
		}
		this.add({ n: n.toString() });
	}
}

export class HexPrimes extends PuzzleGenerator {
	/** Inspired by HumanEval #78 */
	static docstring = "* Determine which characters of a hexidecimal correspond to prime numbers\n\n        Sample Input:\n        \"123ABCD\"\n\n        Sample Output:\n        [False, True, True, False, True, False True]";
	static sat (primes, n = "A4D4455214122CE192CCBE3") {
		/**
		 * Determine which characters of a hexidecimal correspond to prime numbers
		 *
		 * Sample Input:
		 * "123ABCD"
		 *
		 * Sample Output:
		 * [false, true, true, false, true, false, true]
		 */
		return primes.every((p, i) => p === "2357BD".includes(n[i]));
	}

	static sol (n) {
		return Array.from(n).map((c) => "2357BD".includes(c));
	}

	getExample () {
		return { n: "A4D4455214122CE192CCBE3" };
	}

	genRandom () {
		const digits = this.random.randrange(30);
		const n = Math.floor(Math.random() * 10 ** digits).toString(16);
		this.add({ n });
	}
}

export class Binarize extends PuzzleGenerator {
	/** Inspired by HumanEval #79 */
	static docstring = "* Write n base 2 followed and preceded by 'bits'\n        Sample Input:\n        2\n\n        Sample Output:\n        bits10bits";
	static sat (b, n = 5324680297138495285n) {
		/**
		 * Write n base 2 followed and preceded by 'bits'
		 * Sample Input:
		 * 2
		 *
		 * Sample Output:
		 * bits10bits
		 */
		const targetN = BigInt(n);

		// Check prefix and suffix
		if (!b.startsWith("bits") || !b.endsWith("bits")) return false;
		if (b.length < 8) return false;

		const inside = b.slice(4, -4);

		// Check binary characters
		if (!/^[01]+$/.test(inside)) return false;

		// Check for leading zeros (unless the string is just "0")
		if (inside.length > 1 && inside[0] === "0") return false;

		// Convert binary string to BigInt
		let m = 0n;
		for (let c of inside) {
			m = 2n * m + BigInt(c);
		}

		return m === targetN;
	}

	static sol (n) {
		const s = BigInt(n).toString(2);
		return `bits${s}bits`;
	}

	getExample () {
		return { n: 5324680297138495285n };
	}

	genRandom () {
		const digits = Math.floor(this.random() * 30);
		// Generating large random numbers in JS
		const max = 10n ** BigInt(digits);
		const n = BigInt(Math.floor(this.random() * Number(max)));
		this.add({ n: n });
	}
}

export class NearbyDuplicates extends PuzzleGenerator {
	/** Inspired by HumanEval #80 */
	static docstring = "* A string is happy if every three consecutive characters are distinct. Find two indices making s unhappy.\n        Sample Input:\n        \"street\"\n\n        Sample Output:\n        [3, 4]";
	static sat (indices, s = "I am an unhappy string!") {
		/**
		 * A string is happy if every three consecutive characters are distinct. Find two indices making s unhappy.
		 * Sample Input:
		 * "street"
		 *
		 * Sample Output:
		 * [3, 4]
		 */
		const [i, j] = indices;
		return s[i] === s[j] && 0 <= i && i < j && j < i + 3;
	}

	static sol (s) {
		for (let i = 0; i < s.length - 2; i++) {
			if (s[i] === s[i + 1]) return [i, i + 1];
			if (s[i] === s[i + 2]) return [i, i + 2];
		}
	}

	getExample () {
		return { s: "I am an unhappy string!" };
	}

	genRandom () {
		const a = this.random.string({ minLen: 1 });
		const sep = this.random.choice(["", this.random.char()]);
		const s = a + sep + a[a.length - 1] + this.random.string({ minLen: 1 });
		this.add({ s });
	}
}

export class Grader extends PuzzleGenerator {
	static docstring = "* Convert GPAs to letter grades according to the following table:\n        4.0: A+\n        3.7: A\n        3.4: A-\n        3.0: B+\n        2.7: B\n        2.4: B-\n        2.0: C+\n        1.7: C\n        1.4: C-\n        below: F\n\n        Sample input: [4.0, 3.5, 3.8]\n        Sample output: ['A+', 'A-', 'A']";
	/** Inspired by HumanEval #81 */
	static sat (grades, gpas = [2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9]) {
		if (grades.length !== gpas.length) return false;
		const letters = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "F"];
		const scores = [4.0, 3.7, 3.4, 3.0, 2.7, 2.4, 2.0, 1.7, 1.4, 0.0];
		for (let i = 0; i < grades.length; i++) {
			const grade = grades[i];
			const gpa = gpas[i];
			const idx = letters.indexOf(grade);
			if (idx === -1) return false;
			if (gpa < scores[idx]) return false;
			if (idx > 0 && gpa > scores[idx - 1]) return false;
		}
		return true;
	}

	static sol (gpas) {
		const letters = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "F"];
		const scores = [4.0, 3.7, 3.4, 3.0, 2.7, 2.4, 2.0, 1.7, 1.4, 0.0];
		return gpas.map((gpa) => {
			let idx = 0;
			while (gpa < scores[idx]) idx++;
			return letters[idx];
		});
	}

	getExample () {
		return { gpas: [2.8, 3.1, 4.0, 2.2, 3.1, 2.5, 0.9] };
	}

	genRandom () {
		const length = this.random.randrange(10);
		const gpas = Array.from({ length }, () => this.random.random() * 4.0);
		this.add({ gpas });
	}
}

export class FactorString extends PuzzleGenerator {
	static docstring = "* Find a string which when repeated more than once gives s\n        Sample Input:\n        \"haha\"\n\n        Sample Output:\n        \"ha\"";
	/** Inspired by HumanEval #82 */
	static sat (factor, s = "catscatcatscatcatscat") {
		if (!factor.length || factor.length >= s.length) return false;
		const repeat = Math.floor(s.length / factor.length);
		return repeat > 1 && factor.repeat(repeat) === s;
	}

	static sol (s) {
		for (let i = 1; i < s.length; i++) {
			if (s.length % i !== 0) continue;
			const prefix = s.slice(0, i);
			if (prefix.repeat(s.length / i) === s) return prefix;
		}
	}

	getExample () {
		return { s: "catscatcatscatcatscat" };
	}

	genRandom () {
		const piece = this.random.pseudo_word();
		const repeat = this.random.randrange(2, 10);
		this.add({ s: piece.repeat(repeat) });
	}
}

export class OneEnded extends PuzzleGenerator {
	static docstring = "* Find all n-digit integers that start or end with 1\n\n        1 => [1]";
	/** Inspired by HumanEval #83 */
	static sat (nums, n = 5) {
		const count = n > 1 ? 18 * 10 ** (n - 2) : 1;
		const strings = new Set(nums.map((num) => num.toString()));
		if (strings.size !== count) return false;
		return nums.every((num) => {
			const str = num.toString();
			return str.length === n && (str.startsWith("1") || str.endsWith("1"));
		});
	}

	static sol (n) {
		const start = n === 1 ? 1 : 10 ** (n - 1);
		const end = 10 ** n;
		const ans = [];
		for (let i = start; i < end; i++) {
			const str = i.toString();
			if (str.length !== n) continue;
			if (str.startsWith("1") || str.endsWith("1")) ans.push(i);
		}
		return ans;
	}

	getExample () {
		return { n: 2 };
	}
}

export class BitSum extends PuzzleGenerator {
	/** Inspired by HumanEval #84 */
	static docstring = "* Find an b-bit integer with a bit-sum of s\n\n        b=3, s=2 => 5 # 5 is 101 in binary";
	static sat (n, b = 107, s = 25) {
		const str = n.toString(2);
		if (str[0] === "-") return false;
		if (str.length !== b) return false;
		const digitSum = [...str].reduce((acc, digit) => acc + Number(digit), 0);
		return digitSum === s;
	}

	static sol (b, s) {
		return parseInt("1".repeat(s) + "0".repeat(b - s), 2);
	}

	getExample () {
		return { b: 107, s: 25 };
	}

	genRandom () {
		const b = this.random.randrange(1, 1000);
		const s = this.random.randrange(1, b + 1);
		this.add({ b, s });
	}
}

export class EvenOddSum extends PuzzleGenerator {
	/** Inspired by HumanEval #85 */
	static docstring = "* Find the sum of the even elements that are at odd indices\n\n        [1, 2, 8, 3, 9, 4] => 6";
	static sat (
		evenOddSum,
		nums = [
			2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350,
		],
	) {
		for (let i = 1; i < nums.length; i += 2) {
			if (nums[i] % 2 === 0) evenOddSum -= nums[i];
		}
		return evenOddSum === 0;
	}

	static sol (nums) {
		return nums.reduce(
			(total, num, idx) =>
				idx % 2 === 1 && num % 2 === 0 ? total + num : total,
			0,
		);
	}

	getExample () {
		return {
			nums: [
				2341, 125146894, 12521, -12451293476325, 535284623934, 132974693614350,
			],
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ nums });
	}
}

export class AntiShuffle extends PuzzleGenerator {
	/** Inspired by HumanEval #86 */
	static docstring = "* Create a new string by taking s, and word by word rearranging its characters in ascii order\n        Sample input:\n        'maltos wow'\n\n        Sample output:\n        'almost oww'";
	static sat (shuffled, orig = "Hello world!!!") {
		const words = shuffled.split(" ");
		const source = orig.split(" ");
		if (words.length !== source.length) return false;
		for (let i = 0; i < words.length; i++) {
			const a = [...words[i]].sort().join("");
			const b = [...source[i]].sort().join("");
			if (a !== b) return false;
		}
		return shuffled.length === orig.length;
	}

	static sol (orig) {
		return orig
			.split(" ")
			.map((word) => [...word].sort().join(""))
			.join(" ");
	}

	getExample () {
		return { orig: "Hello world!!!" };
	}

	genRandom () {
		const orig = Array.from({ length: this.random.randrange(1, 6) }, () =>
			this.random.pseudo_word(),
		).join(" ");
		this.add({ orig });
	}
}

export class UnevenFind extends PuzzleGenerator {
	/** Inspired by HumanEval #87 */
	static docstring = "* Find the indices of all occurrences of target in the uneven matrix\n        Sample input:\n        uneven=[[2, 3, 2], [], [9, 2]], target=2\n\n        Sample output:\n        [[0, 0], [0, 2], [2, 1]]";
	static sat (
		indices,
		uneven = [[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]],
		target = 17,
	) {
		for (const [i, j] of indices) {
			if (uneven[i][j] !== target) return false;
		}
		for (let i = 0; i < uneven.length; i++) {
			for (let j = 0; j < uneven[i].length; j++) {
				if (uneven[i][j] === target) {
					if (!indices.some(([row, col]) => row === i && col === j))
						return false;
				}
			}
		}
		return true;
	}

	static sol (uneven, target) {
		const ans = [];
		for (let i = 0; i < uneven.length; i++) {
			for (let j = 0; j < uneven[i].length; j++) {
				if (uneven[i][j] === target) ans.push([i, j]);
			}
		}
		return ans;
	}

	getExample () {
		return {
			uneven: [[1, 3, 2, 32, 17], [17, 2, 48, 17], [], [9, 35, 4], [3, 17]],
			target: 17,
		};
	}

	genRandom () {
		const target = this.random.randrange(100);
		const uneven = Array.from({ length: this.random.randrange(10) }, () =>
			Array.from({ length: this.random.randrange(10) }, () =>
				this.random.choice([target, this.random.randrange(100)]),
			),
		);
		this.add({ uneven, target });
	}
}

export class UpDownSort extends PuzzleGenerator {
	/** Inspired by HumanEval #88 */
	static docstring = "* Reorder nums in increasing/decreasing order based on whether the first plus last element is even/odd\n\n        Sample input:\n        [1, 7, 4]\n\n        Sample output:\n        [1, 4, 7] # because 1 + 4 is odd\n\n        Sample input:\n        [1, 7, 5]\n\n        Sample output:\n        [8, 5, 1] # because 1 + 5 is even";
	static sat (upDown, nums = [17, 2, 3, 523, 18, -2, 0, 2, -1]) {
		const counts = new Map();
		for (const num of nums) counts.set(num, (counts.get(num) ?? 0) + 1);
		for (const num of upDown) counts.set(num, (counts.get(num) ?? 0) - 1);
		if ([...counts.values()].some((value) => value !== 0)) return false;
		const increasingSign = (nums[0] + nums[nums.length - 1]) % 2 === 1 ? 1 : -1;
		for (let i = 0; i < upDown.length - 1; i++) {
			if ((upDown[i + 1] - upDown[i]) * increasingSign < 0) return false;
		}
		return true;
	}

	static sol (nums) {
		const increasing = (nums[0] + nums[nums.length - 1]) % 2 === 1;
		return [...nums].sort((a, b) => (increasing ? a - b : b - a));
	}

	getExample () {
		return { nums: [17, 2, 3, 523, 18, -2, 0, 2, -1] };
	}
}

export class SubstitutionCypher extends PuzzleGenerator {
	/** Inspired by HumanEval #89 */
	static docstring = "* Apply a substitution cypher in which each character is advanced by two multiplied by two places.\n\n        'substitution cypher' => 'wyfwxmxyxmsr$g}tliv'";
	static sat (encrypted, orig = "Hello, world!") {
		if (encrypted.length !== orig.length) return false;
		for (let i = 0; i < orig.length; i++) {
			if (encrypted.charCodeAt(i) - 4 !== orig.charCodeAt(i)) return false;
		}
		return true;
	}

	static sol (orig) {
		return orig
			.split("")
			.map((char) => String.fromCharCode(char.charCodeAt(0) + 4))
			.join("");
	}

	getExample () {
		return { orig: "Hello, world!" };
	}

	genRandom () {
		const orig = Array.from({ length: this.random.randrange(1, 6) }, () =>
			this.random.pseudo_word(),
		).join(" ");
		this.add({ orig });
	}
}

export class SecondSmallestUnique extends PuzzleGenerator {
	/** Inspired by HumanEval #90 */
	static docstring = "* Find the second smallest unique number in the list nums.\n\n        Sample input:\n        [2, 5, 2, 7, 9]\n\n        Sample output:\n        5";
	static sat (
		n,
		nums = [
			17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589,
		],
	) {
		if (!nums.includes(n)) return false;
		const unique = [...new Set(nums.filter((x) => x <= n))];
		unique.sort((a, b) => a - b);
		return unique.length === 2 && unique[1] === n;
	}

	static sol (nums) {
		const unique = [...new Set(nums)].sort((a, b) => a - b);
		return unique[1];
	}

	getExample () {
		return {
			nums: [
				17, -1023589211, -293485382500, 31, -293485382500, 105762, 94328103589,
			],
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(2, 10) }, () =>
			this.random.randrange(-10, 10),
		);
		if (new Set(nums).size >= 2) this.add({ nums });
	}
}

export class FindBored extends PuzzleGenerator {
	/** Inspired by HumanEval #91 */
	static docstring = "* A bored sentence starts with the word \"I\". Find all bored sentences in s. Sentence delimiters are '.!?'\n\n        --- Example input ---\n        'I wrote this. You read it? I think I am so cool. In another time, I would be lame.'\n\n        --- Example output ---\n        ['I wrote this', ' I think I am so cool']\n\n        ";
	static sat (
		boring,
		text = "This is not boring. I am boring! I am sooo tired.",
	) {
		const sentences = text.replace(/[!?]/g, ".").split(".");
		const boringAndExciting = boring.concat(
			sentences.filter((s) => {
				const firstWord = s.trim().split(" ")[0];
				return firstWord !== "I";
			}),
		);
		const sorted = (arr) => [...arr].sort();
		return (
			JSON.stringify(sorted(boringAndExciting)) ===
			JSON.stringify(sorted(sentences))
		);
	}

	static sol (text) {
		return text
			.replace(/[!?]/g, ".")
			.split(".")
			.filter((s) => s.trim().split(" ")[0] === "I");
	}

	getExample () {
		return { text: "This is not boring. I am boring! I am sooo tired." };
	}

	genRandom () {
		let text = "";
		while (this.random.random() < 0.75) {
			const length = this.random.randrange(6);
			const prefix = this.random.choice([[], ["I"]]);
			const words = prefix.concat(
				Array.from({ length }, () => this.random.pseudo_word()),
			);
			text += words.join(" ") + this.random.choice([".", "!", "?"]);
		}
		this.add({ text });
	}
}

export class IdentifyZeroTrips extends PuzzleGenerator {
	/** Inspired by HumanEval #92 */
	static docstring = "* Determine which triples sum to zero\n\n        --- Example input ---\n        [1, 2, 4, -3, 5]\n\n        --- Example output ---\n        [0, 1, 3]";
	static sat (
		zeroSums,
		trips = [
			[1253532, -3920635, 332],
			[-24, 18, 6],
			[0, 5, -5],
			[1, 1, 1],
			[-20, 17, 4],
		],
	) {
		if (zeroSums.length !== trips.length) return false;
		return zeroSums.every((z, idx) => {
			const sum = trips[idx].reduce((total, n) => total + n, 0);
			return z === (sum === 0);
		});
	}

	static sol (trips) {
		return trips.map((trip) => trip.reduce((sum, n) => sum + n, 0) === 0);
	}

	getExample () {
		return {
			trips: [
				[1253532, -3920635, 332],
				[-24, 18, 6],
				[0, 5, -5],
				[1, 1, 1],
				[-20, 17, 4],
			],
		};
	}

	genRandom () {
		const trips = Array.from({ length: this.random.randrange(2, 20) }, () =>
			Array.from({ length: 3 }, () => {
				const value = this.random.randrange(-10, 11);
				return value;
			}),
		);
		for (const trip of trips) {
			if (this.random.randrange(2)) {
				trip[2] = trip[0] + trip[1];
			}
		}
		this.add({ trips });
	}
}

export class WeirdDecodeVowels extends PuzzleGenerator {
	/** Inspired by HumanEval #93 */
	static docstring = "* Find string s that, when case is flipped gives target where vowels are replaced by chars two later.\n        --- Example input ---\n        'THIS is a TEST'\n\n        --- Example output ---\n        'thks KS C tgst'";
	static sat (s, target = "Hello, world!") {
		const shiftVowel = (char) => String.fromCharCode(char.charCodeAt(0) + 2);
		const translate = target.replace(/[aeiouAEIOU]/g, shiftVowel);
		const swapCase = (str) =>
			str
				.split("")
				.map((char) =>
					char === char.toLowerCase() ? char.toUpperCase() : char.toLowerCase(),
				)
				.join("");
		return swapCase(s) === translate;
	}

	static sol (target) {
		const shiftVowel = (char) => String.fromCharCode(char.charCodeAt(0) + 2);
		const swapCase = (str) =>
			str
				.split("")
				.map((char) =>
					char === char.toLowerCase() ? char.toUpperCase() : char.toLowerCase(),
				)
				.join("");
		return swapCase(target.replace(/[aeiouAEIOU]/g, shiftVowel));
	}

	getExample () {
		return { target: "Hello, world!" };
	}

	genRandom () {
		const target = Array.from({ length: this.random.randrange(1, 4) }, () =>
			this.random.pseudo_word(),
		).join(" ");
		this.add({ target });
	}
}

export class LargestPrimeDigitSum extends PuzzleGenerator {
	/** Inspired by HumanEval #94 */
	static docstring = "* Find the index of the largest prime in the list and the sum of its digits\n\n        --- Example input ---\n        [2, 4, 7, 19, 21]\n\n        --- Example output ---\n        [3, 10]";
	static sat (
		ans,
		nums = [23, 17, 201, 14, 10473, 43225, 421, 423, 11, 10, 2022, 342157],
	) {
		const [i, digitSum] = ans;
		const value = nums[i];
		const isPrime = (n) => {
			if (n <= 1) return false;
			for (let j = 2; j <= Math.floor(Math.sqrt(n)); j++) {
				if (n % j === 0) return false;
			}
			return true;
		};
		if (!isPrime(value)) return false;
		if (nums.some((n, idx) => idx !== i && isPrime(n) && n > value))
			return false;
		const expectedSum = [...value.toString()].reduce(
			(sum, chr) => sum + Number(chr),
			0,
		);
		return digitSum === expectedSum;
	}

	static sol (nums) {
		const isPrime = (n) => {
			if (n <= 1) return false;
			for (let i = 2; i <= Math.floor(Math.sqrt(n)); i++) {
				if (n % i === 0) return false;
			}
			return true;
		};
		let bestIndex = 0;
		let bestValue = -Infinity;
		for (let i = 0; i < nums.length; i++) {
			if (isPrime(nums[i]) && nums[i] > bestValue) {
				bestIndex = i;
				bestValue = nums[i];
			}
		}
		if (bestValue === -Infinity) return [0, 0];
		const digitSum = [...bestValue.toString()].reduce(
			(sum, chr) => sum + Number(chr),
			0,
		);
		return [bestIndex, digitSum];
	}

	getExample () {
		return {
			nums: [23, 17, 201, 14, 10473, 43225, 421, 423, 11, 10, 2022, 342157],
		};
	}

	genRandom () {
		const nums = Array.from({ length: 10 }, () =>
			this.random.randrange(1, 10 ** this.random.randrange(1, 8)),
		);
		const isPrime = (n) => {
			if (n <= 1) return false;
			for (let i = 2; i <= Math.floor(Math.sqrt(n)); i++) {
				if (n % i === 0) return false;
			}
			return true;
		};
		if (nums.some((n) => isPrime(n))) {
			this.add({ nums });
		}
	}
}

export class OddCase extends PuzzleGenerator {
	/** Inspired by HumanEval #95 */
	static docstring = "* Find the dictionary key whose case is different than all other keys\n\n        --- Example input ---\n        {\"red\": \"\", \"GREEN\": \"\", \"blue\": \"orange\"}\n\n        --- Example output ---\n        \"GREEN\"";
	static sat (
		different,
		d = {
			cat: "CAT",
			tree: "T",
			"pick me": "not",
			OK: "red",
			blah: "blah",
			z: "Z",
		},
	) {
		if (!(different in d)) return false;
		const isDifferentLower = different === different.toLowerCase();
		for (const key of Object.keys(d)) {
			if (key === different) continue;
			if ((key === key.toLowerCase()) === isDifferentLower) return false;
		}
		return true;
	}

	static sol (d) {
		const keys = Object.keys(d);
		for (const key of keys) {
			const isLower = key === key.toLowerCase();
			if (
				keys.every(
					(other) =>
						other === key || (other === other.toLowerCase()) !== isLower,
				)
			) {
				return key;
			}
		}
	}

	getExample () {
		return {
			d: {
				cat: "CAT",
				tree: "T",
				"pick me": "not",
				OK: "red",
				blah: "blah",
				z: "Z",
			},
		};
	}

	genRandom () {
		let mostlyUpper = this.random.choice([true, false]);
		const transform = (word) =>
			mostlyUpper ? word.toUpperCase() : word.toLowerCase();
		const d = {};
		for (let i = 0; i < 10; i++) {
			d[transform(this.random.pseudo_word())] = this.random.pseudo_word();
		}
		mostlyUpper = !mostlyUpper;
		d[transform(this.random.pseudo_word())] = this.random.pseudo_word();
		this.add({ d });
	}
}

export class PrimesUpTo extends PuzzleGenerator {
	/** Inspired by HumanEval #96 */
	static docstring = "* Find all primes up to n\n\n        --- Example input ---\n        9\n\n        --- Example output ---\n        [2, 3, 5, 7]";
	static sat (primes, n = 1234) {
		const isPrime = (num) => {
			if (num <= 1) return false;
			for (let i = 2; i <= Math.floor(Math.sqrt(num)); i++) {
				if (num % i === 0) return false;
			}
			return true;
		};
		if (
			!primes.every(
				(p, idx) => p > 1 && primes.slice(0, idx).every((q) => p % q !== 0),
			)
		)
			return false;
		const covered = new Set();
		for (const prime of primes) {
			for (let j = prime; j < n; j += prime) {
				covered.add(j);
			}
		}
		return covered.size === Math.max(n - 2, 0);
	}

	static sol (n) {
		const primes = [];
		const candidates = new Set();
		for (let i = 2; i < n; i++) candidates.add(i);
		for (let i = 2; i < n; i++) {
			if (candidates.has(i)) {
				primes.push(i);
				for (let j = i; j < n; j += i) {
					candidates.delete(j);
				}
			}
		}
		return primes;
	}

	getExample () {
		return { n: 9 };
	}

	genRandom () {
		const n = this.random.randrange(20 * 1000);
		this.add({ n });
	}
}

export class UnitsProduct extends PuzzleGenerator {
	/** Inspired by HumanEval #97 */
	static docstring = "* Find the product of the units digits in the numbers\n\n        [12, 34] => 8";
	static sat (prod, nums = [17, 24, 39, 15, 11, 201, 97, 65, 18]) {
		if (!nums.every((n) => n !== 0)) return prod === 0;
		for (const n of nums) {
			const k = Math.abs(n % 10);
			if (k === 0) return prod === 0;
			if (prod % k !== 0) return false;
			prod /= k;
		}
		return prod === 1;
	}

	static sol (nums) {
		return nums.reduce((acc, n) => acc * Math.abs(n % 10), 1);
	}

	getExample () {
		return { nums: [17, 24, 39, 15, 11, 201, 97, 65, 18] };
	}

	genRandom () {
		const nums = Array.from({ length: 10 }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ nums });
	}
}

export class UppercaseEven extends PuzzleGenerator {
	/** Inspired by HumanEval #98 */
	static docstring = "* Find the positions of all uppercase vowels (not counting Y) in even indices\n\n        \"EAT here NOW\" => [0, 10]";
	static sat (positions, s = "ThIs is A tEsT, Or *IS* iT?") {
		const vowels = "AEIOU";
		for (const idx of positions) {
			if (!vowels.includes(s[idx])) return false;
		}
		for (let i = 0; i < s.length; i++) {
			if (i % 2 === 0 && vowels.includes(s[i]) && !positions.includes(i))
				return false;
		}
		return true;
	}

	static sol (s) {
		const vowels = "AEIOU";
		return Array.from({ length: s.length }, (_, i) =>
			i % 2 === 0 && vowels.includes(s[i]) ? i : -1,
		).filter((i) => i !== -1);
	}

	getExample () {
		return { s: "ThIs is A tEsT, Or *IS* iT?" };
	}

	genRandom () {
		const base = this.random.pseudo_word();
		const s = Array.from(base)
			.map((char) =>
				this.random.choice([char.toLowerCase(), char.toUpperCase()]),
			)
			.join("");
		this.add({ s });
	}
}

export class ClosestInteger extends PuzzleGenerator {
	/** Inspired by HumanEval #99 */
	static docstring = "* Round to nearest integer\n\n        --- input ---\n        3.7\n\n        --- output ---\n        4";
	static sat (n, x = 329437923.5) {
		return Math.abs(n - x) <= 0.5;
	}

	static sol (x) {
		return Math.round(x);
	}

	getExample () {
		return { x: 329437923.5 };
	}

	genRandom () {
		const magnitude = this.random.randrange(1, 21);
		let x = this.random.random() * 10 ** magnitude;
		if (this.random.random() < 0.5) x = -x;
		if (this.random.randrange(10) === 0) x = Math.floor(x) - 0.5;
		this.add({ x });
	}
}

export class StonePiles extends PuzzleGenerator {
	/** Inspired by HumanEval #100 */
	static docstring = "* We are making n stone piles! The first pile has n stones. If n is even, then all piles have an even\n        number of stones. If n is odd, all piles have an odd number of stones. Each pile must more stones\n        than the previous pile but as few as possible. Return the number of stones in each pile.\n\n        2 => [2, 4]";
	static sat (li, n = 909) {
		if (li.length !== n) return false;
		if (n === 0) return li.length === 0;
		if (!li.length) return false;
		if (li[0] !== n) return false;
		for (let i = 1; i < li.length; i++) {
			if (li[i] - li[i - 1] !== 2) return false;
		}
		return true;
	}

	static sol (n) {
		return Array.from({ length: n }, (_, i) => n + 2 * i);
	}

	getExample () {
		return { n: 909 };
	}

	genRandom () {
		const n = Math.max(1, this.random.randrange(10 ** 5));
		this.add({ n });
	}
}

export class CompleteSplit extends PuzzleGenerator {
	/** Inspired by HumanEval #101 */
	static docstring = "* \n        Split a string of words separated by commas and spaces into 2 lists: words and separators\n\n        Sample input: \"Hi there, Anna\"\n        Sample output: [[\"Hi\", \"there\", \"Anna\"], [\" \", \", \"]]";
	static sat (
		splits,
		string = "Hello, world!  You look like you're on turtles.",
	) {
		const [words, separators] = splits;
		if (words.length !== separators.length + 1) return false;
		const trailing = [...separators, " "];
		for (let i = 0; i < words.length; i++) {
			const word = words[i];
			const sep = trailing[i];
			if (word.includes(" ") || word.includes(",")) return false;
			if (!sep || sep.length === 0) return false;
			if (![...sep].every((c) => c === " " || c === ",")) return false;
		}
		let merged = "";
		for (let i = 0; i < words.length - 1; i++) {
			merged += words[i] + separators[i];
		}
		merged += words[words.length - 1];
		return merged === string;
	}

	static sol (string) {
		const parts = string.split(/([ ,]+)/);
		const words = parts.filter((_, i) => i % 2 === 0);
		const separators = parts.filter((_, i) => i % 2 === 1);
		return [words, separators];
	}

	getExample () {
		return { string: "Hello, world!  You look like you're on turtles." };
	}

	genRandom () {
		const choices = [" ", ",", ".", this.random.pseudo_word()];
		let string = "";
		for (let i = 0; i < this.random.randrange(20); i++) {
			string += this.random.choice(choices);
		}
		this.add({ string });
	}
}

export class BiggestEven extends PuzzleGenerator {
	/** Inspired by HumanEval #102 */
	static docstring = "* Return the biggest even number between a and b inclusive, or -1 if there is no such number\n\n        Example input:\n        a=20, b=99\n\n        Example output:\n        98";
	static sat (x, a = 145, b = 24126846790974) {
		if (x === -1) {
			for (let i = a; i <= b; i++) {
				if (i % 2 === 0) return false;
			}
			return true;
		}
		if (x < a || x > b || x % 2 !== 0) return false;
		for (let i = x + 1; i <= b; i++) {
			if (i % 2 === 0) return false;
		}
		return true;
	}

	static sol (a, b) {
		if (a > b || (a === b && a % 2 === 1)) return -1;
		return b % 2 === 0 ? b : b - 1;
	}

	getExample () {
		return { a: 145, b: 24126846790974 };
	}

	genRandom () {
		const a = this.random.randrange(10 ** this.random.randrange(10));
		const b = this.random.randrange(10 ** this.random.randrange(10));
		this.add({ a, b });
	}
}

export class BinaryAverage extends PuzzleGenerator {
	/** Inspired by HumanEval #103 */
	static docstring = "* Return the average of the numbers a through b rounded to nearest integer, in binary\n        (or -1 if there are no such numbers)\n\n        a=4, b=7 => '110' because the mean of 4, 5, 6 is 5 which is 110 in binary";
	static sat (s, a = -103252, b = 10657) {
		const value = parseInt(s, 2);
		const numbers = [];
		for (let i = a; i < b; i++) numbers.push(i);
		if (!numbers.length) return value === -1;
		const mean = numbers.reduce((acc, n) => acc + n, 0) / numbers.length;
		const lower = Math.floor(mean);
		const upper = Math.ceil(mean);
		return (
			Math.abs(mean - value) <=
			Math.min(Math.abs(mean - lower), Math.abs(mean - upper))
		);
	}

	static sol (a, b) {
		const numbers = [];
		for (let i = a; i < b; i++) numbers.push(i);
		if (!numbers.length) return "-1";
		const mean = numbers.reduce((acc, n) => acc + n, 0) / numbers.length;
		return Math.round(mean).toString(2);
	}

	getExample () {
		return { a: -103252, b: 10657 };
	}

	genRandom () {
		const sign = this.random.choice([-1, 1]);
		const a = sign * this.random.randrange(10 ** this.random.randrange(5));
		const b = this.random.randrange(10 ** this.random.randrange(6));
		this.add({ a, b });
	}
}

export class SortedOdds extends PuzzleGenerator {
	/** Inspired by HumanEval #104 */
	static docstring = "* Find the sublist of numbers with only odd digits in increasing order\n\n        [17, 21, 18, 1, 4] => [1, 17, 21]";
	static sat (
		sub,
		nums = [17, 20, -100, 101, 423258, 19949, 0, 20174, 9351773, -11],
	) {
		if (sub.length > nums.length) return false;
		const isOddDigits = (n) =>
			[...Math.abs(n).toString()].every((c) => Number(c) % 2 === 1);
		const remaining = new Map();
		for (const n of nums) remaining.set(n, (remaining.get(n) || 0) + 1);
		for (let i = 0; i < sub.length; i++) {
			const n = sub[i];
			if (!isOddDigits(n)) return false;
			if (Math.min(...sub.slice(i)) !== n) return false;
			if (!remaining.has(n) || remaining.get(n) <= 0) return false;
			remaining.set(n, remaining.get(n) - 1);
		}
		for (const [n, count] of remaining) {
			if (count > 0 && isOddDigits(n)) return false;
		}
		return true;
	}

	static sol (nums) {
		const isOddDigits = (n) =>
			[...Math.abs(n).toString()].every((c) => Number(c) % 2 === 1);
		return nums.filter(isOddDigits).sort((a, b) => a - b);
	}

	getExample () {
		return {
			nums: [17, 20, -100, 101, 423258, 19949, 0, 20174, 9351773, -11],
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.randrange(
				-(10 ** this.random.randrange(8)),
				10 ** this.random.randrange(8),
			),
		);
		this.add({ nums });
	}
}

export class BackwardsDigits extends PuzzleGenerator {
	/** Inspired by HumanEval #105 */
	static docstring = "* Return the single digits in nums sorted backwards and converted to English words\n\n        [2, 3, 4, 5, 17] => ['five', 'four', 'three', 'two']";
	static sat (
		backwardsDigits,
		nums = [0, 2, 14, -2, 3, 8, 4, 5, 5, 7, 21, 101, 41, 2, 9, 6],
	) {
		const digitNames = [
			"",
			"one",
			"two",
			"three",
			"four",
			"five",
			"six",
			"seven",
			"eight",
			"nine",
		];
		const values = backwardsDigits.map((word) => digitNames.indexOf(word));
		if (values.includes(-1)) return false;
		for (let i = 0; i < values.length; i++) {
			const n = values[i];
			if (n === 0) return false;
			if (n !== Math.max(...values.slice(i))) return false;
			const occurrences = values.filter((v) => v === n).length;
			const expected = nums.filter((v) => v === n).length;
			if (occurrences !== expected) return false;
		}
		return nums.every((n) => {
			if (n >= 1 && n <= 9) {
				return values.includes(n);
			}
			return true;
		});
	}

	static sol (nums) {
		const digitNames = [
			"",
			"one",
			"two",
			"three",
			"four",
			"five",
			"six",
			"seven",
			"eight",
			"nine",
		];
		return nums
			.filter((n) => n >= 1 && n <= 9)
			.sort((a, b) => b - a)
			.map((n) => digitNames[n]);
	}

	getExample () {
		return {
			nums: [0, 2, 14, -2, 3, 8, 4, 5, 5, 7, 21, 101, 41, 2, 9, 6],
		};
	}

	genRandom () {
		const limit = this.random.choice([12, 100]);
		const nums = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.randrange(-5, limit),
		);
		this.add({ nums });
	}
}

export class AlternatingFactorials extends PuzzleGenerator {
	/** Inspired by HumanEval #106 */
	static docstring = "* Output a list of n integers, where the mth entry is m! if m is even or else (1+2+...+m)\n\n        5 => [1, 2, 6, 9, 120]";
	static sat (li, n = 100) {
		if (li.length !== n) return false;
		for (let i = 0; i < n; i++) {
			const m = li[i];
			if (i < 2) {
				if (m !== i + 1) return false;
			} else if (i % 2 === 1) {
				if (m !== li[i - 2] + i + (i + 1)) return false;
			} else {
				if (m !== li[i - 2] * i * (i + 1)) return false;
			}
		}
		return true;
	}

	static sol (n) {
		const ans = [];
		for (let i = 0; i < n; i++) {
			let value;
			if (i < 2) value = i + 1;
			else if (i % 2 === 1) value = ans[i - 2] + i + (i + 1);
			else value = ans[i - 2] * i * (i + 1);
			ans.push(value);
		}
		return ans;
	}

	getExample () {
		return { n: 100 };
	}

	genRandom () {
		const n = this.random.randrange(100);
		this.add({ n });
	}
}

export class EvenPalindromeNumbers extends PuzzleGenerator {
	/** Inspired by HumanEval #107 */
	static docstring = "* Find all even palindromes up to n\n\n        3 => [0, 2]";
	static sat (pals, n = 1099, count = 49) {
		const seen = new Set();
		for (const num of pals) {
			if (num < 0 || num > n || num % 2 !== 0) return false;
			const s = num.toString();
			if (s !== [...s].reverse().join("")) return false;
			seen.add(num);
		}
		return seen.size >= count;
	}

	static sol (n) {
		const ans = [];
		for (let i = 0; i <= n; i += 2) {
			const s = i.toString();
			if (s === [...s].reverse().join("")) ans.push(i);
		}
		return ans;
	}

	getExample () {
		return { n: 1099, count: 49 };
	}

	genRandom () {
		const n = this.random.randrange(10 * 1000);
		const pals = EvenPalindromeNumbers.sol(n);
		this.add({ n, count: pals.length });
	}
}

export class PositiveDigitSums extends PuzzleGenerator {
	/** Inspired by HumanEval #108 */
	static docstring = "* Filter for the numbers in nums whose sum of digits is > 0, where the first digit can be negative.\n\n        [12, -7, -102, -100] => [12, -102]";
	static sat (pos, nums = [-804, 9124, -945, 2410, 0, 21, -123]) {
		const checkSum = (n) => {
			const text = n.toString();
			const firstTwo = parseInt(text.slice(0, 2), 10);
			const rest = text.slice(2);
			const tail = rest.split("").reduce((acc, c) => acc + Number(c), 0);
			return firstTwo + tail > 0;
		};
		const combined = [...pos, ...nums];
		for (const n of combined) {
			if (checkSum(n)) {
				if (!pos.includes(n)) return false;
			} else if (pos.includes(n)) {
				return false;
			}
		}
		return true;
	}

	static sol (nums) {
		const checkSum = (n) => {
			const text = n.toString();
			const firstTwo = parseInt(text.slice(0, 2), 10);
			const rest = text.slice(2);
			const tail = rest.split("").reduce((acc, c) => acc + Number(c), 0);
			return firstTwo + tail > 0;
		};
		return nums.filter(checkSum);
	}

	getExample () {
		return { nums: [-804, 9124, -945, 2410, 0, 21, -123] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(1, 10) }, () =>
			this.random.randrange(-(10 ** 5), 5 * 10 ** 4),
		);
		this.add({ nums });
	}
}

export class RotateSort extends PuzzleGenerator {
	/** Inspired by HumanEval #109 */
	static docstring = "* \n        An array is ring-sorted if it is a \"rotation\" of a non-decreasing list.\n        Remove at most one element from arr to make it ring-sorted.\n\n        [1, 2, 3, -1, 6, 0] => [1, 2, 3, -1, 0]";
	static sat (original, arr = [2, 3, -1, -1, 0, 1, 1]) {
		const makeRing = (list) =>
			[...list].sort((a, b) => a - b).concat([...list].sort((a, b) => a - b));
		const ring = makeRing(original).join(",");
		if (!ring.includes(original.join(","))) return false;
		for (let i = 0; i <= arr.length; i++) {
			const candidate = arr.slice(0, i).concat(arr.slice(i + 1));
			if (
				candidate.length === original.length &&
				candidate.every((v, idx) => v === original[idx])
			) {
				return true;
			}
		}
		return false;
	}

	static sol (arr) {
		const isRingSorted = (list) => {
			const sorted = [...list].sort((a, b) => a - b);
			const ring = [...sorted, ...sorted].join(",");
			return ring.includes(list.join(","));
		};
		const candidates = [
			arr,
			...arr.map((_, i) => arr.slice(0, i).concat(arr.slice(i + 1))),
		];
		for (const candidate of candidates) {
			if (isRingSorted(candidate)) return candidate;
		}
		return arr;
	}

	getExample () {
		return { arr: [2, 3, -1, -1, 0, 1, 1] };
	}

	genRandom () {
		const n = this.random.randrange(10);
		const original = Array.from({ length: n }, () =>
			this.random.randrange(10),
		).sort((a, b) => a - b);
		const shift = this.random.randrange(n + 1);
		const rotated = original.slice(shift).concat(original.slice(0, shift));
		rotated.splice(
			this.random.randrange(rotated.length + 1),
			0,
			this.random.randrange(10),
		);
		this.add({ arr: rotated });
	}
}

export class ParityExchange extends PuzzleGenerator {
	/** Inspired by HumanEval #110 */
	static docstring = "* \n        Find a sequence of swaps (indices into two lists) such that, after making those swaps, all numbers in the\n        first list are even\n\n        [1, 3, 4] [2, 4, 5] => [0, 1]";
	static sat (
		swaps,
		nums1 = [1, 3, 2, 4, 5, 8, 7, 11],
		nums2 = [0, 7, 0, 8, 19, 4, 41, 43, 42],
	) {
		const copy1 = [...nums1];
		const copy2 = [...nums2];
		for (const [i, j] of swaps) {
			const tmp = copy1[i];
			copy1[i] = copy2[j];
			copy2[j] = tmp;
		}
		return copy1.every((n) => n % 2 === 0);
	}

	static sol (nums1, nums2) {
		const odds = nums1
			.map((n, i) => (n % 2 !== 0 ? i : -1))
			.filter((i) => i !== -1);
		const evens = nums2
			.map((n, i) => (n % 2 === 0 ? i : -1))
			.filter((i) => i !== -1);
		const pairs = [];
		for (let idx = 0; idx < Math.min(odds.length, evens.length); idx++) {
			pairs.push([odds[idx], evens[idx]]);
		}
		return pairs;
	}

	getExample () {
		return {
			nums1: [1, 3, 2, 4, 5, 8, 7, 11],
			nums2: [0, 7, 0, 8, 19, 4, 41, 43, 42],
		};
	}

	genRandom () {
		const nums1 = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-10, 10),
		);
		const nums2 = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-10, 10),
		);
		if (
			nums1.filter((n) => n % 2 !== 0).length <=
			nums2.filter((n) => n % 2 === 0).length
		) {
			this.add({ nums1, nums2 });
		}
	}
}

export class CharCounts extends PuzzleGenerator {
	/** Inspired by HumanEval #111 */
	static docstring = "* Find a string consisting of space-separated characters with given counts\n\n        {\"f\": 1, \"o\": 2} => \"oof\"";
	static sat (s, counts = { a: 4, b: 17, d: 101, e: 0, f: 12 }) {
		const chars = s.split(" ");
		for (const ch of chars) {
			if (chars.filter((c) => c === ch).length !== counts[ch]) return false;
		}
		return (
			chars.length === Object.values(counts).reduce((sum, v) => sum + v, 0)
		);
	}

	static sol (counts) {
		return Object.entries(counts)
			.flatMap(([char, times]) => Array(times).fill(char))
			.join(" ");
	}

	getExample () {
		return { counts: { a: 4, b: 17, d: 101, e: 0, f: 12 } };
	}

	genRandom () {
		const alpha = "abcdefghijklmnopqrstuvwxyz";
		const counts = {};
		for (let i = 0; i < this.random.randrange(10); i++) {
			const letter = alpha[this.random.randrange(alpha.length)];
			counts[letter] = (counts[letter] || 0) + this.random.randrange(10);
		}
		this.add({ counts });
	}
}

export class DelPalindrome extends PuzzleGenerator {
	/** Inspired by HumanEval #112 */
	static docstring = "* \n        Return a pair of a strings where the first string is the same as a with all the characters of b removed,\n        and the second string is 'True' if this string is a palindrome otherwise 'False'.\n\n        a=\"madam, I'm adam.\" b = \"Yes, we're here.\" => ['madamImadam', 'True']";
	static sat (strings, a = "this is a test", b = "cat") {
		const [s, isPal] = strings;
		let built = "";
		for (const char of a) {
			if (!b.includes(char)) built += char;
		}
		return s === built && `${s === [...s].reverse().join("")}` === isPal;
	}

	static sol (a, b) {
		const filtered = [...a].filter((char) => !b.includes(char)).join("");
		return [filtered, `${filtered === [...filtered].reverse().join("")}`];
	}

	getExample () {
		return { a: "this is a test", b: "cat" };
	}

	genRandom () {
		const a = this.random.pseudo_word(0, 50);
		const b = this.random.pseudo_word(0, 10);
		this.add({ a, b });
	}
}

export class ReplaceMe extends PuzzleGenerator {
	/** Inspired by HumanEval #113 */
	static docstring = "* For each string in lst, count the number of odd digits. Find a string with no t's such that replacing\n        this number by t gives the string 'this is a test'\n\n        [\"123\", \"2\"] => [\"2his is a 2es2\", \"0his a 0es0\"]";
	static sat (answers, lst = ["234515", "21503", "2506236943"]) {
		if (answers.length !== lst.length) return false;
		for (let i = 0; i < lst.length; i++) {
			const oddDigits = lst[i]
				.split("")
				.filter((c) => Number(c) % 2 === 1).length;
			if (answers[i].includes("t")) return false;
			if (
				answers[i].replace(new RegExp(oddDigits, "g"), "t") !== "this is a test"
			)
				return false;
		}
		return true;
	}

	static sol (lst) {
		return lst.map((value) => {
			const oddDigits = value
				.split("")
				.filter((c) => Number(c) % 2 === 1).length;
			return "this is a test".replace(/t/g, oddDigits.toString());
		});
	}

	getExample () {
		return { lst: ["234515", "21503", "2506236943"] };
	}

	genRandom () {
		const lst = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(10 ** this.random.randrange(10)).toString(),
		);
		this.add({ lst });
	}
}

export class MinSubArraySum extends PuzzleGenerator {
	/** Inspired by HumanEval #114 */
	static docstring = "* Find the start and end of the smallest-sum subarray of [(base^i mod p) - p/2 for i=start,..., end]\n\n        base=3, p=7, upper =-3 => [0, 3]\n        # because -3 is the sum of the elements [0:3] of [-2, 0, -1, 3, 1, 2, -2, 0, -1, 3 ...";
	static sat (startEnd, base = 7, p = 50741, upper = -4897754) {
		const [start, end] = startEnd;

		const modPow = (b, exp, mod) => {
			let result = 1;
			b = b % mod;
			while (exp > 0) {
				if (exp % 2 === 1) result = (result * b) % mod;
				b = (b * b) % mod;
				exp = Math.floor(exp / 2);
			}
			return result;
		};

		let sum = 0;
		for (let i = start; i < end; i++) {
			sum += modPow(base, i, p) - Math.floor(p / 2);
		}
		return sum <= upper;
	}

	static sol (base, p, upper) {
		let tot = 0;
		let bestTot = 0;
		let bestStart = 0;
		let bestEnd = 0;
		let largestCum = 0;
		let largestCumIdx = 0;
		let n = 1;
		for (let i = 0; i <= p; i++) {
			if (tot > largestCum) {
				largestCum = tot;
				largestCumIdx = i;
			}
			if (tot - largestCum < bestTot) {
				bestTot = tot - largestCum;
				bestStart = largestCumIdx;
				bestEnd = i;
			}
			tot += n - Math.floor(p / 2);
			n = (n * base) % p;
		}
		return [bestStart, bestEnd];
	}

	getExample () {
		return { base: 7, p: 50741, upper: -4897754 };
	}

	genRandom () {
		const p = this.random.randrange(2, 10 * 1000);
		const base = this.random.randrange(1, p);
		let total = 0;
		let bestTot = 0;
		let largestCum = 0;
		let n = 1;
		for (let i = 0; i <= p; i++) {
			if (total > largestCum) largestCum = total;
			if (total - largestCum < bestTot) bestTot = total - largestCum;
			total += n - Math.floor(p / 2);
			n = (n * base) % p;
		}
		const upper = bestTot;
		this.add({ base, p, upper });
	}
}

export class Buckets extends PuzzleGenerator {
	/** Inspired by HumanEval #115 */
	static docstring = "* Given a grid, partition the 1's into groups of capacity [x, y] pairs, with at most one incomplete group";
	static sat (
		wells,
		grid = [
			[1, 1, 0, 1, 1],
			[0, 0, 0, 0, 0],
			[1, 1, 0, 0, 1],
		],
		capacity = 2,
	) {
		const gridCopy = grid.map((row) => row.map(() => 0));
		for (const group of wells) {
			if (group.length > capacity) return false;
			for (const [i, j] of group) {
				if (gridCopy[i][j] !== 0) return false;
				gridCopy[i][j] = 1;
			}
		}
		if (wells.filter((group) => group.length !== capacity).length > 1)
			return false;
		return gridCopy.every((row, i) =>
			row.every((cell, j) => cell === grid[i][j]),
		);
	}

	static sol (grid, capacity) {
		const ans = [];
		for (let i = 0; i < grid.length; i++) {
			for (let j = 0; j < grid[0].length; j++) {
				if (grid[i][j] === 1) {
					if (!ans.length || ans[ans.length - 1].length === capacity)
						ans.push([]);
					ans[ans.length - 1].push([i, j]);
				}
			}
		}
		return ans;
	}

	getExample () {
		return {
			grid: [
				[1, 1, 0, 1, 1],
				[0, 0, 0, 0, 0],
				[1, 1, 0, 0, 1],
			],
			capacity: 2,
		};
	}

	genRandom () {
		const m = this.random.randrange(1, 10);
		const n = this.random.randrange(1, 10);
		const grid = Array.from({ length: m }, () =>
			Array.from({ length: n }, () => this.random.randrange(2)),
		);
		const capacity = this.random.randrange(1, 10);
		this.add({ grid, capacity });
	}
}

export class BinarySort extends PuzzleGenerator {
	/** Inspired by HumanEval #116 */
	static docstring = "* Sort the numbers in arr based on the number of 1's in their binary representation.\n\n        [1, 2, 3, 4, 6] => [1, 2, 4, 3, 6]";
	static sat (ordered, arr = [4, 2, 3, -1, 15, 2, 6, 9, 5, 16, 1048576]) {
		if (ordered.length !== arr.length) return false;
		const sortedArr = [...arr].sort((a, b) => a - b);
		const sortedOrdered = [...ordered].sort((a, b) => a - b);
		if (sortedArr.some((value, index) => value !== sortedOrdered[index]))
			return false;
		const ones = (n) =>
			n
				.toString(2)
				.replace(/-/g, "")
				.split("")
				.filter((digit) => digit === "1").length;
		for (let i = 1; i < ordered.length; i += 1) {
			if (ones(ordered[i - 1]) > ones(ordered[i])) return false;
		}
		return true;
	}

	static sol (arr) {
		const ones = (n) =>
			n
				.toString(2)
				.replace(/-/g, "")
				.split("")
				.filter((digit) => digit === "1").length;
		return arr
			.map((value, index) => ({ value, index }))
			.sort((a, b) => {
				const delta = ones(a.value) - ones(b.value);
				return delta !== 0 ? delta : a.index - b.index;
			})
			.map((entry) => entry.value);
	}

	getExample () {
		return { arr: [4, 2, 3, -1, 15, 2, 6, 9, 5, 16, 1048576] };
	}

	genRandom () {
		const arr = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ arr });
	}
}

export class ConsonantFilter extends PuzzleGenerator {
	/** Inspired by HumanEval #117 */
	static docstring = "* Find all words in the string with n consonants\n\n        Sample input:\n        s=\"An eye for an I\", n=1\n        Sample output:\n        [\"An\", \"eye\", \"an\"]";
	static sat (words, s = "This is not a very hard puzzle", n = 3) {
		let i = 0;
		for (const word of s.split(" ")) {
			const consonants = word
				.toLowerCase()
				.split("")
				.filter((c) => !"aeiou".includes(c)).length;
			if (consonants === n) {
				if (words[i] !== word) return false;
				i += 1;
			}
		}
		return i === words.length;
	}

	static sol (s, n) {
		return s.split(" ").filter(
			(word) =>
				word
					.toLowerCase()
					.split("")
					.filter((c) => !"aeiou".includes(c)).length === n,
		);
	}

	getExample () {
		return { s: "This is not a very hard puzzle", n: 3 };
	}

	genRandom () {
		const s = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.pseudo_word(),
		).join(" ");
		const n = this.random.randrange(7);
		this.add({ s, n });
	}
}

export class VowelSandwich extends PuzzleGenerator {
	/** Inspired by HumanEval #118 */
	static docstring = "* Find any vowel sandwich, a string consisting of a vowel between two consonants, contained in s\n\n        \"sandwhich\" => \"hic\"";
	static sat (ham, s = "Any vowel is OK") {
		const vowels = "aeiou";
		const consonants = "bcdfghjklmnpqrstvwxz";
		return (
			s.includes(ham) &&
			consonants.includes(ham[0].toLowerCase()) &&
			vowels.includes(ham[1].toLowerCase()) &&
			consonants.includes(ham[2].toLowerCase())
		);
	}

	static sol (s) {
		const vowels = "aeiou";
		const consonants = "bcdfghjklmnpqrstvwxz";
		for (let i = 1; i < s.length - 1; i++) {
			const trip = s.slice(i - 1, i + 2);
			if (
				consonants.includes(trip[0].toLowerCase()) &&
				vowels.includes(trip[1].toLowerCase()) &&
				consonants.includes(trip[2].toLowerCase())
			) {
				return trip;
			}
		}
		throw new Error("Ham not found");
	}

	getExample () {
		return { s: "Any vowel is OK" };
	}

	genRandom () {
		const vowels = "aeiou";
		const consonants = "bcdfghjklmnpqrstvwxz";
		const pieces = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.pseudo_word(),
		);
		const ham = `${consonants[this.random.randrange(consonants.length)]}${vowels[this.random.randrange(vowels.length)]}${consonants[this.random.randrange(consonants.length)]}`;
		pieces.push(ham);
		this.add({ s: pieces.join(" ") });
	}
}

export class ParenthesesPermutation extends PuzzleGenerator {
	/** Inspired by HumanEval #119 */
	static docstring = "* The string s consists of groups of parentheses separated by spaces.\n        Permute the groups such that the parentheses match.\n\n        \"( ) )(\" => \"( )( )\"";
	static sat (
		perm,
		s = "))(  )()()() )))(( ))))((( )))))(((( ))))))))((((((( ))))))((((( )))))))(((((( )))))))))(((((((  ((((((((((",
	) {
		const permParts = perm.trim().split(/\s+/).sort();
		const sParts = s.trim().split(/\s+/).sort();

		if (JSON.stringify(permParts) !== JSON.stringify(sParts)) {
			throw new Error("Must be a permutation of the space-delimited 'groups'");
		}

		// Check prefix balance: count of '(' must always be >= count of ')'
		let open = 0;
		let closed = 0;
		for (let i = 0; i < perm.length; i++) {
			if (perm[i] === "(") open++;
			else if (perm[i] === ")") closed++;

			if (open < closed) return false;
		}
		return true;
	}

	static sol (s) {
		const parts = s.trim().split(/\s+/);

		const countChar = (str, char) => str.split(char).length - 1;

		const getMinDepth = (part) => {
			let ans = 0;
			let depth = 0;
			for (const c of part) {
				if (c === ")") {
					depth--;
					ans = Math.min(ans, depth);
				} else {
					depth++;
				}
			}
			return ans;
		};

		const greedyReorder = (subs) => {
			const result = [];
			const queue = [...subs];
			let height = 0;

			while (queue.length > 0) {
				// Filter candidates that don't dip below zero given current height
				const candidates = queue.filter(
					(sub) => getMinDepth(sub) + height >= 0,
				);

				if (candidates.length === 0) break; // Should not happen with valid input

				// Pick the one that increases total height the most
				const best = candidates.reduce((a, b) => {
					const diffA = countChar(a, "(") - countChar(a, ")");
					const diffB = countChar(b, "(") - countChar(b, ")");
					return diffA >= diffB ? a : b;
				});

				height += countChar(best, "(") - countChar(best, ")");
				result.push(best);
				queue.splice(queue.indexOf(best), 1);
			}
			return result;
		};

		const mirror = (sub) => {
			return sub
				.split("")
				.reverse()
				.map((c) => (c === "(" ? ")" : "("))
				.join("");
		};

		// Separate parts that increase/maintain height from those that decrease it
		const lefts = parts.filter((p) => countChar(p, "(") >= countChar(p, ")"));
		const rights = parts
			.filter((p) => countChar(p, "(") < countChar(p, ")"))
			.map(mirror);

		const orderedLefts = greedyReorder(lefts);
		const orderedRights = greedyReorder(rights);

		// Mirror rights back and reverse their order to close the parentheses properly
		const finalRights = orderedRights.reverse().map(mirror);

		return [...orderedLefts, ...finalRights].join(" ");
	}

	getExample () {
		return {
			s: "))(  )()()() )))(( ))))((( )))))(((( ))))))))((((((( ))))))((((( )))))))(((((( )))))))))(((((((  ((((((((((",
		};
	}

	static genRandom () {
		const parts = [];
		let depth = 0;
		let buffer = "";

		const randomChoice = (str) => str[Math.floor(Math.random() * str.length)];

		while (depth > 0 || Math.random() > 0.1) {
			const c = randomChoice(depth > 0 ? "()()()()())" : "(");
			if (c === "(") depth++;
			else if (c === ")") depth--;

			buffer += c;

			if (Math.floor(Math.random() * 10) === 0) {
				parts.push(buffer);
				buffer = "";
			}
		}
		parts.push(buffer);

		// Shuffle
		parts.sort(() => Math.random() - 0.5);
		return parts.join(" ");
	}
}

export class BiggestK extends PuzzleGenerator {
	/** Inspired by HumanEval #120 */
	static docstring = "* Find the largest k numbers\n\n        k=2, [1, 2, 3, 4, 5, 5, 3, 5, 2] => [5, 5]";
	static sat (
		biggest,
		k = 7,
		nums = [31, 1, 2, -10, -2, 4, 17, 18, 20, 14, 20, 21, 18, 0],
	) {
		if (biggest.length !== k) return false;
		const remaining = [...nums];
		for (const n of biggest) {
			const idx = remaining.indexOf(n);
			if (idx === -1) return false;
			remaining.splice(idx, 1);
		}
		if (k === 0 || k === nums.length) return true;
		return Math.max(...remaining) <= Math.min(...biggest);
	}

	static sol (k, nums) {
		return [...nums].sort((a, b) => b - a).slice(0, k);
	}

	getExample () {
		return {
			k: 7,
			nums: [31, 1, 2, -10, -2, 4, 17, 18, 20, 14, 20, 21, 18, 0],
		};
	}

	genRandom () {
		const length = this.random.randrange(20);
		const nums = Array.from({ length }, () => this.random.randrange(-10, 100));
		const k = this.random.randrange(length + 1);
		this.add({ k, nums });
	}
}

export class OddEvenSum extends PuzzleGenerator {
	/** Inspired by HumanEval #121 */
	static docstring = "* Find the sum of the odd elements that are at even indices\n\n        [0, 1, 2, 3, 5, 6] => 5";
	static sat (tot, nums = [18, 42152, 125023521, -1221873620123, 17, 19]) {
		for (let i = 0; i < nums.length; i += 2) {
			if (Math.abs(nums[i]) % 2 === 1) tot -= nums[i];
		}
		return tot === 0;
	}

	static sol (nums) {
		return nums.reduce(
			(sum, n, idx) => (idx % 2 === 0 && Math.abs(n) % 2 === 1 ? sum + n : sum),
			0,
		);
	}

	getExample () {
		return {
			nums: [18, 42152, 125023521, -1221873620123, 17, 19],
		};
	}

	genRandom () {
		const length = this.random.randrange(20);
		const nums = Array.from({ length }, () => this.random.randrange(-100, 100));
		this.add({ nums });
	}
}

export class LongEarlySum extends PuzzleGenerator {
	/** Inspired by HumanEval #122 */
	static docstring = "* Find the sum of the numbers among the first k with more than 2 digits\n\n        k=3, nums=[2, 102, 12, 1000] => 102";
	static sat (
		tot,
		k = 5,
		nums = [1252, 125273523, 0, 42, 100, 214532, 2, 0, 11, 14],
	) {
		for (let i = 0; i < Math.min(k, nums.length); i++) {
			if (Math.abs(nums[i]).toString().length > 2) tot -= nums[i];
		}
		return tot === 0;
	}

	static sol (k, nums) {
		return nums
			.slice(0, k)
			.reduce(
				(sum, n) => (Math.abs(n).toString().length > 2 ? sum + n : sum),
				0,
			);
	}

	getExample () {
		return {
			k: 5,
			nums: [1252, 125273523, 0, 42, 100, 214532, 2, 0, 11, 14],
		};
	}

	genRandom () {
		const length = this.random.randrange(1, 20);
		const nums = Array.from({ length }, () =>
			this.random.randrange(-1000000000, 1000000000),
		);
		const k = this.random.randrange(10);
		this.add({ k, nums });
	}
}

export class OddCollatz extends PuzzleGenerator {
	/** Inspired by HumanEval #123 */
	static docstring = "* Find the odd numbers in the collatz sequence starting at n\n\n        3 => [3, 5, 1]  # because the Collatz sequence starting with 3 is [3, 10, 5, 16, 8, 4, 2, 1]";
	static sat (odds, n = 1243272912731) {
		let numOdds = 0;
		while (true) {
			if (n % 2 !== 0) {
				numOdds += 1;
				if (!odds.includes(n)) return false;
			}
			if (n <= 1) return numOdds === odds.length;
			n = n % 2 !== 0 ? 3 * n + 1 : Math.floor(n / 2);
		}
	}

	static sol (n) {
		const ans = [];
		while (true) {
			if (n % 2 !== 0) ans.push(n);
			if (n <= 1) return ans;
			n = n % 2 !== 0 ? 3 * n + 1 : Math.floor(n / 2);
		}
	}

	getExample () {
		return { n: 1243272912731 };
	}

	genRandom () {
		const exponent = this.random.randrange(1, 10);
		const bound = 10 ** exponent;
		const n = this.random.randrange(1, bound);
		this.add({ n });
	}
}

export class DateDiff extends PuzzleGenerator {
	/** Inspired by HumanEval #124 */
	static docstring = "* Find a valid date mm-dd-yyyy such that the date, viewed as a mathematical expression, evaluates to target\n\n        -2029 => \"10-18-2021\" # because 10-18-2021 == -2029";
	static sat (str, target = -2075) {
		if (str[2] !== "-" || str[5] !== "-") return false;
		const parts = str.split("-");
		if (parts.length !== 3) return false;
		const [m, d, y] = parts.map((p) => Number(p));
		if ([m, d, y].some((n) => Number.isNaN(n))) return false;
		if (m < 1 || m > 12) return false;
		if (d < 1 || d > 31) return false;
		if ([4, 6, 9, 11].includes(m) && d > 30) return false;
		if (m === 2 && d > 29) return false;
		return m - d - y === target;
	}

	static sol (target) {
		if (target >= -30) {
			return `12-01-${String(11 - target).padStart(4, "0")}`;
		}
		return `01-31-${String(-30 - target).padStart(4, "0")}`;
	}

	getExample () {
		return { target: -2075 };
	}

	genRandom () {
		const target = this.random.randrange(-10029, 12);
		this.add({ target });
	}
}

export class StrangeSplit extends PuzzleGenerator {
	/** Inspired by HumanEval #125 */
	static docstring = "* Split s into strings if there is a space in s, otherwise split on commas if there is a comma, otherwise\n        return the list of lowercase letters with odd order (order of a = 0, b = 1, etc.)\n\n        \"a b c\" => [\"a\", \"b\", \"c\"]\n        \"a,b\" => [\"a\", \"b\"]";
	static sat (lst, s = "Hello, world!") {
		if (s.includes(" ")) return lst.join(" ") === s;
		if (s.includes(",")) return lst.join(",") === s;
		const filtered = s
			.split("")
			.filter((c) => c >= "a" && c <= "z" && c.charCodeAt(0) % 2 === 0)
			.join("");
		return lst.join("") === filtered;
	}

	static sol (s) {
		if (s.includes(" ")) return s.split(" ");
		if (s.includes(",")) return s.split(",");
		return s
			.split("")
			.filter((c) => c >= "a" && c <= "z" && c.charCodeAt(0) % 2 === 0);
	}

	getExample () {
		return { s: "Hello, world!" };
	}

	genRandom () {
		const words = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.pseudo_word(),
		);
		const joiner = this.random.choice([" ", ",", ""]);
		const s = words.join(joiner);
		this.add({ s });
	}
}

export class IncreasingViolation extends PuzzleGenerator {
	/** Inspired by HumanEval #126 */
	static docstring = "* \n        Find the indices of two entries that show that the list is not in increasing order.\n        If there are no violations (they are increasing), return an empty list.\n\n        [1,2,3,0,4,5,6] => [1, 3]";
	static sat (
		violation,
		nums = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 17, 17, 18, 19, 20, 22, 24],
	) {
		if (!violation.length) {
			return nums.every(
				(n, idx) => idx === nums.length - 1 || n < nums[idx + 1],
			);
		}
		const [i, j] = violation;
		return (
			Number.isInteger(i) &&
			Number.isInteger(j) &&
			i >= 0 &&
			i < j &&
			nums[i] >= nums[j]
		);
	}

	static sol (nums) {
		for (let i = 0; i < nums.length - 1; i++) {
			if (nums[i] >= nums[i + 1]) return [i, i + 1];
		}
		return [];
	}

	getExample () {
		return {
			nums: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 17, 17, 18, 19, 20, 22, 24],
		};
	}

	genRandom () {
		const length = this.random.randrange(2, 20);
		const nums = Array.from({ length }, () => this.random.randrange(100)).sort(
			(a, b) => a - b,
		);
		if (this.random.randrange(2) === 1) {
			const idx = this.random.randrange(1, nums.length);
			nums[idx] = nums[idx - 1] - 1;
		}
		this.add({ nums });
	}
}

export class PrimeIntervalIntersection extends PuzzleGenerator {
	/** Inspired by HumanEval #127 */
	static docstring = "* Find an interval whose intersection with a given interval has a width that is a prime integer.\n\n        [7, 100] => [0, 10]  # because 10-7=3 is prime";
	static sat (interval2, interval1 = [32157, 93210127]) {
		const width =
			Math.min(interval1[1], interval2[1]) -
			Math.max(interval1[0], interval2[0]);
		if (width <= 1) return false;
		for (let i = 2; i * i <= width; i++) {
			if (width % i === 0) return false;
		}
		return true;
	}

	static sol (interval1) {
		const [a] = interval1;
		return [a, a + 2];
	}

	getExample () {
		return { interval1: [32157, 93210127] };
	}

	genRandom () {
		const a = this.random.randrange(100);
		const b = a + 2 + this.random.randrange(100);
		this.add({ interval1: [a, b] });
	}
}

export class ProductSigns extends PuzzleGenerator {
	/** Inspired by HumanEval #128 */
	static docstring = "* Find the sum of the magnitudes of the elements in the array with a sign that is equal to the product of\n        the signs of the entries.\n\n        [1, -2, 3] => -6  # negative because there is one negative";
	static sat (n, arr = [1, 7, -20052, 14, -3, -11, 1025235, 14]) {
		if (arr.length === 0) return n === 0;
		let tot = 0;
		for (const i of arr) {
			if (tot >= 0) tot += Math.abs(i);
			else tot -= Math.abs(i);
			if (i < 0) tot = -tot;
			else if (i === 0) {
				tot = 0;
				break;
			}
		}
		return n === tot;
	}

	static sol (arr) {
		const tot = arr.reduce((sum, n) => sum + Math.abs(n), 0);
		if (!arr.every((n) => n !== 0)) return 0;
		const negatives = arr.filter((n) => n < 0).length;
		return negatives % 2 === 0 ? tot : -tot;
	}

	getExample () {
		return { arr: [1, 7, -20052, 14, -3, -11, 1025235, 14] };
	}

	genRandom () {
		const arr = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ arr });
	}
}

export class LexPath extends PuzzleGenerator {
	/** Inspired by HumanEval #129 */
	static docstring = "* Find the lexicographically smallest path of length k in graph with given edge matrix (and no dead ends)\n\n        k=3, edges=[[1,3], [0, 3], [2], [3]] => [0, 1, 0] # because 0-1 and 1-0 are edges";
	static sat (path, k = 10, edges = [[2, 4], [3], [4, 1], [4], [0]]) {
		if (path.length !== k) return false;
		for (let i = 1; i < k; i++) {
			const prev = path[i - 1];
			const cur = path[i];
			if (!edges[prev] || !edges[prev].includes(cur)) return false;
		}

		const check = (prefix) => {
			for (let i = 0; i < prefix.length; i++) {
				if (path[i] !== prefix[i]) return path[i] < prefix[i];
			}
			if (prefix.length >= k) return true;
			const last = prefix[prefix.length - 1];
			const nextEdges = edges[last];
			if (!nextEdges) return false;
			return nextEdges.every((next) => check(prefix.concat(next)));
		};

		for (let i = 0; i < edges.length; i++) {
			if (!check([i])) return false;
		}

		return true;
	}

	static sol (k, edges) {
		const path = [];
		while (path.length < k) {
			const last = path[path.length - 1];
			const next = last === undefined ? 0 : Math.min(...edges[last]);
			path.push(next);
		}
		return path;
	}

	getExample () {
		return {
			k: 10,
			edges: [[2, 4], [3], [4, 1], [4], [0]],
		};
	}

	genRandom () {
		const n = this.random.randrange(2, 6);
		const edges = Array.from({ length: n }, () => {
			const size = this.random.randrange(1, n + 1);
			return Array.from({ length: size }, () => this.random.randrange(n));
		});
		const k = this.random.randrange(20);
		this.add({ k, edges });
	}
}

export class Tribonacci extends PuzzleGenerator {
	/** Inspired by HumanEval #130 */
	static docstring = "* Find a sequence where seq[n] == 1 + n / 2 for even n, and\n        seq[n] == seq[n - 1] + seq[n - 2] + seq[n + 1] for odd n < length.";
	static sat (seq, length = 181) {
		if (seq.length <= length) return false;
		for (let n = 0; n < length; n++) {
			if (n % 2 === 0) {
				if (seq[n] !== 1 + Math.floor(n / 2)) return false;
			} else {
				const prev1 = seq[n - 1];
				const prev2 = n >= 2 ? seq[n - 2] : 0;
				if (seq[n + 1] === undefined) return false;
				if (seq[n] !== prev1 + prev2 + seq[n + 1]) return false;
			}
		}
		return true;
	}

	static sol (length) {
		const seq = [];
		while (seq.length <= length) {
			const n = seq.length;
			if (n % 2 === 0) {
				seq.push(1 + Math.floor(n / 2));
			} else {
				const prev1 = seq[n - 1];
				const prev2 = n >= 2 ? seq[n - 2] : 0;
				seq.push(prev1 + prev2 + (1 + Math.floor((n + 1) / 2)));
			}
		}
		return seq.concat([0]);
	}

	getExample () {
		return { length: 181 };
	}

	genRandom () {
		const length = this.random.randrange(1000);
		this.add({ length });
	}
}

export class OddProduct extends PuzzleGenerator {
	/** Inspired by HumanEval #131 */
	static docstring = "* Return the product of the odd digits in n, or 0 if there aren't any\n\n        12345 => 15";
	static sat (prod, n = "14235764939971075543215213") {
		const digits = String(n);
		let total = 1n;
		let seenOdd = false;
		for (const c of digits) {
			const digit = BigInt(c);
			if (digit % 2n === 1n) {
				seenOdd = true;
				total *= digit;
			}
		}
		const expected = seenOdd ? total : 0n;
		return BigInt(prod) === expected;
	}

	static sol (n) {
		const digits = String(n);
		let prod = 1n;
		let found = false;
		for (const c of digits) {
			const digit = BigInt(c);
			if (digit % 2n === 1n) {
				found = true;
				prod *= digit;
			}
		}
		return found ? prod : 0n;
	}

	getExample () {
		return { n: "14235764939971075543215213" };
	}

	genRandom () {
		const digits = Array.from({ length: this.random.randrange(1, 35) }, () =>
			this.random.randrange(10),
		);
		const n = digits.join("");
		this.add({ n });
	}
}

export class ValidBracketSubsequence extends PuzzleGenerator {
	/** Inspired by HumanEval #132 */
	static docstring = "* Find a valid substring of s that contains matching brackets, at least one of which is nested\n\n        \"]][][[]]]\" => \"[][[]]\"";
	static sat (
		valid,
		s = "]]]]]]]]]]]]]]]][[][][][]]]]]]]]]]][[[][[][[[[[][][][]][[[[[[[[[[[[[[[[[[",
	) {
		if (!s.includes(valid)) return false;
		let depth = 0;
		let maxDepth = 0;
		for (const c of valid) {
			if (c === "[") depth++;
			else if (c === "]") depth--;
			maxDepth = Math.max(maxDepth, depth);
			if (depth < 0) return false;
		}
		return depth === 0 && maxDepth > 1;
	}

	static sol (s) {
		const match = /\[((?:\[\])+)\]/.exec(s);
		return match ? match[0] : "";
	}

	getExample () {
		return {
			s: "]]]]]]]]]]]]]]][[][][][]]]]]]]]]]][[[][[][[[[[][][][]][[[[[[[[[[[[[[[[[[",
		};
	}

	genRandom () {
		const parts = Array.from({ length: 20 }, () =>
			this.random.choice(["[", "]"]),
		);
		const match = "[" + "[]".repeat(this.random.randrange(1, 10)) + "]";
		const idx = this.random.randrange(parts.length + 1);
		parts.splice(idx, 0, match);
		this.add({ s: parts.join("") });
	}
}

export class CeilingSquares extends PuzzleGenerator {
	/** Inspired by HumanEval #133 */
	static docstring = "* Round each float in x up to the next integer and return the running total of the integer squares\n\n        [2.4, 3.7, 0.1] => [9, 25, 26]";
	static sat (
		runningSquares,
		x = [201.1, 301.4, -18.1, 1244122.0, 10101.0101, 1e7],
	) {
		if (runningSquares.length !== x.length) return false;
		let total = 0;
		for (let i = 0; i < x.length; i++) {
			const ceiling = Number.isInteger(x[i]) ? x[i] : Math.ceil(x[i]);
			total += ceiling ** 2;
			if (runningSquares[i] !== total) return false;
		}
		return true;
	}

	static sol (x) {
		let total = 0;
		return x.map((value) => {
			const ceiling = Number.isInteger(value) ? value : Math.ceil(value);
			total += ceiling ** 2;
			return total;
		});
	}

	getExample () {
		return { x: [201.1, 301.4, -18.1, 1244122.0, 10101.0101, 1e7] };
	}

	genRandom () {
		const length = this.random.randrange(2, 10);
		const x = Array.from({ length }, () => this.random.uniform(-10, 10));
		this.add({ x });
	}
}

export class LastLetters extends PuzzleGenerator {
	/** Inspired by HumanEval #134 */
	static docstring = "* Determine, for each string in x, whether the last character is an isolated letter\n\n        [\"a b c\", \"abc\"] => [True, False]";
	static sat (
		y,
		x = [
			"Hello, world!",
			"cat",
			"",
			"a test",
			"test a",
			"i e",
			"o",
			"I O U",
			"You and I",
		],
	) {
		if (y.length !== x.length) return false;
		for (let i = 0; i < x.length; i++) {
			const last = x[i].split(" ").pop();
			const expected = last.length === 1 && /[a-zA-Z]/.test(last);
			if (expected !== y[i]) return false;
		}
		return true;
	}

	static sol (x) {
		return x.map((s) => {
			const last = s.split(" ").pop();
			return last.length === 1 && /[a-zA-Z]/.test(last);
		});
	}

	getExample () {
		return {
			x: [
				"Hello, world!",
				"cat",
				"",
				"a test",
				"test a",
				"i e",
				"o",
				"I O U",
				"You and I",
			],
		};
	}

	genRandom () {
		const x = Array.from({ length: this.random.randrange(30) }, () => {
			const words = Array.from({ length: this.random.randrange(3) }, () =>
				this.random.pseudo_word(),
			);
			if (this.random.randrange(3)) words.push(this.random.char());
			return words.join(" ");
		});
		this.add({ x });
	}
}

export class Drops extends PuzzleGenerator {
	/** Inspired by HumanEval #135 */
	static docstring = "* Find the indices for which the nums array drops.\n\n        [1,2,3,0,2,4,1] => [3,6]";
	static sat (
		dropIndexes,
		nums = [
			2, -1, 14, 8, 9, 9, 8, 4, 2, 4, 3, -100, 1000, 18, 4, -2, -3, -3, 1, 0,
		],
	) {
		const drops = [];
		for (let i = 1; i < nums.length; i++) {
			if (nums[i] < nums[i - 1]) drops.push(i);
		}
		return JSON.stringify(drops) === JSON.stringify(dropIndexes);
	}

	static sol (nums) {
		const drops = [];
		for (let i = 1; i < nums.length; i++) {
			if (nums[i] < nums[i - 1]) drops.push(i);
		}
		return drops;
	}

	getExample () {
		return {
			nums: [
				2, -1, 14, 8, 9, 9, 8, 4, 2, 4, 3, -100, 1000, 18, 4, -2, -3, -3, 1, 0,
			],
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(30) }, () =>
			this.random.randrange(-100, 100),
		);
		this.add({ nums });
	}
}

export class LargestNegSmallestPos extends PuzzleGenerator {
	/** Inspired by HumanEval #136 */
	static docstring = "* Find the largest negative ans smallest positive numbers (or 0 if none)\n\n        [-2, -4, 14, 50] => [-2, 14]\n        [3, 22] => [0, 3]";
	static sat (
		extremes,
		nums = [-10, -4, 100, -40, 2, 2, 3, 17, -50, -25, 18, 41, 9, 11, 15],
	) {
		const [neg, pos] = extremes;
		const negatives = nums.filter((n) => n < 0);
		const positives = nums.filter((n) => n > 0);
		if (neg === 0 && negatives.length) return false;
		if (neg !== 0 && neg >= 0) return false;
		if (neg !== 0 && !negatives.includes(neg)) return false;
		if (pos === 0 && positives.length) return false;
		if (pos !== 0 && pos <= 0) return false;
		if (pos !== 0 && !positives.includes(pos)) return false;
		return true;
	}

	static sol (nums) {
		const negatives = nums.filter((n) => n < 0);
		const positives = nums.filter((n) => n > 0);
		return [
			negatives.length ? Math.max(...negatives) : 0,
			positives.length ? Math.min(...positives) : 0,
		];
	}

	getExample () {
		return {
			nums: [-10, -4, 100, -40, 2, 2, 3, 17, -50, -25, 18, 41, 9, 11, 15],
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(20) }, () =>
			this.random.randrange(-1000, 1001),
		);
		this.add({ nums });
	}
}

export class LargestStringNum extends PuzzleGenerator {
	/** Inspired by HumanEval #137 */
	static docstring = "* Find the largest number where commas or periods are decimal points\n\n        [\"99,9\", \"100\"] => 100.0";
	static sat (
		x,
		strNums = [
			"1,3",
			"-11",
			"17.5",
			"-11",
			"2",
			"2.2",
			"2,2",
			"4",
			"-18,18",
			"99.09",
		],
	) {
		if (Math.max(...strNums.map((s) => Number(s.replace(",", ".")))) !== x)
			return false;
		return true;
	}

	static sol (strNums) {
		return Math.max(...strNums.map((s) => Number(s.replace(",", "."))));
	}

	getExample () {
		return {
			strNums: [
				"1,3",
				"-11",
				"17.5",
				"-11",
				"2",
				"2.2",
				"2,2",
				"4",
				"-18,18",
				"99.09",
			],
		};
	}

	genRandom () {
		const strNums = Array.from({ length: this.random.randrange(1, 20) }, () => {
			return this.random.random() > 0.5
				? String(this.random.randrange(-100, 101))
				: String(this.random.uniform(-100, 100)).replace(
					".",
					this.random.choice([",", "."]),
				);
		});
		this.add({ strNums });
	}
}

export class Even4Sum extends PuzzleGenerator {
	static docstring = "* Find four even positive integers that sum to n\n\n        Sample Input:\n        1234567890\n\n        Sample Output:\n        [2, 2, 2, 1234567884]";
	/** Inspired by HumanEval #138 */
	static sat (summands, n = 1234567890) {
		if (summands.length !== 4) return false;
		for (const s of summands) {
			if (s % 2 !== 0 || s <= 0) return false;
		}
		return summands.reduce((sum, n) => sum + n, 0) === n;
	}

	static sol (n) {
		return [2, 2, 2, n - 6];
	}

	getExample () {
		return { n: 1234567890 };
	}

	genRandom () {
		const n = 2 * this.random.randrange(4, 10 ** this.random.randrange(1, 10));
		this.add({ n });
	}
}

export class InverseSuperFactorial extends PuzzleGenerator {
	static docstring = "* Find n such that the super factorial equals the given value\n\n        Sample Input:\n        [1, 2, 1]\n\n        Sample Output:\n        [1, 2, 1]";
	/** Inspired by HumanEval #139 */
	static sat (nums, superFactorials = [1, 2, 1]) {
		for (let idx = 0; idx < superFactorials.length; idx++) {
			let fact = 1;
			let sf = 1;
			const n = nums[idx];
			for (let j = 1; j <= n; j++) {
				fact *= j;
				sf *= fact;
			}
			if (sf !== superFactorials[idx]) return false;
		}
		return true;
	}

	static sol (superFactorials) {
		const remaining = new Set(superFactorials);
		const cache = new Map();
		let fact = 1;
		let sf = 1;
		let n = 1;
		while (remaining.size) {
			fact *= n;
			sf *= fact;
			if (remaining.has(sf)) {
				cache.set(sf, n);
				remaining.delete(sf);
			}
			n += 1;
		}
		return superFactorials.map((value) => cache.get(value) ?? 0);
	}

	getExample () {
		return { superFactorials: [1, 2, 1] };
	}

	genRandom () {
		const superFactorials = Array.from({ length: 11 }, () => {
			let res = 1;
			let fact = 1;
			const n = this.random.randrange(10);
			for (let i = 1; i <= n; i++) {
				fact *= i;
				res *= fact;
			}
			return res;
		});
		this.add({ superFactorials });
	}
}

export class ExpandSpaces extends PuzzleGenerator {
	/** Inspired by HumanEval #140 */
	static docstring = "* Find a string such that, when three or more spaces are compacted to a '-' and one or two spaces are\n        replaced by underscores, leads to the target.\n\n        \"_o-k__?-\" => \"  o        k  ?     \"";
	static sat (orig, target = "-Hello,_world!__This_is-so-easy!-") {
		if (orig.includes("_") || orig.includes("-")) return false;
		let encoded = "";
		let count = 0;
		for (const char of orig) {
			if (char === " ") {
				count += 1;
				continue;
			}
			encoded += count > 2 ? "-" : "_".repeat(count);
			encoded += char;
			count = 0;
		}
		encoded += count > 2 ? "-" : "_".repeat(count);
		return encoded === target;
	}

	static sol (target) {
		return target.replace(/-/g, "   ").replace(/_/g, " ");
	}

	getExample () {
		return { target: "-Hello,_world!__This_is-so-easy!-" };
	}

	genRandom () {
		const target = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.choice([
				"-",
				"_",
				this.random.char(),
				this.random.pseudo_word(),
			]),
		)
			.join("")
			.replace(/ /g, "");
		if (
			target.includes("___") ||
			target.includes("-_") ||
			target.includes("_-") ||
			target.includes("--")
		)
			return;
		this.add({ target });
	}
}

export class FilenameOK extends PuzzleGenerator {
	static docstring = "* Return a list of Yes/No strings that determine whether candidate filename is valid. A valid filename\n        should end in .txt, .exe, or .dll, and should have at most three digits, no additional periods\n\n        [\"train.jpg\", \"doc10234.txt\", \"3eadme.txt\"] = [\"No\", \"No\", \"Yes\"]";
	/** Inspired by HumanEval #141 */
	static sat (
		valids,
		filenames = [
			"cat.txt",
			"!jog.dll",
			"31F9.html",
			"Is this okay?.txt",
			".exe",
			"",
		],
	) {
		if (valids.length !== filenames.length) return false;
		const allowed = ["txt", "dll", "exe"];
		for (let i = 0; i < valids.length; i++) {
			const v = valids[i];
			const f = filenames[i];
			const digits = Array.from(f).filter((c) => /\d/.test(c)).length;
			const parts = f.split(".");
			const prefix = parts[0] || "";
			const suffix = parts.slice(1);
			const hasAllowed = suffix.length === 1 && allowed.includes(suffix[0]);
			if (v === "Yes") {
				if (!hasAllowed) return false;
				if (!/^[A-Za-z]/.test(prefix)) return false;
				if (digits >= 4) return false;
			} else {
				if (v !== "No") return false;
				if (hasAllowed && /^[A-Za-z]/.test(prefix) && digits < 4) return false;
			}
		}
		return true;
	}

	static sol (filenames) {
		const allowed = ["txt", "dll", "exe"];
		return filenames.map((f) => {
			const parts = f.split(".");
			const prefix = parts[0] || "";
			const suffix = parts.slice(1);
			const hasAllowed = suffix.length === 1 && allowed.includes(suffix[0]);
			const digits = Array.from(f).filter((c) => /\d/.test(c)).length;
			return hasAllowed && /^[A-Za-z]/.test(prefix) && digits < 4
				? "Yes"
				: "No";
		});
	}

	getExample () {
		return {
			filenames: [
				"cat.txt",
				"!jog.dll",
				"31F9.html",
				"Is this okay?.txt",
				".exe",
				"",
			],
		};
	}

	genRandom () {
		const filenames = Array.from(
			{ length: this.random.randrange(1, 20) },
			() =>
				this.random.char() +
				this.random.pseudo_word() +
				this.random.char() +
				this.random.choice([".txt", ".dll", ".exe", ".mp4", ".tar.zip"]),
		);
		this.add({ filenames });
	}
}

export class FindStrangeSum extends PuzzleGenerator {
	static docstring = "* Find a list of integers such that tot is the sum of (n^2 if 3 | n, else n^3 if 4 | n, else n)";
	/** Inspired by HumanEval #142 */
	static sat (lst, tot = 1125181293221n) {
		const value = lst.reduce((acc, n) => {
			if (n % 3 === 0) return acc + BigInt(n) ** 2n;
			if (n % 4 === 0) return acc + BigInt(n) ** 3n;
			return acc + BigInt(n);
		}, 0n);
		return value === BigInt(tot);
	}

	static sol (tot) {
		const residue = (((BigInt(tot) - 1n) % 12n) + 12n) % 12n;
		const ones = Number(residue);
		return [...Array(ones)].map(() => 1).concat(Number(BigInt(tot) - residue));
	}

	getExample () {
		return { tot: 1125181293221 };
	}

	genRandom () {
		const sign = this.random.choice([-1, 1]);
		const magnitude = 10 ** this.random.randrange(10);
		const tot = sign * this.random.randrange(0, magnitude);
		this.add({ tot });
	}
}

export class PrimeWords extends PuzzleGenerator {
	/** Inspired by HumanEval #143 */
	static docstring = "* Find the string consisting of all the words whose lengths are prime numbers\n\n        \"A bird in the hand is worth two in the bush\" => \"in the is worth two in the\"";
	static sat (
		primes,
		s = "This is a test of whether you would want to do such strange puzzles",
	) {
		const words = s.split(" ");
		const primeWords = primes.split(" ");
		let idx = 0;
		for (const word of words) {
			if (isPrime(word.length)) {
				if (primeWords[idx] !== word) return false;
				idx += 1;
			}
		}
		return idx === primeWords.length;
	}

	static sol (s) {
		return s
			.split(" ")
			.filter((word) => isPrime(word.length))
			.join(" ");
	}

	getExample () {
		return {
			s: "This is a test of whether you would want to do such strange puzzles",
		};
	}

	genRandom () {
		const s = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.pseudo_word(),
		).join(" ");
		this.add({ s });
	}
}

export class SimplifyProductFraction extends PuzzleGenerator {
	/** Inspired by HumanEval #144 */
	static docstring = "* Write x * y as the shortest equivalent fraction using at most max_len chars\n\n        x=\"-2/3\", y=\"-3/8\", max_len=3 => \"1/4\"";
	static sat (z, x = "-8142432/763083", y = "66/-13474", max_len = 18) {
		const parse = (value) => value.split("/").map((digit) => Number(digit));
		const [[a, b], [c, d], [u, v]] = [x, y, z].map(parse);
		return a * c * v === b * d * u && z.length <= max_len;
	}

	static sol (x, y, max_len) {
		const parse = (value) => value.split("/").map((digit) => Number(digit));
		const [[a, b], [c, d]] = [x, y].map(parse);
		let num = a * c;
		let den = b * d;
		if (num < 0 && den < 0) {
			num = -num;
			den = -den;
		}
		if (num === 0) return "0/1";
		const dcd = gcd(Math.abs(num), Math.abs(den));
		return `${num / dcd}/${den / dcd}`;
	}

	static naive (x, y) {
		const parse = (value) => value.split("/").map((digit) => Number(digit));
		const [[a, b], [c, d]] = [x, y].map(parse);
		return `${a * c}/${b * d}`;
	}

	getExample () {
		return { x: "-8142432/763083", y: "66/-13474", max_len: 18 };
	}

	genRandom () {
		const create = () =>
			this.random.choice([-1, 1, 1]) *
			this.random.randrange(10 ** this.random.randrange(10));
		const [a, b, c, d] = [create(), create(), create(), create()];
		if (b === 0 || d === 0) return;
		const x = `${a}/${b}`;
		const y = `${c}/${d}`;
		const sol = SimplifyProductFraction.sol(x, y);
		const naive = SimplifyProductFraction.naive(x, y);
		if (sol.length < naive.length - 3) {
			this.add({ x, y, max_len: sol.length });
		}
	}
}

export class SortByDigitSum extends PuzzleGenerator {
	static docstring = "* Sort a list of numbers by the sum of their digits\n\n        Sample Input:\n        [1, 0, -1, -100, 10, 14, 235251, 11, 10000, 2000001, -155]\n\n        Sample Output:\n        [1, 0, -1, -100, 10, 14, 235251, 11, 10000, 2000001, -155]";
	/** Inspired by HumanEval #145 */
	static sat (
		ordered,
		nums = [1, 0, -1, -100, 10, 14, 235251, 11, 10000, 2000001, -155],
	) {
		const digitSum = (n) =>
			String(n)
				.replace(/-/, "")
				.split("")
				.reduce((sum, ch) => sum + Number(ch), 0);
		const digitSums = ordered.map((value) => digitSum(value));
		const sortedNums = [...nums].sort((a, b) => a - b);
		const sortedOrdered = [...ordered].sort((a, b) => a - b);
		const sortedDigitSums = [...digitSums].sort((a, b) => a - b);
		return (
			sortedNums.length === sortedOrdered.length &&
			sortedNums.every((value, index) => value === sortedOrdered[index]) &&
			digitSums.length === sortedDigitSums.length &&
			digitSums.every((value, index) => value === sortedDigitSums[index])
		);
	}

	static sol (nums) {
		const digitSum = (n) =>
			String(n)
				.replace(/-/, "")
				.split("")
				.reduce((sum, ch) => sum + Number(ch), 0);
		return [...nums].sort((a, b) => digitSum(a) - digitSum(b));
	}

	getExample () {
		return { nums: [1, 0, -1, -100, 10, 14, 235251, 11, 10000, 2000001, -155] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-1000, 1000),
		);
		this.add({ nums });
	}
}

export class BigOdds extends PuzzleGenerator {
	static docstring = "* Find numbers greater than 10 with odd first and last digits\n\n        Sample Input:\n        [204, 109, 203, 17, 45, 11, 21, 99, 909, 16, -33, 3, 17]\n\n        Sample Output:\n        [109, 203, 17, 45, 21, 99, 909]";
	/** Inspired by HumanEval #146 */
	static sat (
		odds,
		nums = [204, 109, 203, 17, 45, 11, 21, 99, 909, 16, -33, 3, 17],
	) {
		const checkOddDigits = (n) => {
			const digits = String(n);
			return (
				digits.length &&
				[digits[0], digits[digits.length - 1]].every((d) => Number(d) % 2 === 1)
			);
		};
		const counts = (list) => {
			const map = new Map();
			for (const value of list) map.set(value, (map.get(value) || 0) + 1);
			return map;
		};
		const numsMap = counts(nums);
		for (const o of odds) {
			if (o <= 10 || !checkOddDigits(o)) return false;
			if (numsMap.get(o) !== odds.filter((value) => value === o).length)
				return false;
		}
		return nums.every(
			(value) => odds.includes(value) || value <= 10 || !checkOddDigits(value),
		);
	}

	static sol (nums) {
		const checkOddDigits = (n) => {
			const digits = String(n);
			return (
				digits.length &&
				[digits[0], digits[digits.length - 1]].every((d) => Number(d) % 2 === 1)
			);
		};
		return nums.filter((n) => n > 10 && checkOddDigits(n));
	}

	getExample () {
		return { nums: [204, 109, 203, 17, 45, 11, 21, 99, 909, 16, -33, 3, 17] };
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(10) }, () =>
			this.random.randrange(-1000, 20000),
		);
		this.add({ nums });
	}
}

export class Threeples extends PuzzleGenerator {
	/** Inspired by HumanEval #147 */
	static docstring = "* Find all triples of increasing indices where the sum of the numbers is divisible by three\n\n        a=[1, 2, 4, 8, 14, 10], count=2 => [[0, 2, 5], [1, 3, 4]] = > because 1 + 4 + 10, 2 + 8 + 14 are divisible by 3";
	static sat (
		trips,
		a = [1, 0, -17, 42, 321, 36, 429, 35, 10, 923, 35, 18, 0, 17, 24, 32, 8],
		count = 221,
	) {
		const seen = new Set();
		for (const trip of trips) {
			seen.add(trip.join(","));
			const [i, j, k] = trip;
			if (!(0 <= i && i < j && j < k && (a[i] + a[j] + a[k]) % 3 === 0))
				return false;
		}
		return seen.size >= count;
	}

	static sol (a, count) {
		const triples = [];
		for (let k = 2; k < a.length; k++) {
			for (let j = 1; j < k; j++) {
				for (let i = 0; i < j; i++) {
					if ((a[i] + a[j] + a[k]) % 3 === 0) triples.push([i, j, k]);
				}
			}
		}
		return triples;
	}

	getExample () {
		const a = [
			1, 0, -17, 42, 321, 36, 429, 35, 10, 923, 35, 18, 0, 17, 24, 32, 8,
		];
		return { a, count: 221 };
	}

	genRandom () {
		const a = Array.from({ length: this.random.randrange(30) }, () =>
			this.random.randrange(-1, 10),
		);
		const count = Threeples.sol(a).length;
		this.add({ a, count });
	}
}

export class PlanetRange extends PuzzleGenerator {
	static docstring = "* Find the planets between two given planets in the solar system\n\n        Sample Input:\n        \"Mars\", \"Neptune\"\n\n        Sample Output:\n        [\"Jupiter\", \"Saturn\", \"Uranus\"]";
	/** Inspired by HumanEval #148 */
	static sat (planets_between, a = "Mars", b = "Neptune") {
		const joined = [a, ...planets_between, b].join(" ");
		return (
			!planets_between.join("").includes(" ") &&
			"Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto".includes(joined)
		);
	}

	static sol (a, b) {
		const planets =
			"Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto".split(" ");
		return planets.slice(planets.indexOf(a) + 1, planets.indexOf(b));
	}

	getExample () {
		return { a: "Mars", b: "Neptune" };
	}

	genRandom () {
		const planets =
			"Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto".split(" ");
		const i = this.random.randrange(1, planets.length);
		const j = this.random.randrange(i);
		this.add({ a: planets[j], b: planets[i] });
	}
}

export class EvenWords extends PuzzleGenerator {
	static docstring = "* Find even-length words from a list, sorted by length then alphabetically\n\n        Sample Input:\n        [\"The\", \"worm\", \"ate\", \"a\", \"bird\", \"imagine\", \"that\", \"!\", \"Absurd\", \"!!\"]\n\n        Sample Output:\n        [\"a\", \"!!\", \"ate\", \"bird\", \"that\", \"worm\", \"Absurd\", \"imagine\"]";
	/** Inspired by HumanEval #149 */
	static sat (
		evens,
		words = [
			"The",
			"worm",
			"ate",
			"a",
			"bird",
			"imagine",
			"that",
			"!",
			"Absurd",
			"!!",
		],
	) {
		for (let i = 0; i < evens.length; i++) {
			const word = evens[i];
			if (word.length % 2 !== 0) return false;
			if (!words.includes(word)) return false;
			if (i > 0 && evens[i].length < evens[i - 1].length) return false;
		}
		return words.every((word) => word.length % 2 === 1 || evens.includes(word));
	}

	static sol (words) {
		return words
			.filter((word) => word.length % 2 === 0)
			.sort((a, b) =>
				a.length === b.length ? a.localeCompare(b) : a.length - b.length,
			);
	}

	getExample () {
		return {
			words: [
				"The",
				"worm",
				"ate",
				"a",
				"bird",
				"imagine",
				"that",
				"!",
				"Absurd",
				"!!",
			],
		};
	}

	genRandom () {
		const words = Array.from({ length: this.random.randrange(1, 10) }, () =>
			this.random.pseudo_word(),
		);
		this.add({ words });
	}
}

export class PrimeSel extends PuzzleGenerator {
	/** Inspired by HumanEval #150 */
	static docstring = "* Find a list of all numbers that are adjacent to a prime number in the list, sorted without duplicates\n\n        [2, 17, 16, 0, 6, 4, 5] => [2, 4, 16, 17]";
	static sat (
		neighbors,
		nums = [
			14, 7, 11, 13, 7, 4, 19, 2, 55, 13, 31, 14, 2, 9, -7, 0, 88, 13, 13,
		],
	) {
		const primes = new Set();
		const isPrime = (m) =>
			m > 1 &&
			Array.from({ length: m - 2 }, (_, idx) => idx + 2).every((i) => m % i);
		for (let i = 0; i < nums.length; i++) {
			if (
				(i > 0 && isPrime(nums[i - 1])) ||
				(i < nums.length - 1 && isPrime(nums[i + 1]))
			) {
				primes.add(nums[i]);
			}
		}
		const neighborSet = new Set(neighbors);
		return (
			neighborSet.size === neighbors.length &&
			[...neighborSet].every((value) => primes.has(value)) &&
			[...neighborSet].every((value, index, array) =>
				index === array.length - 1 ? true : value <= array[index + 1],
			)
		);
	}

	static sol (nums) {
		const prime = (m) =>
			m > 1 &&
			Array.from({ length: m - 2 }, (_, idx) => idx + 2).every((i) => m % i);
		return [
			...new Set(
				nums.filter(
					(value, index) =>
						(index > 0 && prime(nums[index - 1])) ||
						(index < nums.length - 1 && prime(nums[index + 1])),
				),
			),
		].sort((a, b) => a - b);
	}

	getExample () {
		return {
			nums: [
				14, 7, 11, 13, 7, 4, 19, 2, 55, 13, 31, 14, 2, 9, -7, 0, 88, 13, 13,
			],
		};
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(30) }, () =>
			this.random.randrange(-1, 20),
		);
		this.add({ nums });
	}
}

export class EvenSqure extends PuzzleGenerator {
	static docstring = "* Sum the squares of even positive integers in a list\n\n        Sample Input:\n        [123.0, 872322.0, 542.2, -127.5, 18214.0, 3732.4, 12832.4, 23523800.0]\n\n        Sample Output:\n        553483464324";
	/** Inspired by HumanEval #151 */
	static sat (
		tot,
		xs = [123.0, 872322.0, 542.2, -127.5, 18214.0, 3732.4, 12832.4, 23523800.0],
	) {
		for (const x of xs) {
			if (Number.isInteger(x) && x > 0 && x % 2 === 0) {
				tot -= x ** 2;
			}
		}
		return tot === 0;
	}

	static sol (xs) {
		return xs
			.filter((x) => Number.isInteger(x) && x > 0 && x % 2 === 0)
			.reduce((sum, x) => sum + x ** 2, 0);
	}

	getExample () {
		return {
			xs: [
				123.0, 872322.0, 542.2, -127.5, 18214.0, 3732.4, 12832.4, 23523800.0,
			],
		};
	}

	genRandom () {
		const xs = Array.from(
			{ length: this.random.randrange(10) },
			() =>
				this.random.randrange(-100000, 300000) *
				this.random.choice([1.0, this.random.random()]),
		);
		if (EvenSqure.sol(xs) > 0 || this.random.randrange(10) === 0) {
			this.add({ xs });
		}
	}
}

export class ArrayDiff extends PuzzleGenerator {
	static docstring = "* Find the element-wise difference between two arrays\n\n        Sample Input:\n        [1, 2, 3, 0, 4, 17, 2, 4, 5, 9, 8, 4], [1, 2, 3, 4, 0, 16, 2, 3, 5, 9, 8, 4]\n\n        Sample Output:\n        [0, 0, 0, 4, -4, -1, 0, -1, 0, 0, 0, 0]";
	/** Inspired by HumanEval #152 */
	static sat (
		b,
		a = [1, 2, 3, 0, 4, 17, 2, 4, 5, 9, 8, 4],
		c = [1, 2, 3, 4, 0, 16, 2, 3, 5, 9, 8, 4],
	) {
		if (b.length !== a.length || a.length !== c.length) return false;
		return b.every((value, idx) => a[idx] + value === c[idx]);
	}

	static sol (a, c) {
		return a.map((value, idx) => c[idx] - value);
	}

	getExample () {
		return {
			a: [1, 2, 3, 0, 4, 17, 2, 4, 5, 9, 8, 4],
			c: [1, 2, 3, 4, 0, 16, 2, 3, 5, 9, 8, 4],
		};
	}

	genRandom () {
		const length = this.random.randrange(5, 30);
		const a = Array.from({ length }, () => this.random.randrange(-20, 20));
		const c = Array.from({ length }, () => this.random.randrange(-20, 20));
		this.add({ a, c });
	}
}

export class StrongestExtension extends PuzzleGenerator {
	/** Inspired by HumanEval #153 */
	static docstring = "* Find the class_name.extension for the extension that has the largest #capitals - #lowercase letters";
	static sat (
		s,
		class_name = "TestClass",
		extensions = ["extEnd", "LOL", "SuPeRbLy", "v9ACLQWTEW", "PickMe", "AI"],
	) {
		if (!s.startsWith(`${class_name}.`)) return false;
		const ext = s.slice(class_name.length + 1);
		const caseDelta = (value) =>
			Array.from(value).reduce((score, ch) => {
				if (/[A-Z]/.test(ch)) return score + 1;
				if (/[a-z]/.test(ch)) return score - 1;
				return score;
			}, 0);
		const deltas = extensions.map(caseDelta);
		const maxDelta = Math.max(...deltas);
		return extensions.includes(ext) && caseDelta(ext) === maxDelta;
	}

	static sol (class_name, extensions) {
		const caseDelta = (value) =>
			Array.from(value).reduce((score, ch) => {
				if (/[A-Z]/.test(ch)) return score + 1;
				if (/[a-z]/.test(ch)) return score - 1;
				return score;
			}, 0);
		const best = extensions.reduce((winner, extension) =>
			caseDelta(extension) > caseDelta(winner) ? extension : winner,
		);
		return `${class_name}.${best}`;
	}

	getExample () {
		return {
			class_name: "TestClass",
			extensions: ["extEnd", "LOL", "SuPeRbLy", "v9ACLQWTEW", "PickMe", "AI"],
		};
	}

	genRandom () {
		let class_name = this.random.pseudo_word();
		class_name = class_name.charAt(0).toUpperCase() + class_name.slice(1);
		const extensions = Array.from(
			{ length: this.random.randrange(5, 15) },
			() => randomCaseWord(this.random),
		);
		this.add({ class_name, extensions });
	}
}

export class RotateString extends PuzzleGenerator {
	static docstring = "* Find a rotation of string s that appears in string t\n\n        Sample Input:\n        \"light star\", \"I love to look at the starlight!\"\n\n        Sample Output:\n        \"starlight\"";
	/** Inspired by HumanEval #154 */
	static sat (r, s = "light star", t = "I love to look at the starlight!") {
		return r.length === s.length && (s + s).includes(r) && t.includes(r);
	}

	static sol (s, t) {
		for (let i = 0; i < s.length; i++) {
			const candidate = s.slice(i) + s.slice(0, i);
			if (t.includes(candidate)) return candidate;
		}
		return "";
	}

	getExample () {
		return { s: "light star", t: "I love to look at the starlight!" };
	}

	genRandom () {
		const t = Array.from({ length: 10 }, () => this.random.pseudo_word()).join(
			" ",
		);
		if (!t.length) return;
		const start = this.random.randrange(0, t.length - 1);
		const end = this.random.randrange(start + 1, t.length + 1);
		const r = t.slice(start, end);
		const i = this.random.randrange(r.length);
		const s = r.slice(i) + r.slice(0, i);
		this.add({ s, t });
	}
}

export class EvenOddDigits extends PuzzleGenerator {
	static docstring = "* Find an integer n >= 0 with the given number of even and odd digits.\n\n        evens=3, odds=4 => 2381695";
	/** Inspired by HumanEval #155 */
	static sat (n, evens = 17, odds = 3) {
		let evenCount = evens;
		let oddCount = odds;
		for (const c of String(n)) {
			if (Number(c) % 2 === 0) evenCount -= 1;
			else oddCount -= 1;
		}
		return evenCount === 0 && oddCount === 0;
	}

	static sol (evens, odds) {
		return "2".repeat(evens) + "1".repeat(odds);
	}

	getExample () {
		return { evens: 17, odds: 3 };
	}

	genRandom () {
		const evens = this.random.randrange(150);
		const odds = this.random.randrange(150);
		if (evens + odds > 0) this.add({ evens, odds });
	}
}

export class RomanNumerals extends PuzzleGenerator {
	static docstring = "* Convert a number to its Roman numeral representation\n\n        Sample Input:\n        2414\n\n        Sample Output:\n        \"MMCDXIV\"";
	/** Inspired by HumanEval #156 */
	static sat (roman, n = 2414) {
		const key = {
			1000: "m",
			900: "cm",
			500: "d",
			400: "cd",
			100: "c",
			90: "xc",
			50: "l",
			40: "xl",
			10: "x",
			9: "ix",
			5: "v",
			4: "iv",
			1: "i",
		};
		let m = 0;
		let str = roman.toLowerCase();
		for (const base of [1000, 100, 10, 1]) {
			for (const mul of [9, 4, 5, 1, 1, 1]) {
				const val = base * mul;
				if (val in key && str.startsWith(key[val])) {
					m += val;
					str = str.slice(key[val].length);
					if (mul === 9 || mul === 4) break; // 9 or 4 can't be followed by anything else
				}
			}
		}
		return m === n;
	}

	static sol (n) {
		const units = {
			m: 1000,
			cm: 900,
			d: 500,
			cd: 400,
			c: 100,
			xc: 90,
			l: 50,
			xl: 40,
			x: 10,
			ix: 9,
			v: 5,
			iv: 4,
			i: 1,
		};
		let roman = "";
		for (const symbol of Object.keys(units)) {
			while (n >= units[symbol]) {
				roman += symbol;
				n -= units[symbol];
			}
		}
		return roman;
	}

	getExample () {
		return { n: 2414 };
	}

	genRandom () {
		const n = this.random.randrange(1, 4000);
		this.add({ n });
	}
}

export class PythagoreanTriples extends PuzzleGenerator {
	/** Inspired by HumanEval #157 */
	static docstring = "* Find m Pythagorean triples a^2 + b^2 == c^2 for integers 0 < a < b < c <= n, in sorted order\n\n        (n=6, m=1) => [[3, 4, 5]]";
	static _cache = new Map();

	static sat (triples, n = 920, m = 799) {
		if (triples.length < m) return false;
		for (const [a, b, c] of triples) {
			if (!(0 < a && a < b && b < c && c <= n)) return false;
			if (a * a + b * b !== c * c) return false;
		}
		const sorted = [...triples].sort((p, q) => {
			for (let i = 0; i < 3; i++) {
				if (p[i] !== q[i]) return p[i] - q[i];
			}
			return 0;
		});
		return JSON.stringify(triples) === JSON.stringify(sorted);
	}

	static sol (n, m) {
		const triples = [];
		for (let a = 3; a < Math.floor(n / Math.SQRT2); a++) {
			const maxB = Math.floor(Math.sqrt(n * n - a * a));
			for (let b = a + 1; b <= maxB; b++) {
				const c_squared = a * a + b * b;
				const c = Math.sqrt(c_squared);
				// Use int() truncation like Python does, then check if c^2 matches
				const c_int = Math.floor(c);
				if (c_int * c_int === c_squared) {
					triples.push([a, b, c_int]);
				}
			}
		}
		return triples;
	}

	getExample () {
		return { n: 920, m: 799 };
	}

	genRandom () {
		const n = this.random.randrange(1, 1000);
		if (!PythagoreanTriples._cache.has(n)) {
			PythagoreanTriples._cache.set(n, PythagoreanTriples.sol(n, null).length);
		}
		this.add({ n, m: PythagoreanTriples._cache.get(n) });
	}
}

export class MostUnique extends PuzzleGenerator {
	static docstring = "* Select a string from the pool with the most unique characters\n\n        [\"woooow\", \"cow\"] => \"cow\"";
	/** Inspired by HumanEval #158 */
	static sat (
		s,
		pool = [
			"cat",
			"catatatatctsa",
			"abcdefhijklmnop",
			"124259239185125",
			"",
			"foo",
			"unique",
		],
	) {
		if (!pool.includes(s)) return false;
		const uniqueCount = (value) => new Set(value).size;
		const maxUnique = Math.max(...pool.map(uniqueCount));
		return uniqueCount(s) === maxUnique;
	}

	static sol (pool) {
		return pool.reduce((winner, word) =>
			new Set(word).size > new Set(winner).size ? word : winner,
		);
	}

	getExample () {
		return {
			pool: [
				"cat",
				"catatatatctsa",
				"abcdefhijklmnop",
				"124259239185125",
				"",
				"foo",
				"unique",
			],
		};
	}

	genRandom () {
		const pool = Array.from({ length: this.random.randrange(2, 10) }, () =>
			this.random.pseudo_word(0, 5),
		);
		this.add({ pool });
	}
}

export class HungryRabbits extends PuzzleGenerator {
	static docstring = "* For each triple of eaten, need, stock return a pair of total appetite and remaining\n\n        [[2, 5, 6], [3, 9, 22]] => [[7, 1], [12, 13]]";
	/** Inspired by HumanEval #159 */
	static sat (
		results,
		stats = [
			[2, 3, 18],
			[4, 9, 2],
			[2, 5, 7],
			[3, 8, 12],
			[4, 9, 106],
		],
	) {
		if (results.length !== stats.length) return false;
		for (let i = 0; i < stats.length; i++) {
			const [tot, remaining] = results[i];
			const [eaten, need, stock] = stats[i];
			if (tot - eaten !== Math.min(need, stock)) return false;
			if (stock < need && remaining !== 0) return false;
			if (stock >= need && remaining + need !== stock) return false;
		}
		return true;
	}

	static sol (stats) {
		return stats.map(([eaten, need, stock]) => [
			eaten + Math.min(need, stock),
			Math.max(0, stock - need),
		]);
	}

	getExample () {
		return {
			stats: [
				[2, 3, 18],
				[4, 9, 2],
				[2, 5, 7],
				[3, 8, 12],
				[4, 9, 106],
			],
		};
	}

	genRandom () {
		const stats = Array.from({ length: this.random.randrange(1, 10) }, () =>
			Array.from({ length: 3 }, () => this.random.randrange(10)),
		);
		this.add({ stats });
	}
}

export class EvaluateOperators extends PuzzleGenerator {
	static docstring = "* Find a permutation of the operators +-*/^% which when inserted between nums evaluates to target\n\n        target=3, nums=[7, 2, 3, 4, 5, 1, 6] => [\"+\", \"*\", \"**\", \"%\", \"//\", \"-\"]\n                                                # because 7 + 2 * 3 ** 4 % 5 // 1 - 6 == 3";
	/** Inspired by HumanEval #160 */
	static sat (ops, target = 2021, nums = [4, 6, 2, 1, 1, 3, 9]) {
		const allowed = ["**", "*", "+", "-", "//", "%"];
		if (ops.length !== allowed.length) return false;
		const unique = new Set(ops);
		if (unique.size !== allowed.length) return false;
		if (!allowed.every((op) => ops.includes(op))) return false;
		if (nums.length !== ops.length + 1) return false;
		const value = evaluateExpression(nums, ops);
		return value === target;
	}

	static sol (target, nums) {
		const allowed = ["**", "*", "+", "-", "//", "%"];
		for (const permutation of permute(allowed)) {
			if (evaluateExpression(nums, permutation) === target) return permutation;
		}
		throw new Error("No permutation produces the target");
	}

	getExample () {
		return { target: 2021, nums: [4, 6, 2, 1, 1, 3, 9] };
	}

	genRandom () {
		const ops = ["**", "*", "+", "-", "//", "%"];
		this.random.shuffle(ops);
		const nums = Array.from({ length: ops.length + 1 }, () =>
			this.random.randrange(1, 10),
		);
		try {
			const target = evaluateExpression(nums, ops);
			this.add({ target, nums });
		} catch (error) {
			// skip invalid sequences (division by zero)
		}
	}
}

function isPrime (n) {
	if (n <= 1) return false;
	if (n <= 3) return true;
	for (let i = 2; i <= Math.floor(Math.sqrt(n)); i++) {
		if (n % i === 0) return false;
	}
	return true;
}

function gcd (a, b) {
	if (b === 0) return a;
	return gcd(b, a % b);
}

function randomCaseWord (random) {
	const letters = "abcdefghijklmnopqrstuvwxyz".split("");
	const length = random.randrange(1, 8);
	let word = "";
	for (let i = 0; i < length; i++) {
		const letter = letters[random.randrange(letters.length)];
		word += random.choice([letter, letter.toUpperCase()]);
	}
	return word;
}

function permute (arr) {
	const results = [];
	const current = [];
	const used = Array(arr.length).fill(false);
	function backtrack () {
		if (current.length === arr.length) {
			results.push([...current]);
			return;
		}
		for (let i = 0; i < arr.length; i++) {
			if (used[i]) continue;
			used[i] = true;
			current.push(arr[i]);
			backtrack();
			current.pop();
			used[i] = false;
		}
	}
	backtrack();
	return results;
}

function precedence (op) {
	if (op === "**") return 4;
	if (["*", "/", "//", "%"].includes(op)) return 3;
	return 2;
}

function isRightAssociative (op) {
	return op === "**";
}

function applyOperator (op, left, right) {
	switch (op) {
		case "+":
			return left + right;
		case "-":
			return left - right;
		case "*":
			return left * right;
		case "/":
			if (right === 0) throw new Error("division by zero");
			return left / right;
		case "//":
			return floorDiv(left, right);
		case "%": {
			if (right === 0) throw new Error("division by zero");
			const remainder = left % right;
			return ((remainder % right) + right) % right;
		}
		case "**":
			return left ** right;
		default:
			throw new Error(`Unknown operator: ${op}`);
	}
}

function floorDiv (left, right) {
	if (right === 0) throw new Error("division by zero");
	return Math.floor(left / right);
}

function evaluateExpression (nums, ops) {
	const values = [nums[0]];
	const operators = [];
	const shouldApply = (topOp, incomingOp) => {
		if (isRightAssociative(incomingOp)) {
			return precedence(topOp) > precedence(incomingOp);
		}
		return precedence(topOp) >= precedence(incomingOp);
	};
	for (let i = 0; i < ops.length; i++) {
		const currentOp = ops[i];
		const nextValue = nums[i + 1];
		while (
			operators.length &&
			shouldApply(operators[operators.length - 1], currentOp)
		) {
			const op = operators.pop();
			const right = values.pop();
			const left = values.pop();
			values.push(applyOperator(op, left, right));
		}
		operators.push(currentOp);
		values.push(nextValue);
	}
	while (operators.length) {
		const op = operators.pop();
		const right = values.pop();
		const left = values.pop();
		values.push(applyOperator(op, left, right));
	}
	return values[0];
}

export class ZobristCollision extends PuzzleGenerator {
	static docstring = "* Find a collision for the given Zobrist chess board hash: https://en.wikipedia.org/wiki/Zobrist_hashing\n\n        Each of the two positions should be encoded as a list of 64 integers 0-12";
	/** Inspired by HumanEval #162 */
	static sat (positions) {
		/**
		 * Find a collision for the given Zobrist chess board hash: https://en.wikipedia.org/wiki/Zobrist_hashing
		 *
		 * Each of the two positions should be encoded as a list of 64 integers 0-12
		 */
		const table = Array.from({ length: 8 }, (_, i) =>
			Array.from(
				{ length: 13 },
				(_, j) => (i * 429436219 + j * 100239120) % 63491564,
			),
		);

		const zobrist = (pos) => {
			let h = 0;
			for (let i = 0; i < 8; i++) {
				if (pos[i]) {
					h ^= table[i][pos[i]];
				}
			}
			return h % 100000;
		};

		const [a, b] = positions;
		return zobrist(a) === zobrist(b) && JSON.stringify(a) !== JSON.stringify(b);
	}

	static sol () {
		const hashes = {};
		const table = Array.from({ length: 8 }, (_, i) =>
			Array.from(
				{ length: 13 },
				(_, j) => (i * 429436219 + j * 100239120) % 63491564,
			),
		);

		const zobrist = (pos) => {
			let h = 0;
			for (let i = 0; i < 8; i++) {
				if (pos[i]) {
					h ^= table[i][pos[i]];
				}
			}
			return h % 100000;
		};

		for (let i = 0; i < 1000; i++) {
			const pos = Array.from({ length: 8 }, () =>
				Math.floor(Math.random() * 13),
			);
			const h = zobrist(pos);
			if (hashes[h]) {
				return [pos, hashes[h]];
			} else {
				hashes[h] = pos;
			}
		}
	}

	getExample () {
		return {};
	}

	genRandom () {
		// Since sol() finds a collision, we can just use the example
		this.add({});
	}
}

export class EvenBetween extends PuzzleGenerator {
	static docstring = "* Find integers [a, b] that are at least 5 apart and such that concatenating the even numbers\n        between them gives the string s\n\n        \"32343638\" => [31, 38]";
	/** Inspired by HumanEval #163 */
	static sat (ab, s = "3298832990329923299432996329983300033002") {
		/**
		 * Find integers [a, b] that are at least 5 apart and such that concatenating the even numbers
		 * between them gives the string s
		 *
		 * "32343638" => [31, 38]
		 */
		const [a, b] = ab;
		if (Math.abs(a - b) <= 4) return false;
		const min = Math.min(a, b);
		const max = Math.max(a, b);
		const evens = [];
		for (let i = min; i <= max; i++) {
			if (i % 2 === 0) evens.push(i.toString());
		}
		return s === evens.join("");
	}

	static sol (s) {
		for (let i = 1; i <= s.length; i++) {
			let n = parseInt(s.slice(0, i));
			n -= (n + 1) % 2; // make n odd
			let m = n + 1; // next even
			let t = "";
			while (t.length < s.length) {
				t += m.toString();
				m += 2;
			}
			if (s === t) {
				return [n, m - 1];
			}
		}
		throw new Error("No solution found");
	}

	getExample () {
		return {
			s: "3298832990329923299432996329983300033002",
		};
	}

	genRandom () {
		const a = Math.floor(Math.random() * (100000 - 100)) + 100;
		const b = a + Math.floor(Math.random() * 17) + 5;
		let s = "";
		for (let i = a; i <= b; i++) {
			if (i % 2 === 0) s += i.toString();
		}
		this.add({ s });
	}
}
