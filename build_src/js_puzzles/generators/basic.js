/**
 * Problems testing basic knowledge -- easy to solve if you understand what is being asked
 */

import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

export class SumOfDigits extends PuzzleGenerator {
	static tags = [Tags.math];
	static docstring = "Find a number that its digits sum to a specific value.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math];
	}

	getExample () {
		return { s: 679 };
	}
	/**
	 * Find a number that its digits sum to a specific value.
	 */
	static sat (answer, s) {
		if (typeof answer !== "string") return false;
		const sum = answer.split("").reduce((acc, ch) => acc + Number(ch), 0);
		return sum === s;
	}

	static sol (s) {
		const nines = Math.floor(s / 9);
		return "9".repeat(nines) + String(s % 9);
	}

	genRandom () {
		const r = this.random();
		const s = Math.floor(r * 100000);
		this.add({ s });
	}
}

export class FloatWithDecimalValue extends PuzzleGenerator {
	static tags = [Tags.math];
	static docstring = "Create a float with a specific decimal.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math];
	}

	getExample () {
		return { v: 9, d: 0.0001 };
	}

	/**
	 * Create a float with a specific decimal.
	 */
	static sat (z, v, d) {
		const val = Math.floor((z * (1 / d)) % 10);
		return val === v;
	}

	/**
	 * Reference solution for FloatWithDecimalValue.
	 */
	static sol (v, d) {
		return v * d;
	}

	genRandom () {
		const v = Math.floor(this.random() * 10);
		const a = Math.floor(this.random() * 11) - 5; // exponent between -5 and 5
		const d = 10 ** a;
		// verify solver works numerically; if not, skip
		const z = v * d;
		if (Math.floor((z * (1 / d)) % 10) !== v) return;
		this.add({ v, d });
	}
}

export class LineIntersection extends PuzzleGenerator {
	static tags = [Tags.math];
	static docstring = "Find the intersection of two lines.\n        Solution should be a list of the (x,y) coordinates.\n        Accuracy of fifth decimal digit is required.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math];
	}

	getExample () {
		return { a: 2, b: -1, c: 1, d: 2021 };
	}

	/**
	 * Find the intersection of two lines.
	 * Solution should be a list of the (x,y) coordinates.
	 * Accuracy of fifth decimal digit is required.
	 */
	static sat (e, a, b, c, d) {
		if (!Array.isArray(e) || e.length < 2) return false;
		const x = e[0] / e[1];
		return Math.abs(a * x + b - c * x - d) < 1e-5;
	}

	/**
	 * Reference solution for LineIntersection.
	 */
	static sol (a, b, c, d) {
		return [d - b, a - c];
	}

	genRandom () {
		const a = Math.floor(this.random() * 200001) - 100000;
		let c = a;
		while (c === a) c = Math.floor(this.random() * 200001) - 100000;
		const b = Math.floor(this.random() * 200001) - 100000;
		const d = Math.floor(this.random() * 200001) - 100000;
		this.add({ a, b, c, d });
	}
}

export class IfProblem extends PuzzleGenerator {
	static tags = [Tags.trivial];
	static docstring = "Satisfy a simple if statement";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial];
	}

	getExample () {
		return { a: 324554, b: 1345345 };
	}

	/**
	 * Satisfy a simple if statement.
	 */
	static sat (x, a, b) {
		if (a < 50) return x + a === b;
		return x - 2 * a === b;
	}

	/**
	 * Reference solution for IfProblem.
	 */
	static sol (a, b) {
		if (a < 50) return b - a;
		return b + 2 * a;
	}

	genRandom () {
		const a = Math.floor(this.random() * 101);
		const b = Math.floor((this.random() - 0.5) * 2 * 1e8);
		this.add({ a, b });
	}
}

export class IfProblemWithAnd extends PuzzleGenerator {
	static tags = [Tags.trivial];
	static docstring = "Satisfy a simple if statement with an and clause";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial];
	}

	getExample () {
		return { a: 9384594, b: 1343663 };
	}

	/**
	 * Satisfy a simple if statement with an and clause.
	 */
	static sat (x, a, b) {
		if (x > 0 && a > 50) return x - a === b;
		return x + a === b;
	}

	/**
	 * Reference solution for IfProblemWithAnd.
	 */
	static sol (a, b) {
		if (a > 50 && b > a) return b + a;
		return b - a;
	}

	genRandom () {
		const a = Math.floor(this.random() * 101);
		const b = Math.floor(this.random() * 1e8);
		this.add({ a, b });
	}
}

export class IfProblemWithOr extends PuzzleGenerator {
	static tags = [Tags.trivial];
	static docstring = "Satisfy a simple if statement with an or clause";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial];
	}

	getExample () {
		return { a: 253532, b: 1230200 };
	}

	/**
	 * Satisfy a simple if statement with an or clause.
	 */
	static sat (x, a, b) {
		if (x > 0 || a > 50) return x - a === b;
		return x + a === b;
	}

	/**
	 * Reference solution for IfProblemWithOr.
	 */
	static sol (a, b) {
		if (a > 50 || b > a) return b + a;
		return b - a;
	}

	genRandom () {
		const a = Math.floor(this.random() * 101);
		const b = Math.floor((this.random() - 0.5) * 2 * 1e8);
		this.add({ a, b });
	}
}

export class IfCases extends PuzzleGenerator {
	static tags = [Tags.trivial];
	static docstring = "Satisfy a simple if statement with multiple cases";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial];
	}

	getExample () {
		return { a: 4, b: 54368639 };
	}

	/**
	 * Satisfy a simple if statement with multiple cases.
	 */
	static sat (x, a, b) {
		if (a === 1) return x % 2 === 0;
		if (a === -1) return x % 2 === 1;
		return x + a === b;
	}

	/**
	 * Reference solution for IfCases.
	 */
	static sol (a, b) {
		if (a === 1) return 0;
		if (a === -1) return 1;
		return b - a;
	}

	genRandom () {
		const a = Math.floor(this.random() * 11) - 5;
		const b = Math.floor((this.random() - 0.5) * 2 * 1e8);
		this.add({ a, b });
	}
}

export class ListPosSum extends PuzzleGenerator {
	static tags = [Tags.trivial];
	static docstring = "Find a list of n non-negative integers that sum up to s";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial];
	}

	getExample () {
		return { n: 5, s: 19 };
	}

	/**
	 * Find a list of n non-negative integers that sum up to s.
	 */
	static sat (x, n, s) {
		if (!Array.isArray(x) || x.length !== n) return false;
		if (x.reduce((a, b) => a + b, 0) !== s) return false;
		return x.every((a) => a > 0);
	}

	/**
	 * Reference solution for ListPosSum.
	 */
	static sol (n, s) {
		const x = [1];
		for (let i = 0; i < n - 1; i++) x.push(1);
		x[0] = s - n + 1;
		return x;
	}

	genRandom () {
		const n = Math.floor(this.random() * 1e4) + 1;
		const s = Math.floor(this.random() * 1e8) + n;
		this.add({ n, s });
	}
}

export class ListDistinctSum extends PuzzleGenerator {
	static tags = [Tags.math];
	static docstring = "Construct a list of n distinct integers that sum up to s";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math];
	}

	getExample () {
		return { n: 4, s: 2021 };
	}

	/**
	 * Construct a list of n distinct integers that sum up to s.
	 */
	static sat (x, n, s) {
		if (!Array.isArray(x) || x.length !== n) return false;
		if (x.reduce((a, b) => a + b, 0) !== s) return false;
		return new Set(x).size === n;
	}

	/**
	 * Reference solution for ListDistinctSum.
	 */
	static sol (n, s) {
		let a = 1;
		const x = [];
		while (x.length < n - 1) {
			x.push(a);
			a = -a;
			if (x.includes(a)) a += 1;
		}
		if (x.includes(s - x.reduce((a, b) => a + b, 0))) {
			x.length = 0;
			for (let i = 0; i < n - 1; i++) x.push(i);
		}
		x.push(s - x.reduce((a, b) => a + b, 0));
		return x;
	}

	genRandom () {
		const n = Math.floor(this.random() * 1000) + 1;
		const s = Math.floor(this.random() * 1e8) + n + 1;
		this.add({ n, s });
	}
}

export class BasicStrCounts extends PuzzleGenerator {
	static tags = [Tags.strings];
	static docstring = "Find a string that has count1 occurrences of s1 and count2 occurrences of s2 and starts and ends with\nthe same 10 characters";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.strings];
	}

	getExample () {
		return { s1: "a", s2: "b", count1: 50, count2: 30 };
	}

	/**
	 * Find a string that has count1 occurrences of s1 and count2 occurrences of s2 and starts and ends with
	 * the same 10 characters.
	 */
	static sat (str, s1, s2, count1, count2) {
		return (
			str.split(s1).length - 1 === count1 &&
			str.split(s2).length - 1 === count2 &&
			str.slice(0, 10) === str.slice(-10)
		);
	}

	/**
	 * Reference solution for BasicStrCounts.
	 */
	static sol (s1, s2, count1, count2) {
		let ans = "";
		if (s1 === s2) {
			ans = (s1 + "?").repeat(count1);
		} else if (s1.includes(s2)) {
			ans = (s1 + "?").repeat(count1);
			while (ans.split(s2).length - 1 < count2) ans += s2 + "?";
		} else {
			ans = (s2 + "?").repeat(count2);
			while (ans.split(s1).length - 1 < count1) ans += s1 + "?";
		}
		return "?".repeat(10) + ans + "?".repeat(10);
	}

	genRandom () {
		const s1 = this.randomChoiceWords(3);
		const s2 = this.randomChoiceWords(3);
		const count1 = Math.floor(this.random() * 100);
		const count2 = Math.floor(this.random() * 100);
		if (
			this.constructor.sat(
				this.constructor.sol(s1, s2, count1, count2),
				s1,
				s2,
				count1,
				count2,
			)
		) {
			this.add({ s1, s2, count1, count2 });
		}
	}

	randomChoiceWords (maxLen) {
		const len = Math.floor(this.random() * maxLen) + 1;
		const chars = "abcdefghijklmnopqrstuvwxyz";
		let w = "";
		for (let i = 0; i < len; i++)
			w += chars[Math.floor(this.random() * chars.length)];
		return w;
	}
}

export class ZipStr extends PuzzleGenerator {
	static tags = [Tags.strings, Tags.trivial];
	static docstring = "Find a string that contains each string in substrings alternating, e.g., 'cdaotg' for 'cat' and 'dog'";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.strings, Tags.trivial];
	}

	getExample () {
		return { substrings: ["foo", "bar", "baz", "oddball"] };
	}

	/**
	 * Find a string that contains each string in substrings alternating, e.g., 'cdaotg' for 'cat' and 'dog'.
	 */
	static sat (s, substrings) {
		return substrings.every((sub, i) =>
			s
				.split("")
				.filter((_, idx) => idx % substrings.length === i)
				.join("")
				.includes(sub),
		);
	}

	/**
	 * Reference solution for ZipStr.
	 */
	static sol (substrings) {
		const m = Math.max(...substrings.map((x) => x.length));
		let out = "";
		for (let i = 0; i < m; i++)
			for (const s of substrings) out += i < s.length ? s[i] : " ";
		return out;
	}

	genRandom () {
		const count = Math.floor(this.random() * 4) + 1;
		const substrings = [];
		for (let i = 0; i < count; i++) substrings.push(this.randomChoiceWords(6));
		this.add({ substrings });
	}
}

export class ReverseCat extends PuzzleGenerator {
	static tags = [Tags.trivial, Tags.strings];
	static docstring = "Find a string that contains all the substrings reversed and forward";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial, Tags.strings];
	}

	getExample () {
		return { substrings: ["foo", "bar", "baz"] };
	}

	/**
	 * Find a string that contains all the substrings reversed and forward.
	 */
	static sat (s, substrings) {
		return substrings.every(
			(sub) => s.includes(sub) && s.includes(sub.split("").reverse().join("")),
		);
	}

	/**
	 * Reference solution for ReverseCat.
	 */
	static sol (substrings) {
		return (
			substrings.join("") +
			substrings.map((s) => s.split("").reverse().join("")).join("")
		);
	}

	genRandom () {
		const count = Math.floor(this.random() * 4) + 1;
		const substrings = [];
		for (let i = 0; i < count; i++) substrings.push(this.randomChoiceWords(6));
		this.add({ substrings });
	}
}

export class SubstrCount extends PuzzleGenerator {
	static tags = [Tags.brute_force, Tags.strings];
	static docstring = "Find a substring with a certain count in a given string";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.brute_force, Tags.strings];
	}

	getExample () {
		return { string: "moooboooofasd", count: 2 };
	}

	/**
	 * Find a substring with a certain count in a given string.
	 */
	static sat (substring, string, count) {
		return string.split(substring).length - 1 === count;
	}

	/**
	 * Reference solution for SubstrCount.
	 */
	static sol (string, count) {
		for (let i = 0; i < string.length; i++) {
			for (let j = i + 1; j <= string.length; j++) {
				const substring = string.slice(i, j);
				const c = string.split(substring).length - 1;
				if (c === count) return substring;
				if (c < count) break;
			}
		}
		throw new Error("No substring found");
	}

	genRandom () {
		const string = this.randomChoiceWords(20);
		const substrings = [];
		for (let i = 0; i < string.length; i++)
			for (let j = i + 2; j <= string.length; j++)
				if (string.split(string.slice(i, j)).length - 1 > 2)
					substrings.push(string.slice(i, j));
		if (!substrings.length) return;
		const substring = substrings[Math.floor(this.random() * substrings.length)];
		const count = string.split(substring).length - 1;
		this.add({ string, count });
	}
}

export class SublistSum extends PuzzleGenerator {
	static tags = [Tags.math];
	static docstring = "Sum values of sublist by range specifications";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math];
	}

	getExample () {
		return { t: 677, a: 43, e: 125, s: 10 };
	}

	/**
	 * Sum values of sublist by range specifications.
	 */
	static sat (x, t, a, e, s) {
		if (!Array.isArray(x)) return false;
		const nonZero = x.filter((z) => z !== 0);
		const sumRange = (() => {
			let sum = 0;
			for (let i = a; i < e; i += s) sum += x[i] || 0;
			return sum;
		})();
		return (
			t === sumRange &&
			new Set(nonZero).size === nonZero.length &&
			(() => {
				for (let i = a; i < e; i += s) {
					if (!x[i]) return false;
				}
				return true;
			})()
		);
	}

	/**
	 * Reference solution for SublistSum.
	 */
	static sol (t, a, e, s) {
		const x = new Array(e).fill(0);
		for (let i = a; i < e; i += s) x[i] = i;
		const i = (() => {
			let last;
			for (let j = a; j < e; j += s) last = j;
			return last;
		})();
		const correction = t - x.reduce((acc, v) => acc + v, 0) + x[i];
		if (x.includes(correction)) {
			const idx = x.indexOf(correction);
			x[idx] = -1 * correction;
			x[i] = 3 * correction;
		} else {
			x[i] = correction;
		}
		return x;
	}

	genRandom () {
		const t = Math.floor(this.random() * 1e8) + 1;
		const a = Math.floor(this.random() * 100) + 1;
		const e = a + Math.floor(this.random() * 10000);
		const s = Math.floor(this.random() * 10) + 1;
		this.add({ t, a, e, s });
	}
}

export class CumulativeSum extends PuzzleGenerator {
	static tags = [Tags.math, Tags.trivial];
	static docstring = "Find how many values have cumulative sum less than target";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math, Tags.trivial];
	}

	getExample () {
		return { t: 50, n: 10 };
	}

	/**
	 * Find how many values have cumulative sum less than target.
	 */
	static sat (x, t, n) {
		if (!Array.isArray(x) || x.some((v) => v <= 0)) return false;
		let s = 0,
			i = 0;
		for (const v of x.slice().sort((a, b) => a - b)) {
			s += v;
			if (s > t) return i === n;
			i += 1;
		}
		return i === n;
	}

	/**
	 * Reference solution for CumulativeSum.
	 */
	static sol (t, n) {
		return new Array(n).fill(1).concat([t]);
	}

	genRandom () {
		const n = Math.floor(this.random() * 10000) + 1;
		const t = Math.floor(this.random() * 1e10) + n;
		this.add({ t, n });
	}
}

export class EngineerNumbers extends PuzzleGenerator {
	static tags = [Tags.trivial, Tags.strings];
	static docstring = "Find a list of n strings, in alphabetical order, starting with a and ending with b.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial, Tags.strings];
	}

	getExample () {
		return { n: 100, a: "bar", b: "foo" };
	}

	/**
	 * Find a list of n strings, in alphabetical order, starting with a and ending with b.
	 */
	static sat (ls, n, a, b) {
		if (!Array.isArray(ls) || ls.length !== n) return false;
		const set = new Set(ls);
		return (
			set.size === n &&
			ls[0] === a &&
			ls[ls.length - 1] === b &&
			ls.every((v, i, arr) => i === 0 || arr[i - 1] <= v)
		);
	}

	/**
	 * Reference solution for EngineerNumbers.
	 */
	static sol (n, a, b) {
		const arr = [a];
		for (let i = 0; i < n - 2; i++)
			arr.push(a + String.fromCharCode(0) + String(i));
		arr.push(b);
		return arr.sort();
	}

	genRandom () {
		const words = [this.pseudoWord(), this.pseudoWord()];
		const [a, b] = words.sort();
		const n = Math.floor(this.random() * 98) + 2;
		if (a !== b) this.add({ n, a, b });
	}
}

export class PenultimateString extends PuzzleGenerator {
	static tags = [Tags.trivial, Tags.strings];
	static docstring = "Find the alphabetically second to last last string in a list.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial, Tags.strings];
	}

	getExample () {
		return { strings: ["cat", "dog", "bird", "fly", "moose"] };
	}

	/**
	 * Find the alphabetically second to last last string in a list.
	 */
	static sat (s, strings) {
		return strings.includes(s) && strings.filter((t) => t > s).length === 1;
	}

	/**
	 * Reference solution for PenultimateString.
	 */
	static sol (strings) {
		return strings.slice().sort().slice(-2, -1)[0];
	}

	genRandom () {
		const strings = Array.from({ length: 10 }, () => this.randomChoiceWords(6));
		this.add({ strings });
	}
}

export class PenultimateRevString extends PuzzleGenerator {
	static tags = [Tags.trivial, Tags.strings];
	static docstring = "Find the reversed version of the alphabetically second string in a list.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial, Tags.strings];
	}

	getExample () {
		return { strings: ["cat", "dog", "bird", "fly", "moose"] };
	}

	/**
	 * Find the reversed version of the alphabetically second string in a list.
	 */
	static sat (s, strings) {
		return (
			strings.includes(s.split("").reverse().join("")) &&
			strings.filter((t) => t < s.split("").reverse().join("")).length === 1
		);
	}

	/**
	 * Reference solution for PenultimateRevString.
	 */
	static sol (strings) {
		return strings.slice().sort()[1].split("").reverse().join("");
	}

	genRandom () {
		const strings = Array.from({ length: 10 }, () => this.randomChoiceWords(6));
		this.add({ strings });
	}
}

export class CenteredString extends PuzzleGenerator {
	static tags = [Tags.trivial, Tags.strings];
	static docstring = "Find a substring of the given length centered within the target string.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.trivial, Tags.strings];
	}

	getExample () {
		return { target: "foobarbazwow", length: 6 };
	}

	/**
	 * Find a substring of the given length centered within the target string.
	 */
	static sat (s, target, length) {
		const start = Math.floor((target.length - length) / 2);
		return target.slice(start, start + length) === s;
	}

	/**
	 * Reference solution for CenteredString.
	 */
	static sol (target, length) {
		const start = Math.floor((target.length - length) / 2);
		return target.slice(start, start + length);
	}

	genRandom () {
		const target = this.randomChoiceWords(6);
		const length = Math.max(1, Math.floor(this.random() * target.length));
		this.add({ target, length });
	}
}
