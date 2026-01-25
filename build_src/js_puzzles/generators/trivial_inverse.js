import { PuzzleGenerator } from "../puzzle_generator.js";

export class HelloWorld extends PuzzleGenerator {
	static docstring = "Find a string that when concatenated onto 'world' gives 'Hello world'.";
	getExample () {
		return {};
	}

	static sat (s) {
		return typeof s === "string" && s + "world" === "Hello world";
	}

	static sol () {
		return "Hello ";
	}
}

export class BackWorlds extends PuzzleGenerator {
	static docstring = "* Find a string that when reversed and concatenated onto 'world' gives 'Hello world'.";
	getExample () {
		return {};
	}

	static sat (s) {
		return (
			typeof s === "string" &&
			s.split("").reverse().join("") + "world" === "Hello world"
		);
	}

	static sol () {
		return " olleH";
	}

	static sol2 () {
		return "Hello ".split("").reverse().join("");
	}
}

export class StrAdd extends PuzzleGenerator {
	static docstring = "* Solve simple string addition problem.";
	getExample () {
		return { a: "world", b: "Hello world" };
	}

	static sat (st, a, b) {
		return st + a === b;
	}

	static sol (a, b) {
		return b.slice(0, b.length - a.length);
	}

	genRandom () {
		const b = this.pseudoWord();
		const a = b.slice(Math.floor(this.random() * (b.length + 1)));
		this.add({ a, b });
	}
}

export class StrSetLen extends PuzzleGenerator {
	static docstring = "* Find a string with dups duplicate chars";
	getExample () {
		return { dups: 3 };
	}

	static sat (s, dups) {
		return new Set(s.split("")).size === s.length - dups;
	}

	static sol (dups) {
		return "a".repeat(dups + 1);
	}

	gen (target) {
		for (let dups = 0; dups < target; dups++) {
			this.add({ dups });
		}
	}
}

export class StrMul extends PuzzleGenerator {
	static docstring = "* Find a string which when repeated n times gives target";
	getExample () {
		return { target: "foofoofoofoo", n: 2 };
	}

	static sat (s, target, n) {
		return s.repeat(n) === target;
	}

	static sol (target, n) {
		if (n === 0) return "";
		return target.slice(0, Math.floor(target.length / n));
	}

	genRandom () {
		const base = this.pseudoWord();
		const s = base.repeat(Math.floor(this.random() * 3) + 1);
		const n = Math.floor(this.random() * 10);
		this.add({ target: s.repeat(n), n });
	}
}

export class StrMul2 extends PuzzleGenerator {
	static docstring = "* Find n such that s repeated n times gives target";
	getExample () {
		return { target: "foofoofoofoo", s: "foofoo" };
	}

	static sat (n, target, s) {
		return s.repeat(n) === target;
	}

	static sol (target, s) {
		if (s.length === 0) return 1;
		return Math.floor(target.length / s.length);
	}

	genRandom () {
		const s = this.pseudoWord().repeat(Math.floor(this.random() * 3) + 1);
		const n = Math.floor(this.random() * 10);
		this.add({ target: s.repeat(n), s });
	}
}

export class StrLen extends PuzzleGenerator {
	static docstring = "* Find a string of length n";
	getExample () {
		return { n: 10 };
	}

	static sat (s, n) {
		return typeof s === "string" && s.length === n;
	}

	static sol (n) {
		return "a".repeat(n);
	}

	genRandom () {
		const choices = [10, 100, 1000, 10000];
		const n = choices[Math.floor(this.random() * choices.length)];
		this.add({ n });
	}
}

export class StrAt extends PuzzleGenerator {
	static docstring = "* Find the index of target in string s";
	getExample () {
		return { s: "cat", target: "a" };
	}

	static sat (i, s, target) {
		return s[i] === target;
	}

	static sol (s, target) {
		return s.indexOf(target);
	}

	genRandom () {
		const s = this.pseudoWord().repeat(Math.floor(this.random() * 3) + 1);
		const target = s[Math.floor(this.random() * s.length)];
		this.add({ s, target });
	}
}

export class StrNegAt extends PuzzleGenerator {
	static docstring = "* Find the index of target in s using a negative index.";
	getExample () {
		return { s: "cat", target: "a" };
	}

	static sat (i, s, target) {
		const actualIndex = i < 0 ? s.length + i : i;
		return s[actualIndex] === target && i < 0;
	}

	static sol (s, target) {
		return -(s.length - s.indexOf(target));
	}

	genRandom () {
		const s = this.pseudoWord().repeat(Math.floor(this.random() * 3) + 1);
		const target = s[Math.floor(this.random() * s.length)];
		this.add({ s, target });
	}
}

// Helper function to emulate Python slicing [i:j:k] for strings/arrays
function pythonSlice (arr, i, j, k) {
	const len = arr.length;
	let start, stop, step;

	if (k === 0) throw new Error("slice step cannot be zero");
	step = k || 1;

	// Handle negative indices and defaults based on step direction
	if (step > 0) {
		start =
			i === undefined || i === null
				? 0
				: i < 0
					? Math.max(0, len + i)
					: Math.min(i, len);
		stop =
			j === undefined || j === null
				? len
				: j < 0
					? Math.max(0, len + j)
					: Math.min(j, len);
	} else {
		start =
			i === undefined || i === null
				? len - 1
				: i < 0
					? Math.max(-1, len + i)
					: Math.min(i, len - 1);
		stop =
			j === undefined || j === null
				? -1
				: j < 0
					? Math.max(-1, len + j)
					: Math.min(j, len - 1);
	}

	const result = [];
	if (step > 0) {
		for (let idx = start; idx < stop; idx += step) result.push(arr[idx]);
	} else {
		for (let idx = start; idx > stop; idx += step) result.push(arr[idx]);
	}
	return result;
}

export class StrSlice extends PuzzleGenerator {
	static docstring = "* Find the three slice indices that give the specific target in string s";
	getExample () {
		return { s: "hello world", target: "do" };
	}

	static sat (inds, s, target) {
		if (!Array.isArray(inds) || inds.length !== 3) return false;
		const [i, j, k] = inds;
		try {
			const slice = pythonSlice(s.split(""), i, j, k).join("");
			return slice === target;
		} catch (e) {
			return false;
		}
	}

	static sol (s, target) {
		// brute force ranges similar to Python version
		const maxRange = s.length + 1;
		for (let i = -maxRange; i <= maxRange; i++) {
			for (let j = -maxRange; j <= maxRange; j++) {
				for (let k = -maxRange; k <= maxRange; k++) {
					if (k === 0) continue;
					try {
						const slice = pythonSlice(s.split(""), i, j, k).join("");
						if (slice === target) return [i, j, k];
					} catch (e) { }
				}
			}
		}
	}

	genRandom () {
		let s = "";
		const len = this.random.randrange(1, 20);
		for (let n = 0; n < len; n++) {
			s += this.pseudoWord(1, 1);
		}
		const i = this.random.randrange(-s.length - 1, s.length + 1);
		const j = this.random.randrange(-s.length - 1, s.length + 1);
		const k = this.random.randrange(-s.length, s.length + 1);
		if (k === 0) return;
		try {
			const target = pythonSlice(s.split(""), i, j, k).join("");
			this.add({ s, target });
		} catch (e) {
			// Silently skip
		}
	}
}

export class StrIndex extends PuzzleGenerator {
	static docstring = "* Find a string whose *first* index in big_str is index";
	getExample () {
		return { big_str: "foobar", index: 2 };
	}

	static sat (s, big_str, index) {
		return big_str.indexOf(s) === index;
	}

	static sol (big_str, index) {
		return big_str.slice(index);
	}

	genRandom () {
		const big_str = this.pseudoWord();
		const i = Math.floor(this.random() * big_str.length);
		const index = big_str.indexOf(big_str.slice(i));
		this.add({ big_str, index });
	}
}

export class StrIndex2 extends PuzzleGenerator {
	static docstring = "* Find a string whose *first* index of sub_str is index";
	getExample () {
		return { sub_str: "foobar", index: 2 };
	}

	static sat (big_str, sub_str, index) {
		return big_str.indexOf(sub_str) === index;
	}

	static sol (sub_str, index) {
		let i = 65; // 'A'
		while (sub_str.includes(String.fromCharCode(i))) i++;
		return String.fromCharCode(i).repeat(index) + sub_str;
	}

	genRandom () {
		const sub_str = this.pseudoWord();
		const index = Math.floor(this.random() * 1000);
		this.add({ sub_str, index });
	}
}

export class StrIn extends PuzzleGenerator {
	static docstring = "* Find a string of length length that is in both strings a and b";
	getExample () {
		return { a: "abcdef", b: "cdefgh", length: 3 };
	}

	static sat (s, a, b, length) {
		return s.length === length && a.includes(s) && b.includes(s);
	}

	static sol (a, b, length) {
		for (let i = 0; i + length <= a.length; i++) {
			const sub = a.slice(i, i + length);
			if (b.includes(sub)) return sub;
		}
	}

	genRandom () {
		const sub = this.pseudoWord();
		const a = this.pseudoWord() + sub + this.pseudoWord();
		const b = this.pseudoWord() + sub + this.pseudoWord();
		this.add({ a, b, length: sub.length });
	}
}

export class StrIn2 extends PuzzleGenerator {
	static docstring = "* Find a list of >= count distinct strings that are all contained in s";
	getExample () {
		return { s: "hello", count: 15 };
	}

	static sat (substrings, s, count) {
		if (!Array.isArray(substrings)) return false;
		return (
			new Set(substrings).size === substrings.length &&
			substrings.length >= count &&
			substrings.every((sub) => s.includes(sub))
		);
	}

	static sol (s, count) {
		const set = new Set();
		for (let i = 0; i <= s.length; i++)
			for (let j = i + 1; j <= s.length; j++) set.add(s.slice(i, j));
		return ["", ...Array.from(set).sort()];
	}

	genRandom () {
		const s = this.pseudoWord();
		const count = this.sol({ s }).length;
		this.add({ s, count });
	}
}

export class StrCount extends PuzzleGenerator {
	static docstring = "* Find a string with a certain number of copies of a given substring and of a given length";
	getExample () {
		return { substring: "a", count: 2, length: 5 };
	}

	static sat (string, substring, count, length) {
		return (
			string.split(substring).length - 1 === count && string.length === length
		);
	}

	static sol (substring, count, length) {
		let maxCode = 0;
		for (const ch of substring || "a")
			maxCode = Math.max(maxCode, ch.charCodeAt(0));
		const c = String.fromCharCode(maxCode + 1);
		return (
			substring.repeat(count) + c.repeat(length - substring.length * count)
		);
	}

	genRandom () {
		const substring = this.pseudoWord();
		const count = Math.floor(this.random() * 100);
		const length = substring.length * count + Math.floor(this.random() * 1000);
		this.add({ substring, count, length });
	}
}

export class StrSplit extends PuzzleGenerator {
	static docstring = "* Find a string of a given length with a certain split";
	getExample () {
		return { parts: ["I", "love", "dumplings", "!"], length: 100 };
	}

	static sat (x, parts, length) {
		/** Find a string of a given length with a certain split */
		return (
			x.length === length &&
			x
				.split(" ")
				.filter((c) => c !== "")
				.join(" ") === parts.join(" ")
		);
	}

	static sol (parts, length) {
		const joined = parts.join(" ");
		return joined + " ".repeat(length - joined.length);
	}

	genRandom () {
		const parts = Array.from(
			{ length: Math.floor(this.random() * 5) + 1 },
			() => this.pseudoWord(),
		);
		const length = parts.join(" ").length + Math.floor(this.random() * 100);
		this.add({ parts, length });
	}
}

export class StrSplitter extends PuzzleGenerator {
	static docstring = "* Find a separator that when used to split a given string gives a certain result";
	getExample () {
		return {
			parts: ["I", "love", "dumplings", "!", ""],
			string: "I_love_dumplings_!_",
		};
	}

	static sat (
		x,
		parts = ["I", "love", "dumplings", "!", ""],
		string = "I_love_dumplings_!_",
	) {
		if (typeof parts === "object" && parts !== null && !Array.isArray(parts)) {
			// Called with inputs object
			({ parts, string } = parts);
		}
		return JSON.stringify(string.split(x)) === JSON.stringify(parts);
	}

	static sol (parts, string) {
		if (typeof parts === "object" && parts !== null && !Array.isArray(parts)) {
			({ parts, string } = parts);
		}
		if (parts.length <= 1) return string + string;
		const length = Math.floor(
			(string.length - parts.join("").length) / (parts.length - 1),
		);
		const start = parts[0].length;
		return string.slice(start, start + length);
	}

	genRandom () {
		const x = this.pseudoWord();
		const parts = Array.from(
			{ length: Math.floor(this.random() * 5) + 1 },
			() => this.pseudoWord(),
		);
		const string = parts.join(x);
		this.add({ parts, string });
	}
}

export class StrJoiner extends PuzzleGenerator {
	static docstring = "* Find a separator that when used to join a given string gives a certain result. This is related to the previous problem but there are some edge cases that differ.";
	getExample () {
		return {
			parts: ["I!!", "!love", "dumplings", "!", ""],
			string: "I!!!!!love!!dumplings!!!!!",
		};
	}

	static sat (x, parts, string) {
		return parts.join(x) === string;
	}

	static sol (parts, string) {
		if (parts.length <= 1) return "";
		const length = Math.floor(
			(string.length - parts.join("").length) / (parts.length - 1),
		);
		const start = parts[0].length;
		return string.slice(start, start + length);
	}

	genRandom () {
		const x = this.pseudoWord();
		const parts = Array.from(
			{ length: Math.floor(this.random() * 5) + 1 },
			() => this.pseudoWord(),
		);
		const string = parts.join(x);
		this.add({ parts, string });
	}
}

export class StrParts extends PuzzleGenerator {
	static docstring = "* Find parts that when joined give a specific string.";
	getExample () {
		return { sep: "!!", string: "I!!!!!love!!dumplings!!!!!" };
	}

	static sat (parts, sep, string) {
		return parts.join(sep) === string && parts.every((p) => !p.includes(sep));
	}

	static sol (sep, string) {
		return string.split(sep);
	}

	genRandom () {
		const sep = this.pseudoWord();
		const parts = Array.from(
			{ length: Math.floor(this.random() * 5) + 1 },
			() => this.pseudoWord(),
		);
		const string = parts.join(sep);
		this.add({ sep, string });
	}
}

export class ListSetLen extends PuzzleGenerator {
	static docstring = "* Find a list with a certain number of duplicate items";
	getExample () {
		return { dups: 3 };
	}

	static sat (li, dups) {
		return new Set(li).size === li.length - dups;
	}

	static sol (dups) {
		return Array(dups + 1).fill(1);
	}

	genRandom () {
		this.add({ dups: Math.floor(this.random() * 1e5) });
	}
}

export class ListMul extends PuzzleGenerator {
	static docstring = "* Find a list that when multiplied n times gives the target list";
	getExample () {
		return { target: [17, 9, -1, 17, 9, -1], n: 2 };
	}

	static sat (li, target, n) {
		return (
			JSON.stringify(li.concat(...Array(n - 1).fill(li))) ===
			JSON.stringify(target)
		);
	}

	static sol (target, n) {
		if (n === 0) return [];
		return target.slice(0, Math.floor(target.length / n));
	}

	genRandom () {
		const li = Array.from({ length: Math.floor(this.random() * 9) + 1 }, () =>
			Math.floor((this.random() - 0.5) * 2 * 1e5),
		);
		const n = Math.floor(this.random() * 10);
		this.add({ target: Array(n).fill(li).flat(), n });
	}
}

export class ListLen extends PuzzleGenerator {
	static docstring = "* Find a list of a given length n";
	getExample () {
		return { n: 5 };
	}

	static sat (li, n) {
		return Array.isArray(li) && li.length === n;
	}

	static sol (n) {
		return Array(n).fill(1);
	}

	salGeneration () { }

	genRandom () {
		const choices = [10, 100, 1000, 10000];
		const n = choices[Math.floor(this.random() * choices.length)];
		this.add({ n });
	}
}

export class ListAt extends PuzzleGenerator {
	static docstring = "* Find the index of an item in a list. Any such index is fine.";
	getExample () {
		return { li: [17, 31, 91, 18, 42, 1, 9], target: 18 };
	}

	static sat (i, li, target) {
		return li[i] === target;
	}

	static sol (li, target) {
		return li.indexOf(target);
	}

	genRandom () {
		const li = Array.from({ length: Math.floor(this.random() * 19) + 1 }, () =>
			Math.floor((this.random() - 0.5) * 200),
		);
		const target = li[Math.floor(this.random() * li.length)];
		this.add({ li, target });
	}
}

export class ListNegAt extends PuzzleGenerator {
	static docstring = "* Find the index of an item in a list using negative indexing.";
	getExample () {
		return { li: [17, 31, 91, 18, 42, 1, 9], target: 91 };
	}

	static sat (i, li, target) {
		const actualIndex = i < 0 ? li.length + i : i;
		return li[actualIndex] === target && i < 0;
	}

	static sol (li, target) {
		return li.indexOf(target) - li.length;
	}

	genRandom () {
		const li = Array.from({ length: Math.floor(this.random() * 19) + 1 }, () =>
			Math.floor((this.random() - 0.5) * 200),
		);
		const target = li[Math.floor(this.random() * li.length)];
		this.add({ li, target });
	}
}

export class ListSlice extends PuzzleGenerator {
	static docstring = "* Find three slice indices to achieve a given list slice";
	getExample () {
		return { li: [42, 18, 21, 103, -2, 11], target: [-2, 21, 42] };
	}

	static sat (inds, li, target) {
		if (!Array.isArray(inds)) return false;
		const [i, j, k] = inds;
		try {
			const slice = pythonSlice(li, i, j, k);
			return JSON.stringify(slice) === JSON.stringify(target);
		} catch (e) {
			return false;
		}
	}

	static sol (li, target) {
		// Try all combinations of i, j, k to find a match
		const maxRange = li.length + 1;
		for (let i = -maxRange; i <= maxRange; i++) {
			for (let j = -maxRange; j <= maxRange; j++) {
				for (let k = -maxRange; k <= maxRange; k++) {
					if (k === 0) continue; // k cannot be 0
					try {
						const slice = pythonSlice(li, i, j, k);
						if (JSON.stringify(slice) === JSON.stringify(target)) {
							return [i, j, k];
						}
					} catch (e) { }
				}
			}
		}
		return undefined;
	}

	genRandom () {
		const li = [];
		for (let n = 0; n < this.random.randrange(1, 20); n++) {
			li.push(this.random.randrange(-100, 100));
		}
		const i = this.random.randrange(-li.length - 1, li.length + 1);
		const j = this.random.randrange(-li.length - 1, li.length + 1);
		const k = this.random.randrange(-li.length, li.length + 1);
		if (k === 0) return;
		try {
			const target = pythonSlice(li, i, j, k);
			// Only add if non-trivial
			if (
				(target.length !== 0 &&
					JSON.stringify(target) !== JSON.stringify(li)) ||
				this.random.randrange(50) === 0
			) {
				this.add({ li, target });
			}
		} catch (e) {
			// Silently skip
		}
	}
}

export class ListIndex extends PuzzleGenerator {
	static docstring = "* Find the item whose first index in li is index";
	getExample () {
		return { li: [17, 2, 3, 9, 11, 11], index: 4 };
	}

	static sat (item, li, index) {
		return li.indexOf(item) === index;
	}

	static sol (li, index) {
		return li[index];
	}

	genRandom () {
		const li = Array.from({ length: Math.floor(this.random() * 19) + 1 }, () =>
			Math.floor((this.random() - 0.5) * 200),
		);
		const i = Math.floor(this.random() * li.length);
		this.add({ li, index: i });
	}
}

export class ListIndex2 extends PuzzleGenerator {
	static docstring = "* Find a list that contains i first at index index";
	getExample () {
		return { i: 29, index: 10412 };
	}

	static sat (li, i, index) {
		return li.indexOf(i) === index;
	}

	static sol (i, index) {
		return Array(index)
			.fill(i - 1)
			.concat([i]);
	}

	solGenRandom () { }

	genRandom () {
		const i = Math.floor((this.random() - 0.5) * 2 * 1e5);
		const index = Math.floor(this.random() * 1e5);
		this.add({ i, index });
	}
}

export class ListIn extends PuzzleGenerator {
	static docstring = "* Find an item that is in both lists a and b";
	getExample () {
		return { a: ["cat", "dot", "bird"], b: ["tree", "fly", "dot"] };
	}

	static sat (s, a, b) {
		return a.includes(s) && b.includes(s);
	}

	static sol (a, b) {
		return b.find((s) => a.includes(s));
	}

	genRandom () {
		const a = Array.from({ length: Math.floor(this.random() * 99) + 1 }, () =>
			this.pseudoWord(),
		);
		const b = Array.from({ length: Math.floor(this.random() * 99) + 1 }, () =>
			this.pseudoWord(),
		);
		b.splice(
			Math.floor(this.random() * b.length),
			0,
			a[Math.floor(this.random() * a.length)],
		);
		this.add({ a, b });
	}
}

export class IntNeg extends PuzzleGenerator {
	static docstring = "* Solve a unary negation problem";
	getExample () {
		return { a: 93252338 };
	}

	static sat (x, a) {
		return -x === a;
	}

	static sol (a) {
		return -a;
	}

	genRandom () {
		this.add({ a: Math.floor((this.random() - 0.5) * 2 * 1e16) });
	}
}

export class IntSum extends PuzzleGenerator {
	static docstring = "* Solve a sum problem";
	getExample () {
		return { a: 1073258, b: 72352549 };
	}

	static sat (x, a, b) {
		return a + x === b;
	}

	static sol (a, b) {
		return b - a;
	}

	genRandom () {
		this.add({
			a: Math.floor((this.random() - 0.5) * 2 * 1e16),
			b: Math.floor((this.random() - 0.5) * 2 * 1e16),
		});
	}
}

export class IntSub extends PuzzleGenerator {
	static docstring = "* Solve a subtraction problem";
	getExample () {
		return { a: -382, b: 14546310 };
	}

	static sat (x, a, b) {
		return x - a === b;
	}

	static sol (a, b) {
		return a + b;
	}

	genRandom () {
		const m = 10 ** 16;
		this.add({
			a: Math.floor((this.random() - 0.5) * 2 * m),
			b: Math.floor((this.random() - 0.5) * 2 * m),
		});
	}
}

export class IntSub2 extends PuzzleGenerator {
	static docstring = "* Solve a subtraction problem";
	getExample () {
		return { a: 8665464, b: -93206 };
	}

	static sat (x, a, b) {
		return a - x === b;
	}

	static sol (a, b) {
		return a - b;
	}

	genRandom () {
		const m = 10 ** 16;
		this.add({
			a: Math.floor((this.random() - 0.5) * 2 * m),
			b: Math.floor((this.random() - 0.5) * 2 * m),
		});
	}
}

export class IntMul extends PuzzleGenerator {
	static docstring = "* Solve a multiplication problem";
	getExample () {
		return { a: 14302, b: 5 };
	}

	static sat (n, a, b) {
		return b * n + (a % b) === a;
	}

	static sol (a, b) {
		return Math.floor(a / b);
	}

	genRandom () {
		const m = 10 ** 6;
		const a = Math.floor((this.random() - 0.5) * 2 * m);
		const b = Math.floor((this.random() - 0.5) * 200);
		if (b !== 0) this.add({ a, b });
	}
}

export class IntDiv extends PuzzleGenerator {
	static docstring = "* Solve a division problem";
	getExample () {
		return { a: 3, b: 23463462 };
	}

	static sat (n, a, b) {
		return Math.floor(b / n) === a;
	}

	static sol (a, b) {
		if (a === 0) return 2 * b;
		const cand = Math.floor(b / a);
		for (const n of [cand, cand - 1, cand + 1])
			if (Math.floor(b / n) === a) return n;
	}

	genRandom () {
		const m = 10 ** 16;
		const n = Math.floor((this.random() - 0.5) * 2 * m);
		const b = Math.floor((this.random() - 0.5) * 2 * m);
		if (n !== 0) {
			const a = Math.floor(b / n);
			this.add({ a, b });
		}
	}
}

export class IntDiv2 extends PuzzleGenerator {
	static docstring = "* Find n that when divided by b is a";
	getExample () {
		return { a: 345346363, b: 10 };
	}

	static sat (n, a, b) {
		return Math.floor(n / b) === a;
	}

	static sol (a, b) {
		return a * b;
	}

	genRandom () {
		const m = 10 ** 16;
		const a = Math.floor((this.random() - 0.5) * 2 * m);
		const b = Math.floor((this.random() - 0.5) * 2 * m);
		if (b !== 0) this.add({ a, b });
	}
}



// export class IntSquareRoot extends PuzzleGenerator {
// 	static docstring = "Compute an integer that when squared equals perfect-square a.";
// 	getExample () {
// 		return { a: 10201202001 };
// 	}

// 	static sat (x, a) {
// 		return x * x === a;
// 	}

// 	static sol (a) {
// 		return Math.floor(Math.sqrt(a));
// 	}

// 	genRandom () {
// 		const z = Math.floor(this.random() * 2 ** 31);
// 		this.add({ a: z * z });
// 	}
// }



// export class IntNegSquareRoot extends PuzzleGenerator {
// 	static docstring = "Find a negative integer that when squared equals perfect-square a.";
// 	getExample () {
// 		return { a: 10000200001 };
// 	}

// 	static sat (n, a) {
// 		return a === n * n && n < 0;
// 	}

// 	static sol (a) {
// 		return -Math.floor(Math.sqrt(a));
// 	}

// 	solRandom () { }

// 	solGenRandom () { }

// 	genRandom () {
// 		const z = Math.floor(this.random() * 2 ** 31);
// 		this.add({ a: z * z });
// 	}
// }


export class FloatSquareRoot extends PuzzleGenerator {
	static docstring = "Find a number that when squared is close to a.";
	getExample () {
		return { a: 1020 };
	}

	static sat (x, a) {
		return Math.abs(x * x - a) < 10 ** -3;
	}

	static sol (a) {
		return Math.sqrt(a);
	}

	genRandom () {
		this.add({ a: Math.floor(this.random() * 1e10) });
	}
}

export class FloatNegSquareRoot extends PuzzleGenerator {
	static docstring = "Find a negative number that when squared is close to a.";
	getExample () {
		return { a: 1020 };
	}

	static sat (x, a) {
		return Math.abs(x * x - a) < 10 ** -3 && x < 0;
	}

	static sol (a) {
		return -Math.sqrt(a);
	}

	genRandom () {
		this.add({ a: Math.floor(this.random() * 1e10) });
	}
}
