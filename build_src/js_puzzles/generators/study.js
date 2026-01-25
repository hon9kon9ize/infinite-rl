import { PuzzleGenerator } from "../puzzle_generator.js";

export class Study_1 extends PuzzleGenerator {
	static docstring = "* Find a string with 1000 'o's but no two adjacent 'o's.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a string with 1000 'o's but no two adjacent 'o's. */
		return answer.split("o").length - 1 === 1000 && !answer.includes("oo");
	}

	static sol () {
		return "ho".repeat(1000);
	}
}

export class Study_2 extends PuzzleGenerator {
	static docstring = "* Find a string with 1000 'o's, 100 pairs of adjacent 'o's and 801 copies of 'ho'.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a string with 1000 'o's, 100 pairs of adjacent 'o's and 801 copies of 'ho'. */
		return (
			answer.split("o").length - 1 === 1000 &&
			answer.split("oo").length - 1 === 100 &&
			answer.split("ho").length - 1 === 801
		);
	}

	static sol () {
		return "ho".repeat(801) + "o".repeat(199);
	}
}

export class Study_3 extends PuzzleGenerator {
	static docstring = "* Find a permutation of [0, 1, ..., 998] such that the ith element is *not* i, for all i=0, 1, ..., 998.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a permutation of [0, 1, ..., 998] such that the ith element is *not* i, for all i=0, 1, ..., 998. */
		if (!Array.isArray(answer)) return false;
		const sorted = [...answer].sort((a, b) => a - b);
		if (sorted.length !== 999) return false;
		for (let i = 0; i < 999; i++) if (sorted[i] !== i) return false;
		for (let i = 0; i < answer.length; i++) if (answer[i] === i) return false;
		return true;
	}

	static sol () {
		return Array.from({ length: 999 }, (_, i) => (i + 1) % 999);
	}
}

export class Study_4 extends PuzzleGenerator {
	static docstring = "* Find a list of length 10 where the fourth element occurs exactly twice.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a list of length 10 where the fourth element occurs exactly twice. */
		return (
			Array.isArray(answer) &&
			answer.length === 10 &&
			answer.filter((x) => x === answer[3]).length === 2
		);
	}

	static sol () {
		const a = [];
		for (let i = 0; i < 5; i++) a.push(i);
		return [...a, ...a];
	}
}

export class Study_5 extends PuzzleGenerator {
	static docstring = "* Find a list integers such that the integer i occurs i times, for i = 0, 1, 2, ..., 9.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a list integers such that the integer i occurs i times, for i = 0, 1, 2, ..., 9. */
		if (!Array.isArray(answer)) return false;
		for (let i = 0; i < 10; i++)
			if (answer.filter((x) => x === i).length !== i) return false;
		return true;
	}

	static sol () {
		const out = [];
		for (let i = 0; i < 10; i++) for (let j = 0; j < i; j++) out.push(i);
		return out;
	}
}

export class Study_6 extends PuzzleGenerator {
	static docstring = "* Find an integer greater than 10^10 which is 4 mod 123.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find an integer greater than 10^10 which is 4 mod 123. */
		return Number.isInteger(answer) && answer % 123 === 4 && answer > 10 ** 10;
	}

	static sol () {
		return 4 + 10 ** 10 + 123 - (10 ** 10 % 123);
	}
}

export class Study_7 extends PuzzleGenerator {
	static docstring = "* Find a three-digit pattern that occurs more than 8 times in the decimal representation of 8^2888.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a three-digit pattern that occurs more than 8 times in the decimal representation of 8^2888. */
		return (
			typeof answer === "string" &&
			answer.length === 3 &&
			String(8n ** 2888n).split(answer).length - 1 > 8
		);
	}

	static sol () {
		const t = String(8n ** 2888n);
		const seen = new Set();
		let best = null;
		for (let i = 0; i < t.length - 2; i++) {
			const sub = t.slice(i, i + 3);
			if (!seen.has(sub)) {
				seen.add(sub);
				if (best === null || t.split(sub).length - 1 > t.split(best).length - 1)
					best = sub;
			}
		}
		return best;
	}
}

export class Study_8 extends PuzzleGenerator {
	static docstring = "* Find a list of more than 1235 strings such that the 1234th string is a proper substring of the 1235th.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a list of more than 1235 strings such that the 1234th string is a proper substring of the 1235th. */
		return (
			Array.isArray(answer) &&
			answer.length > 1235 &&
			answer[1234] !== answer[1235] &&
			answer[1235].includes(answer[1234])
		);
	}

	static sol () {
		const arr = Array(1235).fill("");
		arr.push("a");
		return arr;
	}
}

export class Study_9 extends PuzzleGenerator {
	static docstring = "* Find a way to rearrange the letters in the pangram \"The quick brown fox jumps over the lazy dog\" to get the pangram \"The five boxing wizards jump quickly\". The answer should be represented as a list of index mappings.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a way to rearrange the letters in the pangram "The quick brown fox jumps over the lazy dog" to get the pangram "The five boxing wizards jump quickly". The answer should be represented as a list of index mappings. */
		const src = "The quick brown fox jumps over the lazy dog";
		const tgt = "The five boxing wizards jump quickly";
		return Array.isArray(answer) && answer.map((i) => src[i]).join("") === tgt;
	}

	static sol () {
		const src = "The quick brown fox jumps over the lazy dog";
		const tgt = "The five boxing wizards jump quickly";
		return Array.from(tgt).map((ch) => src.indexOf(ch));
	}
}

export class Study_10 extends PuzzleGenerator {
	static docstring = "* Find a palindrome of length greater than 11 in the decimal representation of 8^1818.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a palindrome of length greater than 11 in the decimal representation of 8^1818. */
		return (
			typeof answer === "string" &&
			answer === answer.split("").reverse().join("") &&
			answer.length > 11 &&
			String(8n ** 1818n).includes(answer)
		);
	}

	static sol () {
		const t = String(8n ** 1818n);
		for (let le = 12; le <= t.length; le++) {
			for (let i = 0; i + le <= t.length; i++) {
				const sub = t.slice(i, i + le);
				if (sub === sub.split("").reverse().join("")) return sub;
			}
		}
		return null;
	}
}

export class Study_11 extends PuzzleGenerator {
	static docstring = "* Find a list of strings whose length (viewed as a string) is equal to the lexicographically largest element and is equal to the lexicographically smallest element.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a list of strings whose length (viewed as a string) is equal to the lexicographically largest element and is equal to the lexicographically smallest element. */
		return (
			Array.isArray(answer) &&
			Math.min(...answer) === Math.max(...answer) &&
			String(answer[0]) === String(answer.length)
		);
	}

	static sol () {
		return ["1"];
	}
}

export class Study_12 extends PuzzleGenerator {
	static docstring = "* Find a list of 1,000 integers where every two adjacent integers sum to 9, and where the first integer plus 4 is 9.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a list of 1,000 integers where every two adjacent integers sum to 9, and where the first integer plus 4 is 9. */
		return (
			Array.isArray(answer) &&
			answer.length === 1000 &&
			answer.every((v, i) => (i === 0 ? v + 4 : v + answer[i - 1]) === 9)
		);
	}

	static sol () {
		const out = [];
		for (let i = 0; i < 500; i++) {
			out.push(5);
			out.push(4);
		}
		return out;
	}
}

export class Study_13 extends PuzzleGenerator {
	static docstring = "* Find a real number which, when you subtract 3.1415, has a decimal representation starting with 123.456.";
	getExample () {
		return {};
	}

	static sat (answer) {
		return String(answer - 3.1415).startsWith("123.456");
	}

	static sol () {
		return 123.456 + 3.1415;
	}
}

export class Study_14 extends PuzzleGenerator {
	static docstring = "* Find a list of integers such that the sum of the first i integers is i, for i=0, 1, 2, ..., 19.";
	getExample () {
		return {};
	}

	static sat (answer) {
		if (!Array.isArray(answer)) return false;
		for (let i = 0; i < 20; i++)
			if (answer.slice(0, i).reduce((a, b) => a + b, 0) !== i) return false;
		return true;
	}

	static sol () {
		return Array(20).fill(1);
	}
}

export class Study_15 extends PuzzleGenerator {
	static docstring = "* Find a list of integers such that the sum of the first i integers is 2^i -1, for i = 0, 1, 2, ..., 19.";
	getExample () {
		return {};
	}

	static sat (answer) {
		for (let i = 0; i < 20; i++)
			if (answer.slice(0, i).reduce((a, b) => a + b, 0) !== 2 ** i - 1)
				return false;
		return true;
	}

	static sol () {
		return Array.from({ length: 20 }, (_, i) => 2 ** i);
	}
}

export class Study_16 extends PuzzleGenerator {
	static docstring = "* Find a real number such that when you add the length of its decimal representation to it, you get 4.5. Your answer should be the string form of the number in its decimal representation.";
	getExample () {
		return {};
	}

	static sat (answer) {
		const v = parseFloat(answer);
		return !Number.isNaN(v) && v + String(answer).length === 4.5;
	}

	static sol () {
		return String(4.5 - String(4.5).length);
	}
}

export class Study_17 extends PuzzleGenerator {
	static docstring = "* Find a number whose decimal representation is *a longer string* when you add 1,000 to it than when you add 1,001.";
	getExample () {
		return {};
	}

	static sat (answer) {
		return String(answer + 1000).length > String(answer + 1001).length;
	}

	static sol () {
		return -1001;
	}
}

export class Study_18 extends PuzzleGenerator {
	static docstring = "* Find a list of strings that when you combine them in all pairwise combinations gives the six strings:\n        'berlin', 'berger', 'linber', 'linger', 'gerber', 'gerlin'";
	getExample () {
		return {};
	}

	static sat (answer) {
		const wanted = "berlin berger linber linger gerber gerlin".split(" ");
		const pairs = [];
		for (const s of answer)
			for (const t of answer) if (s !== t) pairs.push(s + t);
		return (
			pairs.length === wanted.length && wanted.every((w) => pairs.includes(w))
		);
	}

	static sol () {
		const words = "berlin berger linber linger gerber gerlin".split(" ");
		const seen = new Set();
		const ans = [];
		for (const w of words) {
			const t = w.slice(0, 3);
			if (!seen.has(t)) {
				ans.push(t);
				seen.add(t);
			}
		}
		return ans;
	}
}

export class Study_19 extends PuzzleGenerator {
	static docstring = "* Find a list of integers whose pairwise sums make the set {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.\n        That is find L such that, { i + j | i, j in L } = {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34}.";
	getExample () {
		return {};
	}

	static sat (answer) {
		const s = new Set();
		for (const i of answer) s.add(i);
		const sums = new Set();
		for (const a of answer) for (const b of answer) sums.add(a + b);
		const target = new Set([0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 34]);
		if (sums.size !== target.size) return false;
		for (const v of target) if (!sums.has(v)) return false;
		return true;
	}

	static sol () {
		return [0, 1, 2, 3, 17];
	}
}

export class Study_20 extends PuzzleGenerator {
	static docstring = "* Find a list of integers, starting with 0 and ending with 128, such that each integer either differs from\n        the previous one by one or is thrice the previous one.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a list of integers, starting with 0 and ending with 128, such that each integer either differs from the previous one by one or is thrice the previous one. */
		if (!Array.isArray(answer)) return false;
		let prev = 0;
		for (const j of answer) {
			if (!(j === prev - 1 || j === prev + 1 || j === 3 * prev)) return false;
			prev = j;
		}
		return 128 === prev - 1 || 128 === prev + 1 || 128 === 3 * prev;
	}

	static sol () {
		return [1, 3, 4, 12, 13, 14, 42, 126, 127];
	}
}

export class Study_21 extends PuzzleGenerator {
	static docstring = "* Find a list integers containing exactly three distinct values, such that no integer repeats\n        twice consecutively among the first eleven entries. (So the list needs to have length greater than ten.)";
	getExample () {
		return {};
	}

	static sat (answer) {
		if (!Array.isArray(answer)) return false;
		if (answer.length <= 10) return false;
		for (let i = 0; i < 10; i++) if (answer[i] === answer[i + 1]) return false;
		return new Set(answer).size === 3;
	}

	static sol () {
		return Array.from({ length: 30 }, (_, i) => i % 3);
	}
}

export class Study_22 extends PuzzleGenerator {
	static docstring = "* Find a string s containing exactly five distinct characters which also contains as a substring every other character of s (e.g., if the string s were 'parrotfish' every other character would be 'profs').";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a string s containing exactly five distinct characters which also contains as a substring every other character of s (e.g., if the string s were 'parrotfish' every other character would be 'profs'). */
		const evenChars = answer
			.split("")
			.filter((_, i) => i % 2 === 0)
			.join("");
		return answer.includes(evenChars) && new Set(answer).size === 5;
	}

	static sol () {
		return "abacadaeaaaaaaaaaa";
	}
}

export class Study_23 extends PuzzleGenerator {
	static docstring = "* Find a list of characters which are aligned at the same indices of the three strings 'dee', 'doo', and 'dah!'.";
	getExample () {
		return {};
	}

	static sat (answer) {
		/** Find a list of characters which are aligned at the same indices of the three strings 'dee', 'doo', and 'dah!'. */
		const columns = [
			["d", "d", "d"],
			["e", "o", "a"],
			["e", "o", "h"],
		];
		return columns.some(
			(col) => JSON.stringify(col) === JSON.stringify(answer),
		);
	}

	static sol () {
		return ["d", "d", "d"];
	}
}

export class Study_24 extends PuzzleGenerator {
	static docstring = "* Find a list of integers with exactly three occurrences of seventeen and at least two occurrences of three.";
	getExample () {
		return {};
	}

	static sat (answer) {
		return (
			answer.filter((x) => x === 17).length === 3 &&
			answer.filter((x) => x === 3).length >= 2
		);
	}

	static sol () {
		return [17, 17, 17, 3, 3];
	}
}

export class Study_25 extends PuzzleGenerator {
	static docstring = "* Find a permutation of the string 'Permute me true' which is a palindrome.";
	getExample () {
		return {};
	}

	static sat (answer) {
		return (
			answer.split("").sort().join("") ===
			"Permute me true".split("").sort().join("") &&
			answer === answer.split("").reverse().join("")
		);
	}

	static sol () {
		const base = "Permute me true".slice(1);
		const arr = base.split("").sort();
		const pick = arr.filter((_, i) => i % 2 === 0);
		return pick.join("") + "P" + pick.reverse().join("");
	}
}

export class Study_26 extends PuzzleGenerator {
	static docstring = "* Divide the decimal representation of 8^88 up into strings of length eight.";
	getExample () {
		return {};
	}

	static sat (answer) {
		const t = String(8n ** 88n);
		return answer.join("") === t && answer.every((s) => s.length === 8);
	}

	static sol () {
		const t = String(8n ** 88n);
		const out = [];
		for (let i = 0; i < t.length; i += 8) out.push(t.slice(i, i + 8));
		return out;
	}
}

export class Study_27 extends PuzzleGenerator {
	static docstring = "* Consider a digraph where each node has exactly one outgoing edge. For each edge (u, v), call u the parent and v the child. Then find such a digraph where the grandchildren of the first and second nodes differ but they share the same great-grandchildren. Represented this digraph by the list of children indices.";
	getExample () {
		return {};
	}

	static sat (answer) {
		return (
			answer[answer[0]] !== answer[answer[1]] &&
			answer[answer[answer[0]]] === answer[answer[answer[1]]]
		);
	}

	static sol () {
		return [1, 2, 3, 3];
	}
}

export class Study_28 extends PuzzleGenerator {
	static docstring = "* Find a list of one hundred integers between 0 and 999 which all differ by at least ten from one another.";
	getExample () {
		return {};
	}

	static sat (answer) {
		if (!Array.isArray(answer) || new Set(answer).size !== 100) return false;
		return answer.every(
			(v) =>
				v >= 0 &&
				v < 1000 &&
				[...answer].every((w) => v === w || Math.abs(v - w) >= 10),
		);
	}

	static sol () {
		return Array.from({ length: 100 }, (_, i) => i * 10);
	}
}

export class Study_29 extends PuzzleGenerator {
	static docstring = "* Find a list of more than 995 distinct integers between 0 and 999, inclusive, such that each pair of integers have squares that differ by at least 10.";
	getExample () {
		return {};
	}

	static sat (answer) {
		if (!Array.isArray(answer) || new Set(answer).size <= 995) return false;
		return answer.every(
			(v) =>
				v >= 0 &&
				v < 1000 &&
				[...answer].every((w) => v === w || Math.abs(v * v - w * w) >= 10),
		);
	}

	static sol () {
		const out = [0, 4];
		for (let i = 6; i < 1000; i++) out.push(i);
		return out;
	}
}

export class Study_30 extends PuzzleGenerator {
	static docstring = "* Define f(n) to be the residue of 123 times n mod 1000. Find a list of integers such that the first twenty one are between 0 and 999, inclusive, and are strictly increasing in terms of f(n).";
	getExample () {
		return {};
	}

	static sat (answer) {
		for (let i = 0; i < 20; i++) {
			if (!((123 * answer[i]) % 1000 < (123 * answer[i + 1]) % 1000))
				return false;
			if (answer[i] < 0 || answer[i] >= 1000) return false;
		}
		return true;
	}

	static sol () {
		const arr = Array.from({ length: 1000 }, (_, n) => n)
			.sort((a, b) => ((123 * a) % 1000) - ((123 * b) % 1000))
			.slice(0, 21);
		return arr;
	}
}
