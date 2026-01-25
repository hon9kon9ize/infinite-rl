import { PuzzleGenerator } from "../puzzle_generator.js";

export class IsEven extends PuzzleGenerator {
	static docstring = "Determine if n can be evenly divided into two equal numbers. (Easy)";

	getExample () {
		return { n: 10 };
	}

	/**
	 * Determine if n can be evenly divided into two equal numbers. (Easy)
	 */
	static sat (answer, n) {
		let i = 0;
		while (i <= n) {
			if (i + i === n) {
				return answer === true;
			}
			i += 1;
		}
		return answer === false;
	}

	static sol (n) {
		return n % 2 === 0;
	}

	gen (targetNumInstances) {
		for (let n = 0; n < targetNumInstances; n++) {
			this.add({ n });
		}
	}
}

export class Abbreviate extends PuzzleGenerator {
	static docstring = "Abbreviate strings longer than a given length by replacing everything but the first and last characters by an integer indicating how many characters there were in between them.";

	getExample () {
		return { word: "antidisestablishmentarianism", max_len: 10 };
	}

	/**
	 * Abbreviate strings longer than a given length by replacing everything but the first and last characters by an integer indicating how many characters there were in between them.
	 */
	static sat (answer, word, max_len) {
		if (word.length <= max_len) {
			return word === answer;
		}
		const middle = answer.slice(1, -1);
		return (
			Number(middle) === word.length - 2 &&
			word[0] === answer[0] &&
			word[word.length - 1] === answer[answer.length - 1]
		);
	}

	static sol (word, max_len) {
		if (word.length <= max_len) return word;
		return `${word[0]}${word.length - 2}${word[word.length - 1]}`;
	}

	genRandom () {
		const word = this.random.pseudo_word(3, 30);
		const max_len = this.random.randrange(5, 15);
		this.add({ word, max_len });
	}
}

export class SquareTiles extends PuzzleGenerator {
	static docstring = "Find a minimal list of corner locations for a×a tiles that covers [0, m] × [0, n] and does not double-cover squares.";

	getExample () {
		return { m: 10, n: 9, a: 5, target: 4 };
	}

	/**
	 * Find a minimal list of corner locations for a×a tiles that covers [0, m] × [0, n] and does not double-cover squares.
	 */
	static sat (answer, m, n, a, target) {
		if (!Array.isArray(answer)) return false;
		const covered = new Set();
		for (const [i, j] of answer) {
			for (let x = 0; x < a; x++) {
				for (let y = 0; y < a; y++) {
					covered.add(`${i + x},${j + y}`);
				}
			}
		}
		if (covered.size !== answer.length * a * a) return false;
		for (let x = 0; x < m; x++) {
			for (let y = 0; y < n; y++) {
				if (!covered.has(`${x},${y}`)) return false;
			}
		}
		return answer.length <= target;
	}

	static sol (m, n, a, target) {
		const ans = [];
		for (let x = 0; x < m; x += a) {
			for (let y = 0; y < n; y += a) {
				ans.push([x, y]);
			}
		}
		return ans;
	}

	genRandom () {
		const a = this.random.randrange(1, 11);
		const m = this.random.randrange(1, this.random.choice([10, 100, 1000]));
		const n = this.random.randrange(1, this.random.choice([10, 100, 1000]));
		const target = this.sol({ m, n, a }).length + this.random.randrange(5);
		this.add({ a, m, n, target });
	}
}

export class EasyTwos extends PuzzleGenerator {
	static docstring = "Given a list of lists of triples of integers, return True for each list with a total of at least 2 and False for each other list.";

	getExample () {
		return {
			trips: [
				[1, 1, 0],
				[1, 0, 0],
				[0, 0, 0],
				[0, 1, 1],
				[0, 1, 1],
				[1, 1, 1],
				[1, 0, 1],
			],
		};
	}

	/**
	 * Given a list of lists of triples of integers, return True for each list with a total of at least 2 and False for each other list.
	 */
	static sat (answer, trips) {
		if (!Array.isArray(answer) || answer.length !== trips.length) return false;
		return answer.every((b, idx) => {
			const sum = trips[idx].reduce((acc, val) => acc + val, 0);
			return sum >= 2 ? b === true : b === false;
		});
	}

	static sol (trips) {
		return trips.map((trip) => trip.reduce((a, b) => a + b, 0) >= 2);
	}

	genRandom () {
		const trips = Array.from({ length: this.random.randrange(20) }, () =>
			Array.from({ length: 3 }, () => this.random.randrange(2)),
		);
		this.add({ trips });
	}
}

export class DecreasingCountComparison extends PuzzleGenerator {
	static docstring = "Given a list of non-increasing integers and given an integer k, determine how many positive integers in the list are at least as large as the kth.";

	getExample () {
		return { scores: [100, 95, 80, 70, 65, 9, 9, 9, 4, 2, 1], k: 6 };
	}

	/**
	 * Given a list of non-increasing integers and given an integer k, determine how many positive integers in the list are at least as large as the kth.
	 */
	static sat (answer, scores, k) {
		const threshold = Math.max(scores[k], 1);
		return (
			answer === scores.filter((s) => s >= threshold).length && scores[k] > 0
		);
	}

	static sol (scores, k) {
		const threshold = Math.max(scores[k], 1);
		return scores.filter((s) => s >= threshold).length;
	}

	genRandom () {
		const n = this.random.randrange(1, 50);
		const maxScore = this.random.randrange(50);
		const scores = Array.from({ length: n }, () =>
			this.random.randrange(maxScore + 1),
		).sort((a, b) => b - a);
		const k = this.random.randrange(n);
		this.add({ scores, k });
	}
}

export class VowelDrop extends PuzzleGenerator {
	static docstring = "Given an alphabetic string s, remove all vowels (aeiouy/AEIOUY), insert a \".\" before each remaining letter (consonant), and make everything lowercase.";

	getExample () {
		return { s: "Problems" };
	}

	/**
	 * Given an alphabetic string s, remove all vowels (aeiouy/AEIOUY), insert a "." before each remaining letter (consonant), and make everything lowercase.
	 */
	static sat (answer, s) {
		let idx = 0;
		for (const c of s.toLowerCase()) {
			if ("aeiouy".includes(c)) continue;
			if (answer[idx] !== "." || answer[idx + 1] !== c) return false;
			idx += 2;
		}
		return idx === answer.length;
	}

	static sol (s) {
		return s
			.toLowerCase()
			.split("")
			.filter((c) => !"aeiouy".includes(c))
			.map((c) => `.${c}`)
			.join("");
	}

	genRandom () {
		const word = this.random.pseudo_word(3, 20);
		const s = word
			.split("")
			.map((c) => (this.random.random() > 0.5 ? c.toUpperCase() : c))
			.join("");
		this.add({ s });
	}
}

export class DominoTile extends PuzzleGenerator {
	static docstring = "Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2] with i2 == i1 and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap.";

	getExample () {
		return { m: 10, n: 5, target: 50 };
	}

	/**
	 * Tile an m x n checkerboard with 2 x 1 tiles. The solution is a list of fourtuples [i1, j1, i2, j2] with i2 == i1 and j2 == j1 + 1 or i2 == i1 + 1 and j2 == j1 with no overlap.
	 */
	static sat (answer, m, n, target) {
		const covered = new Set();
		for (const [i1, j1, i2, j2] of answer) {
			if (i2 < i1 || j2 < j1) return false;
			if (Math.abs(i2 - i1) + Math.abs(j2 - j1) !== 1) return false;
			covered.add(`${i1},${j1}`);
			covered.add(`${i2},${j2}`);
		}
		return covered.size === target && covered.size === answer.length * 2;
	}

	static sol (m, n, target) {
		if (m % 2 === 0) {
			const ans = [];
			for (let i = 0; i < m; i += 2) {
				for (let j = 0; j < n; j++) ans.push([i, j, i + 1, j]);
			}
			return ans;
		}
		if (n % 2 === 0) {
			const ans = [];
			for (let i = 0; i < m; i++) {
				for (let j = 0; j < n; j += 2) ans.push([i, j, i, j + 1]);
			}
			return ans;
		}
		const ans = [];
		for (let i = 1; i < m; i += 2) {
			for (let j = 0; j < n; j++) ans.push([i, j, i + 1, j]);
		}
		for (let j = 0; j < n - 1; j += 2) ans.push([0, j, 0, j + 1]);
		return ans;
	}

	genRandom () {
		const m = this.random.randrange(1, 50);
		const n = this.random.randrange(1, 50);
		const target = m * n - ((m * n) % 2);
		this.add({ m, n, target });
	}
}

export class IncDec extends PuzzleGenerator {
	static docstring = "Given a sequence of operations \"++x\", \"x++\", \"--x\", \"x--\", and a target value, find initial value so that the final value is the target value.";

	getExample () {
		return { ops: ["x++", "--x", "--x"], target: 19143212 };
	}

	/**
	 * Given a sequence of operations "++x", "x++", "--x", "x--", and a target value, find initial value so that the final value is the target value.
	 */
	static sat (answer, ops, target) {
		let value = answer;
		for (const op of ops) {
			if (op === "++x" || op === "x++") value += 1;
			else value -= 1;
		}
		return value === target;
	}

	static sol (ops, target) {
		return (
			target -
			ops.filter((op) => op === "++x" || op === "x++").length +
			ops.filter((op) => op === "--x" || op === "x--").length
		);
	}

	genRandom () {
		const target = this.random.randrange(10 ** 5);
		const numOps = this.random.randrange(this.random.choice([10, 100, 1000]));
		const ops = Array.from({ length: numOps }, () =>
			this.random.choice(["x++", "++x", "--x", "x--"]),
		);
		this.add({ ops, target });
	}
}

export class CompareInAnyCase extends PuzzleGenerator {
	static docstring = "Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t.";

	getExample () {
		return { s: "aaAab", t: "aAaaB" };
	}

	mixCase (word) {
		return word
			.split("")
			.map((c) =>
				this.random.random() > 0.5 ? c.toUpperCase() : c.toLowerCase(),
			)
			.join("");
	}

	/**
	 * Ignoring case, compare s, t lexicographically. Output 0 if they are =, -1 if s < t, 1 if s > t.
	 */
	static sat (answer, s, t) {
		const lowerS = s.toLowerCase();
		const lowerT = t.toLowerCase();
		if (answer === 0) return lowerS === lowerT;
		if (answer === 1) return lowerS > lowerT;
		if (answer === -1) return lowerS < lowerT;
		return false;
	}

	static sol (s, t) {
		const lowerS = s.toLowerCase();
		const lowerT = t.toLowerCase();
		if (lowerS === lowerT) return 0;
		return lowerS > lowerT ? 1 : -1;
	}

	genRandom () {
		const base = this.random.pseudo_word(1, 8);
		const s = this.mixCase(base);
		let t;
		if (this.random.randrange(3)) {
			t = this.mixCase(
				s.slice(0, this.random.randrange(s.length + 1)) +
				this.random.pseudo_word(1, 10),
			);
		} else {
			t = this.mixCase(s);
		}
		this.add({ s, t });
	}
}

export class SlidingOne extends PuzzleGenerator {
	static docstring = "We are given a 5x5 matrix with a single 1 like: 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Find a (minimal) sequence of row and column swaps to move the 1 to the center. A move is a string in \"0\"-\"4\" indicating a row swap and \"a\"-\"e\" indicating a column swap";

	getExample () {
		return {
			matrix: [
				[0, 0, 0, 0, 0],
				[0, 0, 0, 0, 1],
				[0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0],
			],
			max_moves: 3,
		};
	}

	/**
	 * We are given a 5x5 matrix with a single 1 like: 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Find a (minimal) sequence of row and column swaps to move the 1 to the center. A move is a string in "0"-"4" indicating a row swap and "a"-"e" indicating a column swap
	 */
	static sat (answer, matrix, max_moves) {
		const mat = matrix.map((row) => row.slice());
		for (const move of answer) {
			const rows = "01234";
			const cols = "abcde";
			if (rows.includes(move)) {
				const i = rows.indexOf(move);
				const j = i + 1;
				[mat[i], mat[j]] = [mat[j], mat[i]];
			}
			if (cols.includes(move)) {
				const j = cols.indexOf(move);
				for (const row of mat) {
					[row[j], row[j + 1]] = [row[j + 1], row[j]];
				}
			}
		}
		return answer.length <= max_moves && mat[2][2] === 1;
	}

	static sol (matrix, max_moves) {
		let i = matrix.findIndex((row) => row.includes(1));
		let j = matrix[i].indexOf(1);
		let ans = "";
		while (i > 2) {
			ans += String(i - 1);
			i -= 1;
		}
		while (i < 2) {
			ans += String(i);
			i += 1;
		}
		const cols = "abcde";
		while (j > 2) {
			ans += cols[j - 1];
			j -= 1;
		}
		while (j < 2) {
			ans += cols[j];
			j += 1;
		}
		return ans.split("");
	}

	gen (targetNumInstances) {
		for (let i = 0; i < 5; i++) {
			for (let j = 0; j < 5; j++) {
				if (this.num_generated_so_far() >= targetNumInstances) return;
				const matrix = Array.from({ length: 5 }, () =>
					Array.from({ length: 5 }, () => 0),
				);
				matrix[i][j] = 1;
				const max_moves = Math.abs(2 - i) + Math.abs(2 - j);
				this.add({ matrix, max_moves });
			}
		}
	}
}

export class SortPlusPlus extends PuzzleGenerator {
	static docstring = "Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3";

	getExample () {
		return { inp: "1+1+3+1+3+2+2+1+3+1+2" };
	}

	/**
	 * Sort numbers in a sum of digits, e.g., 1+3+2+1 -> 1+1+2+3
	 */
	static sat (answer, inp) {
		const normalizedParts = inp.split("+").sort();
		const answerParts = answer.split("+");
		return (
			answerParts.length === normalizedParts.length &&
			answerParts.every((v, idx) => v === normalizedParts[idx])
		);
	}

	static sol (inp) {
		return inp.split("+").sort().join("+");
	}

	genRandom () {
		const length = this.random.randrange(50);
		const inp = Array.from({ length }, () => this.random.choice("123")).join(
			"+",
		);
		this.add({ inp });
	}
}

export class CapitalizeFirstLetter extends PuzzleGenerator {
	static docstring = "Capitalize the first letter of word";

	getExample () {
		return { word: "konjac" };
	}

	/**
	 * Capitalize the first letter of word
	 */
	static sat (answer, word) {
		if (answer.length !== word.length) return false;
		return (
			answer[0] === word[0].toUpperCase() && answer.slice(1) === word.slice(1)
		);
	}

	static sol (word) {
		return word[0].toUpperCase() + word.slice(1);
	}

	genRandom () {
		const word = this.random.pseudo_word(3, 15);
		this.add({ word });
	}
}

export class LongestSubsetString extends PuzzleGenerator {
	static docstring = "You are given a string consisting of a's, b's and c's, find any longest substring containing no repeated consecutive characters.";

	getExample () {
		return { s: "abbbcabbac", target: 7 };
	}

	/**
	 * You are given a string consisting of a's, b's and c's, find any longest substring containing no repeated consecutive characters.
	 */
	static sat (answer, s, target) {
		if (answer.length < target) return false;
		for (let i = 0; i < answer.length - 1; i++) {
			if (answer[i] === answer[i + 1]) return false;
		}
		let idx = 0;
		for (const c of answer) {
			idx = s.indexOf(c, idx);
			if (idx === -1) return false;
			idx += 1;
		}
		return true;
	}

	static sol (s, target) {
		let result = s[0] || "";
		for (let i = 1; i < s.length; i++) {
			if (s[i] !== s[i - 1]) result += s[i];
		}
		return result;
	}

	genRandom () {
		const n = this.random.randrange(this.random.choice([10, 100, 1000]));
		const s = Array.from({ length: n }, () => this.random.choice("abc")).join(
			"",
		);
		this.add({ s, target: this.sol({ s }).length });
	}
}

export class FindHomogeneousSubstring extends PuzzleGenerator {
	static docstring = "You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are all 0's or all 1's.";

	getExample () {
		return { s: "0000101111111000010", k: 5 };
	}

	/**
	 * You are given a string consisting of 0's and 1's. Find an index after which the subsequent k characters are all 0's or all 1's.
	 */
	static sat (answer, s, k) {
		if (typeof answer !== "number" || answer < 0 || answer + k > s.length)
			return false;
		const char = s[answer];
		if (!char) return false;
		return s.slice(answer, answer + k) === char.repeat(k);
	}

	static sol (s, k) {
		const zero = "0".repeat(k);
		const one = "1".repeat(k);
		if (s.includes(zero)) return s.indexOf(zero);
		return s.indexOf(one);
	}

	genRandom () {
		const k = this.random.randrange(1, 20);
		const baseLength = this.random.choice([10, 100, 1000]);
		let s = Array.from({ length: baseLength }, () =>
			this.random.choice("01"),
		).join("");
		if (!s.includes("0".repeat(k)) && !s.includes("1".repeat(k))) {
			const idx = this.random.randrange(s.length + 1);
			const patch = this.random.choice(["0", "1"]).repeat(k);
			s = s.slice(0, idx) + patch + s.slice(idx);
		}
		this.add({ s, k });
	}
}

export class Triple0 extends PuzzleGenerator {
	static docstring = "Find the missing triple of integers to make them all add up to 0 coordinatewise";

	getExample () {
		return {
			nums: [
				[1, 2, 3],
				[9, -2, 8],
				[17, 2, 50],
			],
		};
	}

	/**
	 * Find the missing triple of integers to make them all add up to 0 coordinatewise
	 */
	static sat (answer, nums) {
		if (!Array.isArray(answer) || answer.length !== 3) return false;
		for (let i = 0; i < 3; i++) {
			if (answer[i] + nums.reduce((sum, vec) => sum + vec[i], 0) !== 0)
				return false;
		}
		return true;
	}

	static sol (nums) {
		return [
			-nums.reduce((sum, vec) => sum + vec[0], 0),
			-nums.reduce((sum, vec) => sum + vec[1], 0),
			-nums.reduce((sum, vec) => sum + vec[2], 0),
		];
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(10) }, () => [
			this.random.randrange(-100, 100),
			this.random.randrange(-100, 100),
			this.random.randrange(-100, 100),
		]);
		this.add({ nums });
	}
}

export class TotalDifference extends PuzzleGenerator {
	static docstring = "Find n such that n + a == b * (the sum of the first c integers)";

	getExample () {
		return { a: 17, b: 100, c: 20 };
	}

	/**
	 * Find n such that n + a == b * (the sum of the first c integers)
	 */
	static sat (answer, a, b, c) {
		let sum = 0;
		for (let i = 0; i < c; i++) sum += b * i;
		return answer + a === sum;
	}

	static sol (a, b, c) {
		let sum = 0;
		for (let i = 0; i < c; i++) sum += b * i;
		return -a + sum;
	}

	genRandom () {
		const a = this.random.randrange(1, 100);
		const b = this.random.randrange(1, 100);
		const c = this.random.randrange(1, 100);
		this.add({ a, b, c });
	}
}

export class TripleDouble extends PuzzleGenerator {
	static docstring = "Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w.";

	getExample () {
		return { v: 17, w: 100 };
	}

	/**
	 * Find the smallest n such that if v is tripled n times and w is doubled n times, v exceeds w.
	 */
	static sat (answer, v, w) {
		let v_val = v;
		let w_val = w;
		for (let i = 0; i < answer; i++) {
			if (v_val > w_val) return false;
			v_val *= 3;
			w_val *= 2;
		}
		return v_val > w_val;
	}

	static sol (v, w) {
		let v_val = v;
		let w_val = w;
		let i = 0;
		while (v_val <= w_val) {
			v_val *= 3;
			w_val *= 2;
			i += 1;
		}
		return i;
	}

	genRandom () {
		const w = this.random.randrange(2, 10 ** 9);
		const v = this.random.randrange(1, w);
		this.add({ v, w });
	}
}

export class RepeatDec extends PuzzleGenerator {
	static docstring = "Find the result of applying the following operation to integer m, n times: if the last digit is zero, remove the zero, otherwise subtract 1.";

	getExample () {
		return { m: 1234578987654321, n: 4 };
	}

	/**
	 * Find the result of applying the following operation to integer m, n times: if the last digit is zero, remove the zero, otherwise subtract 1.
	 */
	static sat (answer, m, n) {
		let m_val = m;
		for (let i = 0; i < n; i++) {
			m_val = m_val % 10 === 0 ? Math.floor(m_val / 10) : m_val - 1;
		}
		return answer === m_val;
	}

	static sol (m, n) {
		let m_val = m;
		for (let i = 0; i < n; i++) {
			m_val = m_val % 10 === 0 ? Math.floor(m_val / 10) : m_val - 1;
		}
		return m_val;
	}

	genRandom () {
		const m = this.random.randrange(2, Number.MAX_SAFE_INTEGER);
		const n = this.random.randrange(1, 10);
		this.add({ m, n });
	}
}

export class ShortestDecDelta extends PuzzleGenerator {
	static docstring = "Find a the shortest sequence of integers going from 1 to n where each difference is at most 10. Do not include 1 or n in the sequence.";

	getExample () {
		return { n: 149432, upper: 14943 };
	}

	/**
	 * Find a the shortest sequence of integers going from 1 to n where each difference is at most 10. Do not include 1 or n in the sequence.
	 */
	static sat (answer, n, upper) {
		if (!Array.isArray(answer) || answer.length > upper) return false;
		const seq = [1, ...answer, n];
		for (let i = 0; i < seq.length - 1; i++) {
			if (Math.abs(seq[i + 1] - seq[i]) > 10) return false;
		}
		return true;
	}

	static sol (n, upper) {
		let m = 1;
		const ans = [];
		while (true) {
			m = Math.min(n, m + 10);
			if (m >= n) return ans;
			ans.push(m);
		}
	}

	genRandom () {
		const n = this.random.randrange(1, 10 ** 6);
		const ans = this.sol({ n, upper: null });
		this.add({ n, upper: ans.length });
	}
}

export class MaxDelta extends PuzzleGenerator {
	static docstring = "Given a sequence of integer pairs, p_i, m_i, where \\sum p_i-m_i = 0, find the maximum value, over t, of p_{t+1} + \\sum_{i=1}^t p_i - m_i";

	getExample () {
		return {
			pairs: [
				[3, 0],
				[17, 1],
				[9254359, 19],
				[123, 9254359],
				[0, 123],
			],
		};
	}

	/**
	 * Given a sequence of integer pairs, p_i, m_i, where \sum p_i-m_i = 0, find the maximum value, over t, of p_{t+1} + \sum_{i=1}^t p_i - m_i
	 */
	static sat (answer, pairs) {
		if (typeof answer !== "number") return false;
		let tot = 0;
		let success = false;
		for (const [p, m] of pairs) {
			tot = tot + p - m;
			if (tot > answer) return false;
			if (tot === answer) success = true;
		}
		return success;
	}

	static sol (pairs) {
		let tot = 0;
		let n = 0;
		for (const [p, m] of pairs) {
			tot += p - m;
			if (tot > n) n = tot;
		}
		return n;
	}

	genRandom () {
		let tot = 0;
		const pairs = [];
		while (this.random.randrange(10)) {
			const m = this.random.randrange(tot + 1);
			const p = this.random.randrange(10 ** 6);
			tot += p - m;
			pairs.push([p, m]);
		}
		pairs.push([0, tot]);
		this.add({ pairs });
	}
}

export class CommonCase extends PuzzleGenerator {
	static docstring = "Given a word, replace it either with an upper-case or lower-case depending on whether or not it has more capitals or lower-case letters. If it has strictly more capitals, use upper-case, otherwise, use lower-case.";

	getExample () {
		return { s: "CanYouTellIfItHASmoreCAPITALS" };
	}

	/**
	 * Given a word, replace it either with an upper-case or lower-case depending on whether or not it has more capitals or lower-case letters. If it has strictly more capitals, use upper-case, otherwise, use lower-case.
	 */
	static sat (answer, s) {
		let caps = 0;
		for (const c of s) {
			if (c !== c.toLowerCase()) caps++;
		}
		return (
			answer ===
			(caps > Math.floor(s.length / 2) ? s.toUpperCase() : s.toLowerCase())
		);
	}

	static sol (s) {
		let caps = 0;
		for (const c of s) {
			if (c !== c.toLowerCase()) caps++;
		}
		return caps > Math.floor(s.length / 2) ? s.toUpperCase() : s.toLowerCase();
	}

	genRandom () {
		const s = Array.from({ length: this.random.randrange(5, 30) }, () => {
			const c = this.random.char();
			return this.random.random() > 0.5 ? c.toUpperCase() : c.toLowerCase();
		}).join("");
		this.add({ s });
	}
}

export class Sssuubbstriiingg extends PuzzleGenerator {
	static docstring = "Find the indices of the characters that form the word 'substring' in the given string.";

	getExample () {
		return { string: "Sssuubbstrissiingg" };
	}

	static sat (answer, string) {
		if (!Array.isArray(answer)) return false;
		const sorted = answer.slice().sort((a, b) => a - b);
		if (JSON.stringify(answer) !== JSON.stringify(sorted)) return false;
		const target = "substring";
		return answer.map((i) => string[i]).join("") === target;
	}

	static sol (string) {
		const target = "substring";
		let j = 0;
		const ans = [];
		for (let i = 0; i < string.length && j < target.length; i++) {
			if (string[i] === target[j]) {
				ans.push(i);
				j++;
			}
		}
		return j === target.length ? ans : null;
	}

	genRandom () {
		const chars = Array.from("substring");
		for (let i = 0; i < this.random.randrange(20); i++) {
			const idx = this.random.randrange(chars.length + 1);
			const ch = this.random.choice(
				"   abcdefghijklmnopqrstuvwxyz    ABCDEFGHIJKLMNOPQRSTUVWXYZ  ",
			);
			chars.splice(idx, 0, ch);
		}
		const string = chars.join("");
		this.add({ string });
	}
}

export class Sstriiinggssuubb extends PuzzleGenerator {
	static docstring = "Find indices (possibly negative) that form the word 'intelligent' from the given string.";

	getExample () {
		return { string: "enlightenment" };
	}

	static sat (answer, string) {
		if (!Array.isArray(answer)) return false;
		const target = "intelligent";
		let j = 0;
		for (const idx of answer) {
			const char = idx < 0 ? string[string.length + idx] : string[idx];
			if (char !== target[j]) return false;
			j++;
		}
		return j === target.length;
	}

	static sol (string) {
		const target = "intelligent";
		let j = 0;
		const ans = [];
		for (let i = -string.length; i < string.length && j < target.length; i++) {
			const char = i < 0 ? string[string.length + i] : string[i];
			if (char === target[j]) {
				ans.push(i);
				j++;
			}
		}
		return j === target.length ? ans : null;
	}

	genRandom () {
		const chars = Array.from("intelligent");
		const i = this.random.randrange(chars.length);
		const a = chars.slice(0, i).reverse();
		const b = chars.slice(i).reverse();
		const mixed = [];
		while (a.length > 0 || b.length > 0) {
			if (a.length > 0 && b.length > 0) {
				mixed.push(this.random.choice([a, b]).pop());
			} else {
				mixed.push((a.length > 0 ? a : b).pop());
			}
		}
		for (let j = 0; j < this.random.randrange(20); j++) {
			const idx = this.random.randrange(mixed.length + 1);
			const ch = this.random.choice(
				"   abcdefghijklmnopqrstuvwxyz    ABCDEFGHIJKLMNOPQRSTUVWXYZ  ",
			);
			mixed.splice(idx, 0, ch);
		}
		const string = mixed.join("");
		this.add({ string });
	}
}

export class Moving0s extends PuzzleGenerator {
	static docstring = "Find a sequence of 0's and 1's so that, after n_steps of swapping each adjacent (0, 1), the target sequence is achieved.";

	getExample () {
		return { target: [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], n_steps: 4 };
	}

	static sat (answer, target, n_steps) {
		if (!Array.isArray(answer)) return false;
		const s = answer.slice();
		for (let step = 0; step < n_steps; step++) {
			for (let i = 0; i < s.length - 1; i++) {
				if (s[i] === 0 && s[i + 1] === 1) {
					[s[i], s[i + 1]] = [s[i + 1], s[i]];
				}
			}
		}
		return JSON.stringify(s) === JSON.stringify(target);
	}

	static sol (target, n_steps) {
		const s = target.slice();
		for (let step = 0; step < n_steps; step++) {
			for (let i = s.length - 2; i >= 0; i--) {
				if (s[i] === 1 && s[i + 1] === 0) {
					[s[i], s[i + 1]] = [s[i + 1], s[i]];
				}
			}
		}
		return s;
	}

	genRandom () {
		const seq = Array.from({ length: this.random.randrange(3, 20) }, () =>
			this.random.randrange(2),
		);
		const n_steps = this.random.randrange(seq.length);
		const target = seq.slice();
		for (let step = 0; step < n_steps; step++) {
			for (let i = 0; i < target.length - 1; i++) {
				if (target[i] === 0 && target[i + 1] === 1) {
					[target[i], target[i + 1]] = [target[i + 1], target[i]];
				}
			}
		}
		this.add({ target, n_steps });
	}
}

export class Factor47 extends PuzzleGenerator {
	static docstring = "Find a divisor of n that consists only of the digits 4 and 7.";

	getExample () {
		return { n: 6002685529 };
	}

	static sat (answer, n) {
		if (n % answer !== 0) return false;
		return String(answer)
			.split("")
			.every((c) => c === "4" || c === "7");
	}

	static sol (n) {
		const helper = (soFar, k) => {
			if (k > 0) {
				return helper(soFar * 10 + 4, k - 1) || helper(soFar * 10 + 7, k - 1);
			}
			return n % soFar === 0 ? soFar : null;
		};
		for (
			let length = 1;
			length <= Math.floor(String(n).length / 2) + 1;
			length++
		) {
			const ans = helper(0, length);
			if (ans) return ans;
		}
		return null;
	}

	genRandom () {
		const length = this.random.randrange(1, 8);
		const d = parseInt(
			Array.from({ length }, () => this.random.choice("47")).join(""),
		);
		const n = this.random.randrange(1, 10 ** length) * d;
		if (this.sol({ n }) === d) {
			this.add({ n });
		}
	}
}

export class Count47 extends PuzzleGenerator {
	static docstring = "Find a number greater than n where the count of 4s and 7s in the number consists only of the digits 4 and 7.";

	getExample () {
		return { n: 123456789 };
	}

	static sat (answer, n) {
		if (answer <= n) return false;
		const count47 = String(answer)
			.split("")
			.filter((c) => c === "4" || c === "7").length;
		return String(count47)
			.split("")
			.every((c) => c === "4" || c === "7");
	}

	static sol (n) {
		return parseInt("4444" + "0".repeat(Math.max(0, String(n).length - 3)));
	}

	genRandom () {
		const n = this.random.randrange(10 ** this.random.randrange(2, 30));
		this.add({ n });
	}
}

export class MaybeReversed extends PuzzleGenerator {
	static docstring = "Determine if the string is reversed or not based on the reverse flag.";

	getExample () {
		return { target: "reverse me", reverse: true };
	}

	static sat (answer, target, reverse) {
		const isReversed = answer === target.split("").reverse().join("");
		return isReversed === reverse;
	}

	static sol (target, reverse) {
		return reverse ? target.split("").reverse().join("") : target + "x";
	}

	genRandom () {
		const reverse = this.random.choice([true, false]);
		const target = this.random.pseudo_word();
		this.add({ target, reverse });
	}
}

export class MinBigger extends PuzzleGenerator {
	static docstring = "Select counts for each value to have a total advantage greater than 0, with total selections <= upper.";

	getExample () {
		return {
			val_counts: [
				[4, 3],
				[5, 2],
				[9, 3],
				[13, 13],
				[8, 11],
				[56, 1],
			],
			upper: 11,
		};
	}

	static sat (answer, val_counts, upper) {
		if (!Array.isArray(answer) || answer.length !== val_counts.length)
			return false;
		if (answer.reduce((a, b) => a + b, 0) > upper) return false;
		let advantage = 0;
		for (let i = 0; i < answer.length; i++) {
			const [val, count] = val_counts[i];
			if (answer[i] < 0 || answer[i] > count) return false;
			advantage += val * answer[i] - (val * count) / 2;
		}
		return advantage > 0;
	}

	static sol (val_counts, upper) {
		const n = val_counts.length;
		const pi = Array.from({ length: n }, (_, i) => i).sort(
			(a, b) => val_counts[a][0] - val_counts[b][0],
		);
		let needed = val_counts.reduce((s, [a, b]) => s + a * b, 0) / 2 + 0.1;
		const ans = Array(n).fill(0);
		while (needed > 0) {
			while (val_counts[pi[pi.length - 1]][1] === ans[pi[pi.length - 1]]) {
				pi.pop();
			}
			const i = pi[pi.length - 1];
			ans[i] += 1;
			needed -= val_counts[i][0];
		}
		return ans;
	}

	genRandom () {
		const val_counts = Array.from(
			{ length: this.random.randrange(1, 10) },
			() => [this.random.randrange(1, 100), this.random.randrange(1, 100)],
		);
		const upper = this.sol({ val_counts, upper: null }).reduce(
			(a, b) => a + b,
			0,
		);
		this.add({ val_counts, upper });
	}
}

export class Dada extends PuzzleGenerator {
	static docstring = "Find a string with a given number of a's and d's";

	getExample () {
		return { a: 5129, d: 17 };
	}

	static sat (answer, a, d) {
		return (
			answer.split("").filter((c) => c === "a").length === a &&
			answer.split("").filter((c) => c === "d").length === d &&
			answer.length === a + d
		);
	}

	static sol (a, d) {
		return "a".repeat(a) + "d".repeat(d);
	}

	genRandom () {
		const a = this.random.randrange(10 ** 4);
		const d = this.random.randrange(10 ** 4);
		this.add({ a, d });
	}
}

export class DistinctDigits extends PuzzleGenerator {
	static docstring = "Find a list of count or more different numbers each between a and b that each have no repeated digits";

	getExample () {
		return { a: 100, b: 1000, count: 648 };
	}

	static sat (answer, a, b, count) {
		const unique = new Set(answer);
		if (unique.size < count) return false;
		return answer.every((n) => {
			const s = String(n);
			return s.length === new Set(s).size && n >= a && n <= b;
		});
	}

	static sol (a, b, count) {
		const ans = [];
		for (let n = a; n <= b; n++) {
			const s = String(n);
			if (s.length === new Set(s).size) ans.push(n);
		}
		return ans;
	}

	genRandom () {
		const b = this.random.randrange(1, 10 ** 3);
		const a = this.random.randrange(b);
		const sol = this.sol({ a, b, count: null });
		this.add({ a, b, count: sol.length });
	}
}

export class EasySum extends PuzzleGenerator {
	static docstring = "Sum 1 for each number less than thresh, 2 for each number greater than or equal to thresh.";

	getExample () {
		return { nums: [2, 8, 25, 18, 99, 11, 17, 16], thresh: 17 };
	}

	static sat (answer, nums, thresh) {
		const expected = nums.reduce((sum, n) => sum + (n < thresh ? 1 : 2), 0);
		return answer === expected;
	}

	static sol (nums, thresh) {
		return nums.reduce((sum, n) => sum + (n < thresh ? 1 : 2), 0);
	}

	genRandom () {
		const nums = Array.from({ length: this.random.randrange(30) }, () =>
			this.random.randrange(100),
		);
		const thresh = this.random.randrange(1, 100);
		this.add({ nums, thresh });
	}
}

export class GimmeChars extends PuzzleGenerator {
	static docstring = "Find a string with certain characters";

	getExample () {
		return { chars: ["o", "h", "e", "l", " ", "w", "!", "r", "d"] };
	}

	static sat (answer, chars) {
		for (const c of chars) {
			if (!answer.includes(c)) return false;
		}
		return true;
	}

	static sol (chars) {
		return chars.join("");
	}

	genRandom () {
		const chars = Array.from({ length: this.random.choice([3, 5, 10]) }, () =>
			this.random.char(),
		);
		this.add({ chars });
	}
}

export class HalfPairs extends PuzzleGenerator {
	static docstring = "Find a list of pairs of integers where the number of pairs in which the second number is more than two greater than the first number is a given constant";

	getExample () {
		return { target: 17 };
	}

	static sat (answer, target) {
		let count = target;
		for (const [a, b] of answer) {
			if (b - a >= 2) count--;
		}
		return count === 0;
	}

	static sol (target) {
		return Array.from({ length: target }, () => [0, 2]);
	}

	gen (target_num_instances) {
		let target = 0;
		while (this.numGeneratedSoFar() < target_num_instances) {
			this.add({ target });
			target += 1;
		}
	}
}

export class InvertIndices extends PuzzleGenerator {
	static docstring = "Given a list of integers representing a permutation, invert the permutation.";

	getExample () {
		return { target: [1, 3, 4, 2, 5, 6, 7, 13, 12, 11, 9, 10, 8] };
	}

	static sat (answer, target) {
		for (let i = 1; i <= target.length; i++) {
			if (target[answer[i - 1] - 1] !== i) return false;
		}
		return true;
	}

	static sol (target) {
		const ans = Array(target.length);
		for (let i = 0; i < target.length; i++) {
			ans[target[i] - 1] = i + 1;
		}
		return ans;
	}

	genRandom () {
		const n = this.random.randrange(1, 100);
		const target = Array.from({ length: n }, (_, i) => i + 1);
		this.random.shuffle(target);
		this.add({ target });
	}
}

export class FivePowers extends PuzzleGenerator {
	static docstring = "What are the last two digits of 5^n?";

	getExample () {
		return { n: 7012 };
	}

	static sat (answer, n) {
		if (n <= 10) {
			const fullStr = String(5 ** n);
			return fullStr === fullStr.slice(0, -2) + answer;
		}
		// For large n, just check pattern: 5^n always ends in 25 for n >= 2
		return answer === "25";
	}

	static sol (n) {
		if (n === 0) return "1";
		if (n === 1) return "5";
		return "25";
	}

	gen (target_num_instances) {
		for (let n = 0; n < target_num_instances; n++) {
			this.add({ n });
		}
	}
}

export class CombinationLock extends PuzzleGenerator {
	static docstring = "Shortest Combination Lock Path\n\nGiven a starting a final lock position, find the (minimal) intermediate states, where each transition involves increasing or decreasing a single digit (mod 10).";

	getExample () {
		return { start: "424", combo: "778", target_len: 12 };
	}

	static sat (answer, start, combo, target_len) {
		if (!Array.isArray(answer)) return false;
		if (answer.length > target_len) return false;
		if (!answer.every((s) => s.length === start.length && /^\d+$/.test(s)))
			return false;
		const fullSeq = [start, ...answer, combo];
		for (let idx = 0; idx < fullSeq.length - 1; idx++) {
			const a = fullSeq[idx];
			const b = fullSeq[idx + 1];
			if (a.length !== b.length) return false;
			let diffs = 0;
			for (let i = 0; i < a.length; i++) {
				const diff = Math.abs(parseInt(a[i]) - parseInt(b[i]));
				if (![0, 1, 9].includes(diff)) return false;
				if (diff === 1 || diff === 9) diffs++;
			}
			if (diffs !== 1) return false;
		}
		return true;
	}

	static sol (start, combo, target_len) {
		const n = start.length;
		const ans = [];
		const a = start.split("").map((c) => parseInt(c));
		const b = combo.split("").map((c) => parseInt(c));
		for (let i = 0; i < n; i++) {
			while (a[i] !== b[i]) {
				const diff = (a[i] - b[i] + 10) % 10;
				a[i] = diff < 5 ? (a[i] - 1 + 10) % 10 : (a[i] + 1) % 10;
				if (JSON.stringify(a) !== JSON.stringify(b)) {
					ans.push(a.join(""));
				}
			}
		}
		return ans;
	}

	genRandom () {
		const n = this.random.randrange(1, 11);
		const start = Array.from({ length: n }, () =>
			String(this.random.randrange(10)),
		).join("");
		const combo = Array.from({ length: n }, () =>
			String(this.random.randrange(10)),
		).join("");
		if (start !== combo) {
			const sol = this.sol({ start, combo, target_len: null });
			this.add({ start, combo, target_len: sol.length });
		}
	}
}

export class InvertPermutation extends PuzzleGenerator {
	static docstring = "Find a string that, when a given permutation of characters is applied, has a given result.";

	getExample () {
		return {
			perm: "qwertyuiopasdfghjklzxcvbnm",
			target: "hello are you there?",
		};
	}

	static sat (answer, perm, target) {
		let result = "";
		for (const c of answer) {
			if (perm.includes(c)) {
				const idx = perm.indexOf(c);
				result += perm[(idx + 1) % perm.length];
			} else {
				result += c;
			}
		}
		return result === target;
	}

	static sol (perm, target) {
		let result = "";
		for (const c of target) {
			if (perm.includes(c)) {
				const idx = perm.indexOf(c);
				result += perm[(idx - 1 + perm.length) % perm.length];
			} else {
				result += c;
			}
		}
		return result;
	}

	genRandom () {
		const perm = "qwertyuiopasdfghjklzxcvbnm";
		const target = Array.from({ length: this.random.randrange(1, 10) }, () =>
			this.random.pseudo_word(),
		).join(" ");
		this.add({ perm, target });
	}
}

export class CombinationLockObfuscated extends CombinationLock {
	static docstring = "Figure out what this does only from the code";

	getExample () {
		return { start: "012", combo: "329", target_len: 6 };
	}

	static sat (answer, start, combo, target_len) {
		if (!Array.isArray(answer)) return false;
		if (answer.length > target_len) return false;
		const fullSeq = [start, ...answer, combo];
		for (let idx = 0; idx < fullSeq.length - 1; idx++) {
			const a = fullSeq[idx];
			const b = fullSeq[idx + 1];
			if (a.length !== b.length) return false;
			let sum = 0;
			for (let i = 0; i < a.length; i++) {
				sum += (parseInt(a[i]) - parseInt(b[i])) ** 2 % 10;
			}
			if (sum !== 1) return false;
		}
		return true;
	}

	static sol (start, combo, target_len) {
		return CombinationLock.sol(start, combo, target_len);
	}
}

export class SameDifferent extends PuzzleGenerator {
	static docstring = "Given a list of integers and a target length, create of the given length such that:\n            * The first list must be all different numbers.\n            * The second must be all the same number.\n            * The two lists together comprise a sublist of all the list items";

	getExample () {
		return { items: [5, 4, 9, 4, 5, 5, 5, 1, 5, 5], length: 4 };
	}

	static sat (answer, items, length) {
		if (!Array.isArray(answer) || answer.length !== 2) return false;
		const [a, b] = answer;
		if (!Array.isArray(a) || !Array.isArray(b)) return false;
		if (a.length !== length || b.length !== length) return false;
		if (new Set(a).size !== a.length) return false;
		if (new Set(b).size !== 1) return false;
		const combined = [...a, ...b];
		for (const elem of combined) {
			const countInResult = combined.filter((x) => x === elem).length;
			const countInItems = items.filter((x) => x === elem).length;
			if (countInResult > countInItems) return false;
		}
		return true;
	}

	static sol (items, length) {
		const freq = {};
		for (const item of items) freq[item] = (freq[item] || 0) + 1;
		const [[a, count]] = Object.entries(freq).sort((x, y) => y[1] - x[1]);
		const seen = new Set([a]);
		const dedup = [];
		for (const i of items) {
			if (!seen.has(i)) {
				dedup.push(i);
				seen.add(i);
			}
		}
		const firstList = dedup.concat(Array(length).fill(a)).slice(0, length);
		const secondList = Array(length).fill(parseInt(a));
		return [firstList, secondList];
	}

	genRandom () {
		const items = Array.from({ length: this.random.randrange(5, 100) }, () =>
			this.random.randrange(10),
		);
		const freq = {};
		for (const item of items) freq[item] = (freq[item] || 0) + 1;
		const counts = Object.values(freq).sort((a, b) => b - a);
		const maxCount = counts[0];
		const numUnique = Object.keys(freq).length;
		const length =
			maxCount === numUnique ? maxCount - 1 : Math.min(maxCount, numUnique);
		this.add({ items, length });
	}
}

export class OnesAndTwos extends PuzzleGenerator {
	static docstring = "Find a sequence of 1's and 2's of a given length that that adds up to n";

	getExample () {
		return { n: 10000, length: 5017 };
	}

	static sat (answer, n, length) {
		if (!Array.isArray(answer) || answer.length !== length) return false;
		if (!answer.every((i) => i === 1 || i === 2)) return false;
		return answer.reduce((a, b) => a + b, 0) === n;
	}

	static sol (n, length) {
		const numTwos = n - length;
		const numOnes = 2 * length - n;
		return Array(numTwos).fill(2).concat(Array(numOnes).fill(1));
	}

	genRandom () {
		const n = this.random.randrange(10 ** this.random.randrange(5));
		const length = this.random.randrange(Math.floor((n + 1) / 2), n + 1);
		this.add({ n, length });
	}
}

export class MinConsecutiveSum extends PuzzleGenerator {
	static docstring = "Find a sequence of k consecutive indices whose sum is minimal";

	getExample () {
		return { k: 3, upper: 6, seq: [17, 1, 2, 65, 18, 91, -30, 100, 3, 1, 2] };
	}

	static sat (answer, k, upper, seq) {
		if (typeof answer !== "number" || answer < 0 || answer + k > seq.length)
			return false;
		const sum = seq.slice(answer, answer + k).reduce((a, b) => a + b, 0);
		return sum <= upper;
	}

	static sol (k, upper, seq) {
		let minSum = Infinity;
		let minIdx = 0;
		for (let start = 0; start <= seq.length - k; start++) {
			const sum = seq.slice(start, start + k).reduce((a, b) => a + b, 0);
			if (sum < minSum) {
				minSum = sum;
				minIdx = start;
			}
		}
		return minIdx;
	}

	genRandom () {
		const k = this.random.randrange(1, 11);
		const n = this.random.randrange(k, k + 10 ** this.random.randrange(3));
		const seq = Array.from({ length: n }, () =>
			this.random.randrange(-100, 100),
		);
		let minSum = Infinity;
		for (let start = 0; start <= seq.length - k; start++) {
			const sum = seq.slice(start, start + k).reduce((a, b) => a + b, 0);
			if (sum < minSum) minSum = sum;
		}
		this.add({ k, upper: minSum, seq });
	}
}

export class MaxConsecutiveSum extends PuzzleGenerator {
	static docstring = "Find a sequence of k consecutive indices whose sum is maximal";

	getExample () {
		return {
			k: 3,
			lower: 150,
			seq: [3, 1, 2, 65, 18, 91, -30, 100, 0, 19, 52],
		};
	}

	static sat (answer, k, lower, seq) {
		if (typeof answer !== "number" || answer < 0 || answer + k > seq.length)
			return false;
		const sum = seq.slice(answer, answer + k).reduce((a, b) => a + b, 0);
		return sum >= lower;
	}

	static sol (k, lower, seq) {
		let maxSum = -Infinity;
		let maxIdx = 0;
		for (let start = 0; start <= seq.length - k; start++) {
			const sum = seq.slice(start, start + k).reduce((a, b) => a + b, 0);
			if (sum > maxSum) {
				maxSum = sum;
				maxIdx = start;
			}
		}
		return maxIdx;
	}

	genRandom () {
		const k = this.random.randrange(1, 11);
		const n = this.random.randrange(k, k + 10 ** this.random.randrange(3));
		const seq = Array.from({ length: n }, () =>
			this.random.randrange(-100, 100),
		);
		let maxSum = -Infinity;
		for (let start = 0; start <= seq.length - k; start++) {
			const sum = seq.slice(start, start + k).reduce((a, b) => a + b, 0);
			if (sum > maxSum) maxSum = sum;
		}
		this.add({ k, lower: maxSum, seq });
	}
}

export class MaxConsecutiveProduct extends PuzzleGenerator {
	static docstring = "Find a sequence of k consecutive indices whose product is maximal, possibly looping around";

	getExample () {
		return {
			k: 3,
			lower: 100000,
			seq: [91, 1, 2, 64, 18, 91, -30, 100, 3, 65, 18],
		};
	}

	static sat (answer, k, lower, seq) {
		let prod = 1;
		for (let i = answer; i < answer + k; i++) {
			prod *= seq[((i % seq.length) + seq.length) % seq.length];
		}
		return prod >= lower;
	}

	static sol (k, lower, seq) {
		const prod = (start) => {
			let ans = 1;
			for (let i = start; i < start + k; i++) {
				ans *= seq[((i % seq.length) + seq.length) % seq.length];
			}
			return ans;
		};
		let maxProd = -Infinity;
		let maxIdx = 0;
		for (let i = -seq.length; i < seq.length - k + 1; i++) {
			const p = prod(i);
			if (p > maxProd) {
				maxProd = p;
				maxIdx = i;
			}
		}
		return maxIdx;
	}

	genRandom () {
		const k = this.random.randrange(1, 11);
		const n = this.random.randrange(k, k + 10 ** this.random.randrange(3));
		const seq = Array.from({ length: n }, () =>
			this.random.randrange(-100, 100),
		);
		const prod = (start) => {
			let ans = 1;
			for (let i = start; i < start + k; i++) {
				ans *= seq[((i % seq.length) + seq.length) % seq.length];
			}
			return ans;
		};
		let maxProd = -Infinity;
		for (let i = -seq.length; i < seq.length - k + 1; i++) {
			const p = prod(i);
			if (p > maxProd) maxProd = p;
		}
		this.add({ k, lower: maxProd, seq });
	}
}

export class DistinctOddSum extends PuzzleGenerator {
	static docstring = "Find n distinct positive odd integers that sum to tot";

	getExample () {
		return { tot: 12345, n: 5 };
	}

	static sat (answer, tot, n) {
		if (!Array.isArray(answer) || answer.length !== n) return false;
		if (new Set(answer).size !== n) return false;
		if (answer.reduce((a, b) => a + b, 0) !== tot) return false;
		return answer.every((i) => i > 0 && i % 2 === 1);
	}

	static sol (tot, n) {
		const odds = Array.from({ length: n - 1 }, (_, i) => 2 * i + 1);
		const remaining = tot - odds.reduce((a, b) => a + b, 0);
		return [...odds, remaining];
	}

	genRandom () {
		const n = this.random.randrange(1, 100);
		const odds = [];
		for (let i = 0; i < n; i++) {
			let odd;
			do {
				odd = this.random.randrange(1, 1000) * 2 + 1;
			} while (odds.includes(odd));
			odds.push(odd);
		}
		const tot = odds.reduce((a, b) => a + b, 0);
		this.add({ tot, n });
	}
}

export class MinRotations extends PuzzleGenerator {
	static docstring = "We begin with the string `\"a...z\"`\n\n        An `r`-rotation of a string means shifting it to the right (positive) or left (negative) by `r` characters and\n        cycling around. Given a target string of length n, find the n rotations that put the consecutive characters\n        of that string at the beginning of the r-rotation, with minimal sum of absolute values of the `r`'s.\n\n        For example if the string was `'dad'`, the minimal rotations would be `[3, -3, 3]` with a total of `9`.";

	getExample () {
		return { target: "wonderful", upper: 69 };
	}

	static sat (answer, target, upper) {
		if (!Array.isArray(answer) || answer.length !== target.length) return false;
		let s = "abcdefghijklmnopqrstuvwxyz";
		for (let i = 0; i < target.length; i++) {
			const r = answer[i];
			s = s.slice(r) + s.slice(0, r);
			if (s[0] !== target[i]) return false;
		}
		return answer.reduce((sum, r) => sum + Math.abs(r), 0) <= upper;
	}

	static sol (target, upper) {
		let s = "abcdefghijklmnopqrstuvwxyz";
		const ans = [];
		for (const c of target) {
			const i = s.indexOf(c);
			const r = Math.abs(i) < Math.abs(i - s.length) ? i : i - s.length;
			ans.push(r);
			s = s.slice(r) + s.slice(0, r);
		}
		return ans;
	}

	genRandom () {
		const target = this.random.pseudo_word();
		const upper = this.sol({ target, upper: null }).reduce(
			(a, b) => a + Math.abs(b),
			0,
		);
		this.add({ target, upper });
	}
}

export class BillSums extends PuzzleGenerator {
	static docstring = "Find the shortest sequence (length <= max_len) that sum to n, where each number is in denominations";

	getExample () {
		return { denominations: [1, 25, 35, 84], n: 980, max_len: 14 };
	}

	static sat (answer, denominations, n, max_len) {
		if (!Array.isArray(answer)) return false;
		if (answer.length > max_len) return false;
		if (answer.reduce((a, b) => a + b, 0) !== n) return false;
		return answer.every((b) => denominations.includes(b));
	}

	static sol (denominations, n, max_len) {
		const denoms = Array.from(new Set(denominations)).sort((a, b) => a - b);
		const seqs = [[...Array(denoms.length).fill(0), 0]];
		for (let i = 1; i <= n; i++) {
			let minCount = Infinity;
			let minIdx = 0;
			let minDenom = 0;
			for (let j = 0; j < denoms.length; j++) {
				const k = denoms[j];
				if (k <= i) {
					const prev = seqs[i - k];
					if (prev[prev.length - 1] < minCount) {
						minCount = prev[prev.length - 1];
						minIdx = j;
						minDenom = k;
					}
				}
			}
			const prevSeq = seqs[i - minDenom];
			const newSeq = prevSeq.slice();
			newSeq[minIdx] = newSeq[minIdx] + 1;
			newSeq[newSeq.length - 1] = newSeq[newSeq.length - 1] + 1;
			seqs.push(newSeq);
		}
		const lastSeq = seqs[n];
		const result = [];
		for (let j = 0; j < denoms.length; j++) {
			for (let count = 0; count < lastSeq[j]; count++) {
				result.push(denoms[j]);
			}
		}
		return result;
	}

	genRandom () {
		const denomSet = new Set([1]);
		for (let i = 0; i < this.random.randrange(6); i++) {
			denomSet.add(this.random.randrange(2, this.random.choice([10, 20])));
		}
		const denominations = Array.from(denomSet).sort((a, b) => a - b);
		const n = this.random.randrange(1, 1000);
		const sol = this.sol({ denominations, n, max_len: null });
		this.add({ denominations, n, max_len: sol.length });
	}
}

export class BoxVolume extends PuzzleGenerator {
	static docstring = "Find the side lengths of a box in fewest dimensions (dimension <= max_dim) whose volume is n,\n         where each side length is in options";

	getExample () {
		return {
			options: [2, 512, 1024],
			n: 340282366920938463463374607431768211456,
			max_dim: 13,
		};
	}

	static sat (answer, options, n, max_dim) {
		if (!Array.isArray(answer)) return false;
		if (answer.length > max_dim) return false;
		let prod = 1;
		for (const b of answer) prod *= b;
		if (prod !== n) return false;
		return answer.every((a) => options.includes(a));
	}

	static sol (options, n, max_dim) {
		const opts = Array.from(new Set(options)).sort((a, b) => a - b);
		const base = opts[0];
		const logs = [];
		for (const i of [...opts, n]) {
			let j = 1;
			let log = 0;
			while (j < i) {
				log++;
				j *= base;
			}
			if (j !== i)
				throw new Error("All numbers must be a power of the smallest");
			logs.push(log);
		}
		const denominations = logs.slice(0, -1);
		const targetN = logs[logs.length - 1];
		const seqs = [[...Array(denominations.length).fill(0), 0]];
		for (let i = 1; i <= targetN; i++) {
			let minCount = Infinity;
			let minIdx = 0;
			let minDenom = 0;
			for (let j = 0; j < denominations.length; j++) {
				const k = denominations[j];
				if (k <= i) {
					const prev = seqs[i - k];
					if (prev[prev.length - 1] < minCount) {
						minCount = prev[prev.length - 1];
						minIdx = j;
						minDenom = k;
					}
				}
			}
			const prevSeq = seqs[i - minDenom];
			const newSeq = prevSeq.slice();
			newSeq[minIdx] = newSeq[minIdx] + 1;
			newSeq[newSeq.length - 1] = newSeq[newSeq.length - 1] + 1;
			seqs.push(newSeq);
		}
		const lastSeq = seqs[targetN];
		const result = [];
		for (let j = 0; j < denominations.length; j++) {
			for (let count = 0; count < lastSeq[j]; count++) {
				result.push(opts[j]);
			}
		}
		return result;
	}

	genRandom () {
		const base = this.random.choice([2, 3, 5, 7]);
		const exp = this.random.randrange(1, 20);
		const n = base ** exp;
		if (n < 10 ** 100) {
			const denomSet = new Set([1]);
			for (let i = 0; i < this.random.randrange(6); i++) {
				denomSet.add(this.random.randrange(2, this.random.choice([10, 20])));
			}
			const denominations = Array.from(denomSet).sort((a, b) => a - b);
			const options = denominations.map((d) => base ** d);
			const sol = this.sol({ options, n, max_dim: null });
			this.add({ options, n, max_dim: sol.length });
		}
	}
}
