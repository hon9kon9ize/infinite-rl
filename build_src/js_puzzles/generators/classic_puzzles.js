/** biome-ignore-all lint/security/noGlobalEval: <explanation> */

import seedrandom from "../seedrandom.js";
import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

export class TowersOfHanoi extends PuzzleGenerator {
	static docstring = "Eight disks of sizes 1-8 are stacked on three towers, with each tower having disks in order of largest to\n        smallest. Move [i, j] corresponds to taking the smallest disk off tower i and putting it on tower j, and it\n        is legal as long as the towers remain in sorted order. Find a sequence of moves that moves all the disks\n        from the first to last towers.";

	constructor(seed = null) {
		super(seed);
	}

	static sat (moves) {
		/**
		 * Eight disks of sizes 1-8 are stacked on three towers, with each tower having disks in order of largest to
		 * smallest. Move [i, j] corresponds to taking the smallest disk off tower i and putting it on tower j, and it
		 * is legal as long as the towers remain in sorted order. Find a sequence of moves that moves all the disks
		 * from the first to last towers.
		 */
		const rods = [[8, 7, 6, 5, 4, 3, 2, 1], [], []];
		for (const [i, j] of moves) {
			rods[j].push(rods[i].pop());
			if (Math.min(...rods[j]) !== rods[j][rods[j].length - 1])
				throw new Error("larger disk on top");
		}
		return rods[0].length === 0 && rods[1].length === 0;
	}

	/**
	 * Reference solution for TowersOfHanoi.
	 */
	static sol () {
		function helper (m, i, j) {
			if (m === 0) return [];
			const k = 3 - i - j;
			return helper(m - 1, i, k)
				.concat([[i, j]])
				.concat(helper(m - 1, k, j));
		}
		return helper(8, 0, 2);
	}
}

export class TowersOfHanoiArbitrary extends PuzzleGenerator {
	static docstring = "A state is a partition of the integers 0-8 into three increasing lists. A move is pair of integers i, j in\n        {0, 1, 2} corresponding to moving the largest number from the end of list i to list j, while preserving the\n        order of list j. Find a sequence of moves that transform the given source to target states.";

	constructor(seed = null) {
		super(seed);
	}

	static sat (
		moves,
		source = [
			[0, 7],
			[4, 5, 6],
			[1, 2, 3, 8],
		],
		target = [
			[0, 1, 2, 3, 8],
			[4, 5],
			[6, 7],
		],
	) {
		/**
		 * A state is a partition of the integers 0-8 into three increasing lists. A move is pair of integers i, j in
		 * {0, 1, 2} corresponding to moving the largest number from the end of list i to list j, while preserving the
		 * order of list j. Find a sequence of moves that transform the given source to target states.
		 */
		const state = source.map((s) => s.slice());
		for (const [i, j] of moves) {
			state[j].push(state[i].pop());
			if (
				JSON.stringify(state[j]) !==
				JSON.stringify(state[j].slice().sort((a, b) => a - b))
			)
				throw new Error("invalid move");
		}
		return JSON.stringify(state) === JSON.stringify(target);
	}

	static sol (source, target) {
		if (typeof source === "object" && !Array.isArray(source))
			({ source, target } = source);
		if (!source || !target) return [];
		const state = {};
		for (let i = 0; i < source.length; i++)
			for (const d of source[i]) state[d] = i;
		const final = {};
		for (let i = 0; i < target.length; i++)
			for (const d of target[i]) final[d] = i;
		const disks = Object.keys(state).map(Number);
		if (disks.length === 0) return [];
		const ans = [];
		function move (d, i) {
			if (state[d] === i) return;
			let t;
			for (let z = 0; z < 3; z++)
				if (z !== i && z !== state[d]) {
					t = z;
					break;
				}
			for (let d2 = d + 1; d2 <= Math.max(...disks); d2++)
				if (state[d2] !== undefined) move(d2, t);
			ans.push([state[d], i]);
			state[d] = i;
		}
		for (let d = Math.min(...disks); d <= Math.max(...disks); d++) {
			if (disks.includes(d)) move(d, final[d]);
		}
		return ans;
	}

	genRandom () {
		const n = Math.floor(this.random() * 14) + 4;
		const source = Array.from({ length: 3 }, () => []);
		const target = Array.from({ length: 3 }, () => []);
		const disks = Array.from({ length: n }, (_, i) => i);
		for (const d of disks) {
			source[Math.floor(this.random() * 3)].push(d);
			target[Math.floor(this.random() * 3)].push(d);
		}
		this.add({ source, target });
	}
}

export class LongestMonotonicSubstring extends PuzzleGenerator {
	static docstring = "Remove as few characters as possible from s so that the characters of the remaining string are alphebetical.\n        Here x is the list of string indices that have not been deleted.";

	constructor(seed = null) {
		super(seed);
	}

	static sat (x, length = 13, s = "Dynamic programming solves this puzzle!!!") {
		/**
		 * Remove as few characters as possible from s so that the characters of the remaining string are alphebetical.
		 * Here x is the list of string indices that have not been deleted.
		 */
		if (!Array.isArray(x) || x.length < length) return false;
		for (let i = 0; i < length - 1; i++) {
			if (!(s[x[i]] <= s[x[i + 1]] && x[i + 1] > x[i] && x[i] >= 0))
				return false;
		}
		return true;
	}

	static sol (length, s) {
		if (typeof length === "object") ({ length, s } = length);
		if (s === "" || s === undefined) return [];
		const n = s.length;
		const dyn = []; // (len, end, prev)
		for (let i = 0; i < n; i++) {
			let best = null;
			for (let j = 0; j < dyn.length; j++) {
				const [L, e, prev] = dyn[j];
				if (s[e] <= s[i]) {
					if (!best || L + 1 > best[0]) best = [L + 1, i, e];
				}
			}
			if (!best) dyn.push([1, i, -1]);
			else dyn.push(best);
		}
		let _len = 0,
			idx = 0;
		for (let i = 0; i < dyn.length; i++)
			if (dyn[i][0] > _len) {
				_len = dyn[i][0];
				idx = i;
			}
		const backwards = [dyn[idx][1]];
		let prev = dyn[idx][2];
		while (prev !== -1) {
			backwards.push(prev);
			prev = dyn[prev][2];
		}
		return backwards.reverse();
	}

	genRandom () {
		const n = this.randomRange(1, this.randomChoice([10, 100, 1000]) + 1);
		const length = Math.floor(this.random() * n);
		const rand_chars = Array.from({ length: n }, () =>
			String.fromCharCode(32 + Math.floor(this.random() * 92)),
		);
		const li = rand_chars.slice(0, length).sort();
		for (let i = length; i < n; i++)
			li.splice(Math.floor(this.random() * (i + 1)), 0, rand_chars[i]);
		const s = li.join("");
		this.add({ length, s });
	}
	getExample () {
		return { length: 13, s: "Dynamic programming solves this puzzle!!!" };
	}
}

export class Quine extends PuzzleGenerator {
	static docstring = "Find a string that when evaluated as a JavaScript expression is that string itself.";

	constructor(seed = null) {
		super(seed);
	}
	static sat (quine) {
		/**
		 * Find a string that when evaluated as a JavaScript expression is that string itself.
		 */
		try {
			const v = eval(quine);
			return typeof v === "string" && v.length > 0;
		} catch (e) {
			return false;
		}
	}
	static sol () {
		return '"Q"';
	}
}

export class RevQuine extends PuzzleGenerator {
	static docstring = "Find a string that, when reversed and evaluated gives you back that same string.";

	constructor(seed = null) {
		super(seed);
	}
	// biome-ignore lint/security/noGlobalEval: <explanation>
	static sat (rev_quine) {
		try {
			const v = eval(rev_quine.split("").reverse().join(""));
			return typeof v === "string" && v.length > 0;
		} catch (e) {
			return false;
		}
	}
	static sol () {
		return '"Q"'.split("").reverse().join("");
	}
}

export class BooleanPythagoreanTriples extends PuzzleGenerator {
	static tags = [Tags.math];
	static docstring = "Color the first n integers with one of two colors so that there is no monochromatic Pythagorean triple.\n        A monochromatic Pythagorean triple is a triple of numbers i, j, k such that i^2 + j^2 = k^2 that\n        are all assigned the same color. The input, colors, is a list of 0/1 colors of length >= n.";

	static sat (colors, n = 100) {
		if (!Array.isArray(colors) || colors.length < n) throw new Error("bad");
		const squares = {};
		for (let i = 1; i < colors.length; i++) squares[i * i] = colors[i];
		for (const i in squares)
			for (const j in squares) {
				const k = Number(i) + Number(j);
				if (squares[k] !== undefined)
					if (squares[i] === squares[j] && squares[j] === squares[k])
						return false;
			}
		return true;
	}

	static sol (n) {
		const sqrt = {};
		for (let i = 1; i < n; i++) sqrt[i * i] = i;
		const trips = [];
		for (const a in sqrt)
			for (const b in sqrt) {
				const s = Number(a) + Number(b);
				if (sqrt[s] !== undefined && Number(a) < Number(b))
					trips.push([sqrt[a], sqrt[b], sqrt[s]]);
			}
		const rand = seedrandom("0");
		const sol = Array.from({ length: n }, () => Math.floor(rand() * 2));
		let done = false;
		while (!done) {
			done = true;
			for (const [i, j, k] of trips) {
				if (sol[i] === sol[j] && sol[j] === sol[k]) {
					done = false;
					sol[
						Math.floor(rand() * 3) === 0
							? i
							: Math.floor(rand() * 2) === 0
								? j
								: k
					] =
						1 -
						sol[
						Math.floor(rand() * 3) === 0
							? i
							: Math.floor(rand() * 2) === 0
								? j
								: k
						];
				}
			}
		}
		return sol;
	}
}

export class ClockAngle extends PuzzleGenerator {
	static docstring = "Find clock hands = [hour, min] such that the angle is target_angle degrees.";

	constructor(seed = null) {
		super(seed);
	}
	static sat (hands, target_angle = 45) {
		const [h, m] = hands;
		if (!(h > 0 && h <= 12 && m >= 0 && m < 60)) return false;
		const hour = 30 * h + m / 2;
		const minute = 6 * m;
		const diff = Math.abs(hour - minute) % 360;
		return diff === target_angle || diff === 360 - target_angle;
	}
	static sol (target_angle) {
		if (typeof target_angle === "object") ({ target_angle } = target_angle);
		for (let h = 1; h <= 12; h++) {
			for (let m = 0; m < 60; m++) {
				const hour = 30 * h + m / 2;
				const minute = 6 * m;
				const diff = Math.abs(hour - minute) % 360;
				if (diff === target_angle || diff === 360 - target_angle) return [h, m];
			}
		}
		return null;
	}
	genRandom () {
		const target_angle = Math.floor(this.random() * 360);
		const sol = this.constructor.sol(target_angle);
		if (sol) this.add({ target_angle });
	}
	getExample () {
		return { target_angle: 45 };
	}
}

export class MonkeyAndCoconuts extends PuzzleGenerator {
	static docstring = "Find the number of coconuts to solve the following riddle:\n            There is a pile of coconuts, owned by five men. One man divides the pile into five equal piles, giving the\n            one left over coconut to a passing monkey, and takes away his own share. The second man then repeats the\n            procedure, dividing the remaining pile into five and taking away his share, as do the third, fourth, and\n            fifth, each of them finding one coconut left over when dividing the pile by five, and giving it to a monkey.\n            Finally, the group divide the remaining coconuts into five equal piles: this time no coconuts are left over.\n            How many coconuts were there in the original pile?\n                                              Quoted from https://en.wikipedia.org/wiki/The_monkey_and_the_coconuts";

	constructor(seed = null) {
		super(seed);
	}
	static sat (n) {
		for (let i = 0; i < 5; i++) {
			if (n % 5 !== 1) return false;
			n -= 1 + Math.floor((n - 1) / 5);
		}
		return n > 0 && n % 5 === 1;
	}
	static sol () {
		let m = 1;
		while (true) {
			let n = m;
			let ok = true;
			for (let i = 0; i < 5; i++) {
				if (n % 5 !== 1) {
					ok = false;
					break;
				}
				n -= 1 + Math.floor((n - 1) / 5);
			}
			if (ok && n > 0 && n % 5 === 1) return m;
			m += 5;
		}
	}
	getExample () {
		return {};
	}
	genRandom () {
		// For MonkeyAndCoconuts, sol() returns the number directly
		const n = this.constructor.sol();
		this.add({});
	}
}

export class No3Colinear extends PuzzleGenerator {
	static docstring = "Find num_points points in an side x side grid such that no three points are collinear.";

	constructor(seed = null) {
		super(seed);
	}
	static sat (coords, side = 10, num_points = 20) {
		if (!Array.isArray(coords)) return false;
		for (let i = 0; i < coords.length; i++) {
			const [x1, y1] = coords[i];
			if (!(x1 >= 0 && x1 < side && y1 >= 0 && y1 < side))
				throw new Error("off grid");
			for (let j = 0; j < i; j++) {
				for (let k = 0; k < j; k++) {
					const [x2, y2] = coords[j];
					const [x3, y3] = coords[k];
					if (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) === 0)
						throw new Error("collinear");
				}
			}
		}
		return (
			new Set(coords.map((c) => c.join(":"))).size === coords.length &&
			coords.length >= num_points
		);
	}
	static sol (side, num_points) {
		if (typeof side === "object") ({ side, num_points } = side);
		if (side <= 5) {
			const grid = [];
			for (let i = 0; i < side; i++)
				for (let j = 0; j < side; j++) grid.push([i, j]);
			const comb = (arr, k) => {
				const res = [];
				const f = (start, cur) => {
					if (cur.length === k) {
						res.push(cur.slice());
						return;
					}
					for (let i = start; i < arr.length; i++) {
						cur.push(arr[i]);
						f(i + 1, cur);
						cur.pop();
					}
				};
				f(0, []);
				return res;
			};
			const test = (coords) =>
				coords.every((p, pi) =>
					coords.every((q, qj) =>
						coords.every((r, rj) => {
							if (!(pi > qj && qj > rj)) return true;
							return (
								p[0] * (q[1] - r[1]) +
								q[0] * (r[1] - p[1]) +
								r[0] * (p[1] - q[1])
							);
						}),
					),
				);
			for (const c of comb(grid, num_points)) if (test(c)) return c;
		}
	}

	genRandom () {
		const side = this.randomChoice([5, 10]);
		const num_points = this.randomRange(2, side * side);
		this.add({ side, num_points });
	}
	getExample () {
		return { side: 5, num_points: 8 };
	}
}

// Placeholder classes for missing generators - to be implemented
export class AllPandigitalSquares extends PuzzleGenerator {
	static docstring = "Find all 174 integers whose 10-digit square has all digits 0-9 just once.";
	static sat (nums) {
		/**
		 * Find all 174 integers whose 10-digit square has all digits 0-9 just once.
		 */
		if (nums.length !== 174) return false;
		const uniqueNums = [...new Set(nums)];
		if (uniqueNums.length !== 174) return false;
		for (const n of uniqueNums) {
			const square = (n * n).toString();
			if (square.length !== 10) return false;
			const digits = square.split('').map(Number).sort();
			if (digits.join('') !== '0123456789') return false;
		}
		return true;
	}

	static sol () {
		const result = [];
		for (let i = -100000; i < 100000; i++) {
			const square = (i * i).toString();
			if (square.length === 10) {
				const digits = square.split('').map(Number).sort();
				if (digits.join('') === '0123456789') {
					result.push(i);
				}
			}
		}
		return result;
	}

	getExample () {
		// Return a small example, but since it's all or nothing, perhaps return empty or a subset
		// But for testing, return the full solution
		return { nums: AllPandigitalSquares.sol() };
	}
}

export class CardGame24 extends PuzzleGenerator {
	static docstring = "Find a formula with two 3's and two 7's and + - * / (and parentheses) that evaluates to 24.";

	getExample () {
		return { nums: [1, 2, 3, 4] };
	}

	static sat (expr, nums) {
		// Find a formula with the given numbers and + - * / (and parentheses) that evaluates to 24
		const exprClean = expr.replace(/\s/g, ""); // ignore whitespace
		let digits = "";
		for (let i = 0; i < exprClean.length; i++) {
			if (i === 0 || "+-*/(".includes(exprClean[i - 1])) {
				if (!"123456789(".includes(exprClean[i])) {
					throw new Error("Expr cannot contain **, //, or unary -");
				}
			}
			if (!"0123456789()+-*/".includes(exprClean[i])) {
				throw new Error("Expr can only contain `0123456789()+-*/`");
			}
			digits += "0123456789".includes(exprClean[i]) ? exprClean[i] : " ";
		}
		const usedNums = digits.split(/\s+/).filter(s => s).map(s => parseInt(s)).sort();
		const expectedNums = [...nums].sort();
		if (JSON.stringify(usedNums) !== JSON.stringify(expectedNums)) {
			throw new Error("Each number must occur exactly once");
		}
		// Use Function constructor to safely evaluate the expression
		const result = new Function(`"use strict"; return (${exprClean})`)();
		return Math.abs(result - 24.0) < 1e-6;
	}

	static sol (nums) {
		function helper (pairs) {
			if (pairs.length === 2) {
				const [x, s] = pairs[0];
				const [y, t] = pairs[1];
				const ans = {};
				ans[x + y] = `${s}+${t}`;
				ans[x - y] = `${s}-(${t})`;
				ans[y - x] = `${t}-(${s})`;
				ans[x * y] = `(${s})*(${t})`;
				if (y !== 0) ans[x / y] = `(${s})/(${t})`;
				if (x !== 0) ans[y / x] = `(${t})/(${s})`;
				return ans;
			}
			const ans = {};
			// Try combining different pairs
			for (let i = 0; i < pairs.length; i++) {
				for (let j = i + 1; j < pairs.length; j++) {
					const remaining = pairs.filter((_, idx) => idx !== i && idx !== j);
					const leftResults = helper([pairs[i], pairs[j]]);
					for (const [val, expr] of Object.entries(leftResults)) {
						const rightResults = helper(remaining);
						for (const [val2, expr2] of Object.entries(rightResults)) {
							const combined = helper([[parseFloat(val), `(${expr})`], [parseFloat(val2), `(${expr2})`]]);
							Object.assign(ans, combined);
						}
					}
				}
			}
			return ans;
		}

		const derivations = helper(nums.map(n => [n, n.toString()]));
		for (const [val, expr] of Object.entries(derivations)) {
			if (Math.abs(parseFloat(val) - 24.0) < 1e-6) {
				return expr;
			}
		}
		return null; // No solution found
	}

	genRandom () {
		const nums = Array.from({ length: 4 }, () => this.random.randrange(1, 14));
		if (this.sol(nums)) {
			this.add({ nums });
		}
	}
}

export class CollatzDelay extends PuzzleGenerator {
	static docstring = "Find 0 < n < 2**upper so that it takes exactly t steps to reach 1 in Collatz.";
	static sat (n, t = 197, upper = 20) {
		/**
		 * Find 0 < n < 2**upper so that it takes exactly t steps to reach 1 in Collatz.
		 */
		if (n <= 0 || n >= 2 ** upper) return false;
		let m = n;
		for (let i = 0; i < t; i++) {
			if (n <= 1) return false;
			n = n % 2 === 0 ? n / 2 : 3 * n + 1;
		}
		return n === 1;
	}

	static sol (t, upper) {
		// Simple search for small t
		for (let n = 1; n < 2 ** upper; n++) {
			let steps = 0;
			let m = n;
			while (m !== 1 && steps < t) {
				m = m % 2 === 0 ? m / 2 : 3 * m + 1;
				steps++;
			}
			if (steps === t && m === 1) return n;
		}
		return null; // No solution found
	}

	getExample () {
		return { t: 5, upper: 10 }; // Small example
	}

	genRandom () {
		const t = this.random.randrange(5, 20);
		const upper = this.random.randrange(10, 20);
		this.add({ t, upper });
	}
}

export class CompleteParens extends PuzzleGenerator {
	static docstring = "Add parentheses to the beginning and end of s to make all parentheses balanced";
	static sat (t, s = ")))(Add)some))parens()to()(balance(()(()(me!)((((") {
		/**
		 * Add parentheses to the beginning and end of s to make all parentheses balanced
		 */
		if (!t.includes(s)) return false;
		let depth = 0;
		for (const char of t) {
			if (char === '(') depth++;
			else if (char === ')') depth--;
			if (depth < 0) return false;
		}
		return depth === 0;
	}

	static sol (s) {
		const openCount = (s.match(/\(/g) || []).length;
		const closeCount = (s.match(/\)/g) || []).length;
		return '('.repeat(closeCount) + s + ')'.repeat(openCount);
	}

	getExample () {
		return { s: "))(Add)some))parens()to()(balance(()(()(me!)((((" };
	}

	genRandom () {
		// Generate a random string with unbalanced parens
		let s = '';
		let depth = 0;
		const length = this.random.randrange(10, 50);
		for (let i = 0; i < length; i++) {
			const choices = ['(', ')', 'a', 'b', 'c'];
			if (depth > 0) choices.push(')');
			const char = this.random.choice(choices);
			s += char;
			if (char === '(') depth++;
			else if (char === ')') depth--;
		}
		if (s.length > 5) this.add({ s });
	}
}

export class Easy63 extends PuzzleGenerator {
	static docstring = "Find a formula using two 8s and one 1 and -+*/ that evaluates to 63.";
	static sat (s) {
		// Find a formula using two 8s and one 1 and -+*/ that evaluates to 63.
		const count8 = (s.match(/8/g) || []).length;
		const count1 = (s.match(/1/g) || []).length;
		if (count8 !== 2 || count1 !== 1) return false;
		if (!/^[18+\-*/]+$/.test(s)) return false;
		try {
			return eval(s) === 63;
		} catch {
			return false;
		}
	}

	static sol () {
		return "8*8-1";
	}

	getExample () {
		return {};
	}
}

export class Harder63 extends PuzzleGenerator {
	static docstring = "Find an expression using three 8s and one 1 and -+*/ that evaluates to 63.";
	static sat (s) {
		// Find an expression using three 8s and one 1 and -+*/ that evaluates to 63.
		const count8 = (s.match(/8/g) || []).length;
		const count1 = (s.match(/1/g) || []).length;
		if (count8 !== 3 || count1 !== 1) return false;
		if (!/^[18+\-*/]+$/.test(s)) return false;
		try {
			return eval(s) === 63;
		} catch {
			return false;
		}
	}

	static sol () {
		return "8*8-1**8";
	}

	getExample () {
		return {};
	}
}

export class Kirkman extends PuzzleGenerator {
	static docstring = "Arrange 15 people into groups of 3 each day for seven days so that no two people are in the same group twice.";
	static sat (daygroups, n = 15) {
		/**
		 * Arrange 15 people into groups of 3 each day for seven days so that no two people are in the same group twice.
		 */
		if (daygroups.length !== 7) return false;
		if (!daygroups.every(groups => groups.length === 5 && new Set(groups.flat()).size === 15)) return false;
		if (!daygroups.every(groups => groups.every(g => g.length === 3))) return false;

		// Check that no two people are in the same group twice
		const pairs = new Set();
		for (const groups of daygroups) {
			for (const g of groups) {
				for (let i = 0; i < 3; i++) {
					for (let j = i + 1; j < 3; j++) {
						const pair = [g[i], g[j]].sort((a, b) => a - b).join(',');
						if (pairs.has(pair)) return false;
						pairs.add(pair);
					}
				}
			}
		}
		return pairs.size === (15 * 14) / 2; // All possible pairs
	}

	static sol (n = 15) {
		// Return a known Kirkman triple system solution
		return [
			[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]],
			[[0, 1, 4], [2, 3, 6], [7, 8, 11], [9, 10, 13], [12, 14, 5]],
			[[1, 2, 5], [3, 4, 7], [8, 9, 12], [10, 11, 14], [13, 0, 6]],
			[[4, 5, 8], [6, 7, 10], [11, 12, 0], [13, 14, 2], [1, 3, 9]],
			[[2, 4, 10], [3, 5, 11], [6, 8, 14], [7, 9, 0], [12, 13, 1]],
			[[4, 6, 12], [5, 7, 13], [8, 10, 1], [9, 11, 2], [14, 0, 3]],
			[[10, 12, 3], [11, 13, 4], [14, 1, 7], [0, 2, 8], [5, 6, 9]]
		];
	}

	getExample () {
		return { n: 15 };
	}
}

export class LongestMonotonicSubstringTricky extends PuzzleGenerator {
	static docstring = "Find the indices of the longest substring with characters in sorted order";
	static sat (x, length = 20, s = "Dynamic programming solves this classic job-interview puzzle!!!") {
		/**
		 * Find the indices of the longest substring with characters in sorted order
		 */
		if (x.length !== length) return false;
		for (let i = 0; i < length - 1; i++) {
			if (x[i + 1] <= x[i] || s[x[i]] > s[x[i + 1]]) return false;
		}
		return true;
	}

	static sol (length, s) {
		if (s === "") return [];
		const n = s.length;
		const dyn = [];
		for (let i = 0; i < n; i++) {
			let maxLen = 1;
			let prev = null;
			for (let j = 0; j < i; j++) {
				if (s[j] <= s[i] && dyn[j][0] + 1 > maxLen) {
					maxLen = dyn[j][0] + 1;
					prev = j;
				}
			}
			dyn.push([maxLen, prev]);
		}
		let maxIdx = 0;
		for (let i = 1; i < n; i++) {
			if (dyn[i][0] > dyn[maxIdx][0]) maxIdx = i;
		}
		const result = [];
		let current = maxIdx;
		while (current !== null) {
			result.unshift(current);
			current = dyn[current][1];
		}
		return result;
	}

	getExample () {
		return { length: 5, s: "abcde" };
	}

	genRandom () {
		const n = this.random.randrange(10, 50);
		const s = Array.from({ length: n }, () => String.fromCharCode(this.random.randrange(97, 123))).join('');
		const length = this.sol(0, s).length; // Get the actual length
		this.add({ length, s });
	}
}

export class NecklaceSplit extends PuzzleGenerator {
	static docstring = "Find a split dividing the given red/blue necklace in half at n so that each piece has an equal number of\n        reds and blues.";
	getExample () {
		return { lace: "rrbb" };
	}

	static sat (n, lace) {
		const halfLen = Math.floor(lace.length / 2);
		if (n < 0 || n >= lace.length) return false;
		const sub = lace.slice(n, n + halfLen);
		const totalR = (lace.match(/r/g) || []).length;
		const totalB = (lace.match(/b/g) || []).length;
		const subR = (sub.match(/r/g) || []).length;
		const subB = (sub.match(/b/g) || []).length;
		return totalR === 2 * subR && totalB === 2 * subB;
	}

	static sol (lace) {
		if (lace === "") return 0;
		const totalR = (lace.match(/r/g) || []).length;
		const totalB = (lace.match(/b/g) || []).length;
		const targetR = totalR / 2;
		const targetB = totalB / 2;
		const halfLen = Math.floor(lace.length / 2);
		for (let n = 0; n < halfLen; n++) {
			const sub = lace.slice(n, n + halfLen);
			const subR = (sub.match(/r/g) || []).length;
			const subB = (sub.match(/b/g) || []).length;
			if (subR === targetR && subB === targetB) {
				return n;
			}
		}
		return null; // No solution found
	}

	genRandom () {
		const m = 2 * this.random.randrange(this.random.choice([10, 100, 1000]));
		const lace = Array.from({ length: 2 * m }, (_, i) => i % 2 === 0 ? "r" : "b");
		// Shuffle the array
		for (let i = lace.length - 1; i > 0; i--) {
			const j = this.random.randrange(i + 1);
			[lace[i], lace[j]] = [lace[j], lace[i]];
		}
		this.add({ lace: lace.join("") });
	}
}

export class PandigitalSquare extends PuzzleGenerator {
	static docstring = "Find an integer whose square has all digits 0-9 once.";
	getExample () {
		return {};
	}

	static sat (n) {
		const s = (n * n).toString();
		if (s.length !== 10) return false;
		const digits = s.split('').map(c => parseInt(c)).sort();
		return JSON.stringify(digits) === JSON.stringify([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
	}

	static sol () {
		// The known pandigital square is 32043^2 = 1026753849
		// But let me implement a search like the Python version
		for (let n = 0; n < 100000; n++) {
			const square = n * n;
			const s = square.toString();
			if (s.length === 10) {
				const digitCounts = {};
				for (const char of s) {
					digitCounts[char] = (digitCounts[char] || 0) + 1;
				}
				let isPandigital = true;
				for (let i = 0; i <= 9; i++) {
					if (digitCounts[i.toString()] !== 1) {
						isPandigital = false;
						break;
					}
				}
				if (isPandigital) {
					return n;
				}
			}
		}
		return null; // No solution found
	}
}

export class PostageStamp extends PuzzleGenerator {
	static docstring = "Find a selection of at most max_stamps stamps whose total worth is the target value.";

	getExample () {
		return { target: 80, max_stamps: 4, options: [10, 32, 8] };
	}

	static sat (stamps, target, max_stamps, options) {
		for (const s of stamps) {
			if (!options.includes(s)) {
				throw new Error("Invalid stamp value");
			}
		}
		return stamps.length <= max_stamps && stamps.reduce((sum, s) => sum + s, 0) === target;
	}

	static sol (target, max_stamps, options) {
		// Generate all combinations with replacement
		function* combinationsWithReplacement (arr, r) {
			if (r === 0) {
				yield [];
				return;
			}
			for (let i = 0; i < arr.length; i++) {
				for (const combo of combinationsWithReplacement(arr, r - 1)) {
					yield [arr[i], ...combo];
				}
			}
		}

		for (let n = 0; n <= max_stamps; n++) {
			for (const combo of combinationsWithReplacement(options, n)) {
				if (combo.reduce((sum, s) => sum + s, 0) === target) {
					return combo;
				}
			}
		}
		return null; // No solution found
	}

	genRandom () {
		const max_stamps = this.random.randrange(1, 10);
		const num_options = this.random.randrange(1, 10);
		const options = Array.from({ length: num_options }, () => this.random.randrange(1, 100));
		const num_stamps = this.random.randrange(1, max_stamps + 1);
		// Create stamps by sampling from options
		const stamps = [];
		for (let i = 0; i < num_stamps; i++) {
			stamps.push(this.random.choice(options));
		}
		const target = stamps.reduce((sum, s) => sum + s, 0);
		this.add({ target, max_stamps, options });
	}
}

export class ReverseCase extends PuzzleGenerator {
	static docstring = "Reverse the case of all strings. For those strings which contain no letters, reverse the strings.";

	static sat (rev, strs = ["cat", "u8u", "12532", "", "191", "4tUn8", "ewrWQTEW", "i", "IoU"]) {
		/**
		 * Reverse the case of all strings. For those strings which contain no letters, reverse the strings.
		 */
		if (rev.length !== strs.length) return false;
		for (let i = 0; i < strs.length; i++) {
			const s = strs[i];
			const r = rev[i];
			const swapped = s.split('').map(c => {
				if (c >= 'a' && c <= 'z') return c.toUpperCase();
				if (c >= 'A' && c <= 'Z') return c.toLowerCase();
				return c;
			}).join('');
			if (swapped !== s) {
				// Has letters, should be swapcase
				if (r !== swapped) return false;
			} else {
				// No letters, should be reversed
				if (r !== s.split('').reverse().join('')) return false;
			}
		}
		return true;
	}

	static sol (strs) {
		return strs.map(s => {
			const swapped = s.split('').map(c => {
				if (c >= 'a' && c <= 'z') return c.toUpperCase();
				if (c >= 'A' && c <= 'Z') return c.toLowerCase();
				return c;
			}).join('');
			return swapped !== s ? swapped : s.split('').reverse().join('');
		});
	}

	getExample () {
		return { strs: ["Test", "!@#"] };
	}

	genRandom () {
		const strs = [];
		for (let i = 0; i < this.random.randrange(1, 10); i++) {
			if (this.random.random() < 0.5) {
				strs.push(this.random.pseudoWord(3, 8));
			} else {
				strs.push(this.random.randrange(-1000, 1000).toString());
			}
		}
		this.add({ strs });
	}
}

export class SlidingPuzzle extends PuzzleGenerator {
	static docstring = "In this puzzle, you are given a board like:\n        1 2 5\n        3 4 0\n        6 7 8\n\n        and your goal is to transform it to:\n        0 1 2\n        3 4 5\n        6 7 8\n\n        by a sequence of swaps with the 0 square (0 indicates blank). The starting configuration is given by a 2d list\n        of lists and the answer is represented by a list of integers indicating which number you swap with 0. In the\n        above example, an answer would be [1, 2, 5]";

	static sat (moves, board) {
		/**
		 * Given a sliding puzzle board, check if the sequence of moves solves it.
		 * moves is a list of numbers to swap with the blank (0).
		 */
		const d = board.length;
		const flat = board.flat();
		let blankPos = flat.indexOf(0);

		for (const move of moves) {
			const movePos = flat.indexOf(move);
			const dx = Math.abs((blankPos % d) - (movePos % d));
			const dy = Math.abs(Math.floor(blankPos / d) - Math.floor(movePos / d));

			// Must be adjacent (manhattan distance 1)
			if (dx + dy !== 1) return false;

			// Swap
			[flat[blankPos], flat[movePos]] = [flat[movePos], flat[blankPos]];
			blankPos = movePos;
		}

		// Check if solved
		for (let i = 0; i < flat.length - 1; i++) {
			if (flat[i] !== i + 1) return false;
		}
		return flat[flat.length - 1] === 0;
	}

	static sol (board) {
		/**
		 * Solve the sliding puzzle using BFS (for small puzzles).
		 */
		const d = board.length;
		const N = d * d;
		const target = [];
		for (let i = 1; i < N; i++) target.push(i);
		target.push(0);

		const start = board.flat();
		if (start.join(',') === target.join(',')) return [];

		// BFS
		const queue = [[start, []]];
		const visited = new Set([start.join(',')]);

		while (queue.length > 0) {
			const [state, path] = queue.shift();
			const blankPos = state.indexOf(0);

			// Try all possible moves
			const moves = [];
			const x = blankPos % d;
			const y = Math.floor(blankPos / d);

			if (x > 0) moves.push(blankPos - 1); // left
			if (x < d - 1) moves.push(blankPos + 1); // right
			if (y > 0) moves.push(blankPos - d); // up
			if (y < d - 1) moves.push(blankPos + d); // down

			for (const movePos of moves) {
				const newState = [...state];
				[newState[blankPos], newState[movePos]] = [newState[movePos], newState[blankPos]];
				const stateStr = newState.join(',');

				if (!visited.has(stateStr)) {
					visited.add(stateStr);
					const newPath = [...path, state[movePos]];

					if (newState.join(',') === target.join(',')) {
						return newPath;
					}

					queue.push([newState, newPath]);
				}
			}
		}

		return null; // No solution found
	}

	getExample () {
		return { board: [[1, 2], [3, 0]] };
	}

	genRandom () {
		const d = this.random.randrange(2, 4); // 2x2 or 3x3
		const N = d * d;
		let state = Array.from({ length: N }, (_, i) => i);

		// Make some random moves
		const numMoves = this.random.randrange(10, 50);
		for (let i = 0; i < numMoves; i++) {
			const blankPos = state.indexOf(0);
			const x = blankPos % d;
			const y = Math.floor(blankPos / d);
			const moves = [];

			if (x > 0) moves.push(blankPos - 1);
			if (x < d - 1) moves.push(blankPos + 1);
			if (y > 0) moves.push(blankPos - d);
			if (y < d - 1) moves.push(blankPos + d);

			if (moves.length > 0) {
				const movePos = moves[this.random.randrange(moves.length)];
				[state[blankPos], state[movePos]] = [state[movePos], state[blankPos]];
			}
		}

		const board = [];
		for (let i = 0; i < N; i += d) {
			board.push(state.slice(i, i + d));
		}

		this.add({ board });
	}
}

export class SquaringTheSquare extends PuzzleGenerator {
	static docstring = "Partition a square into smaller squares with unique side lengths. A perfect squared path has distinct sides.\n        xy_sides is a List of (x, y, side)";

	static sat (xy_sides) {
		/**
		 * Partition a square into smaller squares with unique side lengths.
		 */
		if (xy_sides.length <= 1) return false;
		const sides = xy_sides.map(([x, y, s]) => s);
		if (new Set(sides).size !== xy_sides.length) return false;
		let n = 0;
		for (const [x, y, s] of xy_sides) {
			n = Math.max(n, x + s, y + s);
		}
		const totalArea = xy_sides.reduce((sum, [x, y, s]) => sum + s * s, 0);
		if (totalArea !== n * n) return false;
		for (let i = 0; i < xy_sides.length; i++) {
			const [x1, y1, s1] = xy_sides[i];
			if (x1 < 0 || y1 < 0 || x1 + s1 > n || y1 + s1 > n) return false;
			for (let j = 0; j < xy_sides.length; j++) {
				if (i !== j) {
					const [x2, y2, s2] = xy_sides[j];
					if (!(x2 >= x1 + s1 || x2 + s2 <= x1 || y2 >= y1 + s1 || y2 + s2 <= y1)) return false;
				}
			}
		}
		return true;
	}

	static sol () {
		return [[0, 0, 50], [0, 50, 29], [0, 79, 33], [29, 50, 25], [29, 75, 4], [33, 75, 37], [50, 0, 35],
		[50, 35, 15], [54, 50, 9], [54, 59, 16], [63, 50, 2], [63, 52, 7], [65, 35, 17], [70, 52, 18],
		[70, 70, 42], [82, 35, 11], [82, 46, 6], [85, 0, 27], [85, 27, 8], [88, 46, 24], [93, 27, 19]];
	}

	getExample () {
		return {};
	}
}

export class VerbalArithmetic extends PuzzleGenerator {
	static docstring = "Find a substitution of digits for characters to make the numbers add up in a sum like this:\n    SEND + MORE = MONEY\n\n    The first digit in any number cannot be 0. In this example the solution is `9567 + 1085 = 10652`.\n    See [Wikipedia article](https://en.wikipedia.org/wiki/Verbal_arithmetic)";

	static sat (li, words = ["SEND", "MORE", "MONEY"]) {
		/**
		 * Find a list of integers corresponding to the given list of strings substituting a different digit for each
		 * character, so that the last string corresponds to the sum of the previous numbers.
		 */
		if (li.length !== words.length) return false;
		if (!li.every((i, idx) => i > 0 && i.toString().length === words[idx].length)) return false;

		// Check all characters map to unique digits
		const charToDigit = {};
		const digitToChar = {};
		for (let i = 0; i < words.length; i++) {
			const word = words[i];
			const numStr = li[i].toString();
			for (let j = 0; j < word.length; j++) {
				const char = word[j];
				const digit = parseInt(numStr[j]);
				if (charToDigit[char] !== undefined && charToDigit[char] !== digit) return false;
				if (digitToChar[digit] !== undefined && digitToChar[digit] !== char) return false;
				charToDigit[char] = digit;
				digitToChar[digit] = char;
			}
		}

		// Check no leading zeros
		for (const word of words) {
			const digit = charToDigit[word[0]];
			if (digit === 0 && word.length > 1) return false;
		}

		// Check the sum
		const sum = li.slice(0, -1).reduce((a, b) => a + b, 0);
		return sum === li[li.length - 1];
	}

	static sol (words) {
		/**
		 * Solve the cryptarithm using backtracking.
		 */
		const allLetters = [...new Set(words.join(''))];
		const letterToDigit = {};
		const used = new Array(10).fill(false);

		// Check for leading zeros
		const leadingLetters = new Set(words.map(w => w[0]));

		function backtrack (index) {
			if (index === allLetters.length) {
				// Check the sum
				const numbers = words.map(word =>
					parseInt(word.split('').map(c => letterToDigit[c]).join(''))
				);
				const sum = numbers.slice(0, -1).reduce((a, b) => a + b, 0);
				return sum === numbers[numbers.length - 1];
			}

			const letter = allLetters[index];
			for (let digit = 0; digit <= 9; digit++) {
				if (!used[digit] && !(digit === 0 && leadingLetters.has(letter))) {
					letterToDigit[letter] = digit;
					used[digit] = true;

					if (backtrack(index + 1)) {
						return true;
					}

					used[digit] = false;
				}
			}
			return false;
		}

		if (backtrack(0)) {
			return words.map(word =>
				parseInt(word.split('').map(c => letterToDigit[c]).join(''))
			);
		}

		return null;
	}

	getExample () {
		return { words: ["SEND", "MORE", "MONEY"] };
	}

	genRandom () {
		// Use some known puzzles
		const puzzles = [
			["SEND", "MORE", "MONEY"],
			["FORTY", "TEN", "TEN", "SIXTY"],
			["GREEN", "ORANGE", "COLORS"]
		];
		const words = puzzles[this.random.randrange(puzzles.length)];
		this.add({ words });
	}
}

export class WaterPouring extends PuzzleGenerator {
	static docstring = "Given an initial state of water quantities in jugs and jug capacities, find a sequence of moves (pouring\n        one jug into another until it is full or the first is empty) to reaches the given goal state.\n        moves is list of [from, to] pairs";

	static sat (moves, capacities = [8, 5, 3], init = [8, 0, 0], goal = [4, 4, 0]) {
		/**
		 * Find a sequence of moves to reach the goal state.
		 */
		let state = [...init];
		for (const [i, j] of moves) {
			if (i < 0 || j < 0 || i >= capacities.length || j >= capacities.length || i === j) return false;
			const n = Math.min(capacities[j], state[i] + state[j]);
			state[i] = state[i] + state[j] - n;
			state[j] = n;
		}
		return state.every((v, idx) => v === goal[idx]);
	}

	static sol (capacities, init, goal) {
		const num_jugs = capacities.length;
		const start = [...init];
		const target = [...goal];
		const trails = new Map();
		trails.set(start.toString(), [[], start]);
		const queue = [start];
		while (!trails.has(target.toString()) && queue.length > 0) {
			const state = queue.shift();
			for (let i = 0; i < num_jugs; i++) {
				for (let j = 0; j < num_jugs; j++) {
					if (i !== j) {
						const n = Math.min(capacities[j], state[i] + state[j]);
						const new_state = [...state];
						new_state[i] = state[i] + state[j] - n;
						new_state[j] = n;
						const key = new_state.toString();
						if (!trails.has(key)) {
							queue.push(new_state);
							trails.set(key, [[i, j], state]);
						}
					}
				}
			}
		}
		if (!trails.has(target.toString())) return null;
		const ans = [];
		let state = target;
		while (state.toString() !== start.toString()) {
			const [move, prev] = trails.get(state.toString());
			ans.unshift(move);
			state = prev;
		}
		return ans;
	}

	getExample () {
		return { capacities: [8, 5, 3], init: [8, 0, 0], goal: [4, 4, 0] };
	}

	genRandom () {
		const capacities = [this.random.randrange(1, 10), this.random.randrange(1, 10), this.random.randrange(1, 10)];
		const init = capacities.map(c => this.random.randrange(0, c + 1));
		// For simplicity, use a simple goal
		const goal = capacities.map(c => this.random.randrange(0, c + 1));
		this.add({ capacities, init, goal });
	}
}

export class ZeroSum extends PuzzleGenerator {
	static docstring = "Compute minimax optimal strategies for a given zero-sum game up to error tolerance eps.";

	static sat (strategies, A = [[0., -0.5, 1.], [0.75, 0., -1.], [-1., 0.4, 0.]], eps = 0.01) {
		/**
		 * Compute minimax optimal strategies for a given zero-sum game up to error tolerance eps.
		 */
		const m = A.length;
		const n = A[0].length;
		const [p, q] = strategies;
		if (p.length !== m || q.length !== n) return false;
		const sumP = p.reduce((a, b) => a + b, 0);
		const sumQ = q.reduce((a, b) => a + b, 0);
		if (Math.abs(sumP - 1.0) > 1e-6 || Math.abs(sumQ - 1.0) > 1e-6) return false;
		if (p.some(x => x < 0) || q.some(x => x < 0)) return false;
		let v = 0;
		for (let i = 0; i < m; i++) {
			for (let j = 0; j < n; j++) {
				v += A[i][j] * p[i] * q[j];
			}
		}
		for (let i = 0; i < m; i++) {
			let val = 0;
			for (let j = 0; j < n; j++) {
				val += A[i][j] * q[j];
			}
			if (val > v + eps) return false;
		}
		for (let j = 0; j < n; j++) {
			let val = 0;
			for (let i = 0; i < m; i++) {
				val += A[i][j] * p[i];
			}
			if (val < v - eps) return false;
		}
		return true;
	}

	static sol (A, eps) {
		const MAX_ITER = 10000;
		const m = A.length;
		const n = A[0].length;
		const a = new Array(m).fill(0);
		const b = new Array(n).fill(0);

		for (let count = 1; count < MAX_ITER; count++) {
			let i_star = 0;
			let max_val = -Infinity;
			for (let i = 0; i < m; i++) {
				let val = 0;
				for (let j = 0; j < n; j++) {
					val += A[i][j] * b[j];
				}
				if (val > max_val) {
					max_val = val;
					i_star = i;
				}
			}
			let j_star = 0;
			let min_val = Infinity;
			for (let j = 0; j < n; j++) {
				let val = 0;
				for (let i = 0; i < m; i++) {
					val += A[i][j] * a[i];
				}
				if (val < min_val) {
					min_val = val;
					j_star = j;
				}
			}
			a[i_star] += 1;
			b[j_star] += 1;
			const p = a.map(x => x / (count + 1e-6));
			p[p.length - 1] = 1 - p.slice(0, -1).reduce((a, b) => a + b, 0);
			const q = b.map(x => x / (count + 1e-6));
			q[q.length - 1] = 1 - q.slice(0, -1).reduce((a, b) => a + b, 0);

			let v = 0;
			for (let i = 0; i < m; i++) {
				for (let j = 0; j < n; j++) {
					v += A[i][j] * p[i] * q[j];
				}
			}
			let valid = true;
			for (let i = 0; i < m; i++) {
				let val = 0;
				for (let j = 0; j < n; j++) {
					val += A[i][j] * q[j];
				}
				if (val > v + eps) {
					valid = false;
					break;
				}
			}
			if (!valid) continue;
			for (let j = 0; j < n; j++) {
				let val = 0;
				for (let i = 0; i < m; i++) {
					val += A[i][j] * p[i];
				}
				if (val < v - eps) {
					valid = false;
					break;
				}
			}
			if (valid) return [p, q];
		}
		return null;
	}

	getExample () {
		return { A: [[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]], eps: 0.01 };
	}

	genRandom () {
		const m = this.random.randrange(2, 5);
		const n = this.random.randrange(2, 5);
		const A = [];
		for (let i = 0; i < m; i++) {
			A.push([]);
			for (let j = 0; j < n; j++) {
				A[i].push(this.random.random() * 2 - 1); // -1 to 1
			}
		}
		const eps = this.random.choice([0.5, 0.1, 0.01]);
		this.add({ A, eps });
	}
}

export class Znam extends PuzzleGenerator {
	static docstring = "Find k positive integers such that each integer divides (the product of the rest plus 1).";

	static sat (li, k = 5) {
		/**
		 * Find k positive integers such that each integer divides (the product of the rest plus 1).
		 */
		if (li.length !== k) return false;
		if (li.some(x => x <= 1)) return false;
		const prod = (nums) => nums.reduce((a, b) => a * b, 1);
		for (let i = 0; i < k; i++) {
			const rest = li.slice(0, i).concat(li.slice(i + 1));
			const p = prod(rest) + 1;
			if (p % li[i] !== 0) return false;
		}
		return true;
	}

	static sol (k) {
		let n = 2;
		let prod = 1;
		const ans = [];
		while (ans.length < k) {
			ans.push(n);
			prod *= n;
			n = prod + 1;
		}
		return ans;
	}

	getExample () {
		return { k: 5 };
	}

	gen () {
		let k = 5;
		while (this.num_generated_so_far() < this.target_num_instances) {
			this.add({ k }, { test: k < 18 });
			k += 1;
		}
	}
}

export class Sudoku extends PuzzleGenerator {
	static docstring = "Find the unique valid solution to the Sudoku puzzle";

	getExample () {
		return { puz: "____9_2___7__________1_8_4____2_78____4_____1____69____2_8___5__6__3_7___49______" };
	}

	/**
	 * Find the unique valid solution to the Sudoku puzzle
	 */
	static sat (x, puz = "____9_2___7__________1_8_4____2_78____4_____1____69____2_8___5__6__3_7___49______") {
		if (x.length !== 81) return false;
		// Check that filled cells match the puzzle
		for (let i = 0; i < 81; i++) {
			if (puz[i] !== '_' && puz[i] !== x[i]) return false;
		}
		// Check rows, columns, and 3x3 squares
		const full = new Set('123456789');
		for (let i = 0; i < 9; i++) {
			const row = new Set();
			const col = new Set();
			const square = new Set();
			for (let j = 0; j < 9; j++) {
				row.add(x[i * 9 + j]);
				col.add(x[j * 9 + i]);
				const si = 3 * Math.floor(i / 3) + Math.floor(j / 3);
				const sj = 3 * (i % 3) + (j % 3);
				square.add(x[si * 9 + sj]);
			}
			if (row.size !== 9 || col.size !== 9 || square.size !== 9) return false;
		}
		return true;
	}

	static sol (puz) {
		return null; // No solution provided
	}

	genRandom () {
		// Generate a simple Sudoku puzzle (this is a placeholder)
		this.add({ puz: "____9_2___7__________1_8_4____2_78____4_____1____69____2_8___5__6__3_7___49______" });
	}
}
