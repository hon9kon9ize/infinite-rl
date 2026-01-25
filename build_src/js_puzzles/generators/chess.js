import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

export class EightQueensOrFewer extends PuzzleGenerator {
	static tags = [Tags.games, Tags.brute_force, Tags.famous];
	static docstring = "Position min(m, n) <= 8 queens on an m x n chess board so that no pair is attacking each other.";

	getExample () {
		return { m: 8, n: 8 };
	}

	/**
	 * Position min(m, n) <= 8 queens on an m x n chess board so that no pair is attacking each other.
	 */
	static sat (squares, m = 8, n = 8) {
		const k = Math.min(m, n);
		if (!Array.isArray(squares) || squares.length !== k)
			throw new Error("wrong length");
		for (const [i, j] of squares)
			if (!(i >= 0 && i < m && j >= 0 && j < n)) throw new Error("off board");
		const set = new Set();
		for (const [i, j] of squares) {
			set.add(JSON.stringify(["row", i]));
			set.add(JSON.stringify(["col", j]));
			set.add(JSON.stringify(["SE", i + j]));
			set.add(JSON.stringify(["NE", i - j]));
		}
		return set.size === 4 * k;
	}

	/**
	 * Reference solution for EightQueensOrFewer.
	 */
	static sol (m, n) {
		const k = Math.min(m, n);
		const permute = (arr) => {
			const res = [];
			const f = (cur, rem) => {
				if (rem.length === 0) {
					res.push(cur.slice());
					return;
				}
				for (let i = 0; i < rem.length; i++) {
					const v = rem[i];
					rem.splice(i, 1);
					cur.push(v);
					f(cur, rem);
					cur.pop();
					rem.splice(i, 0, v);
				}
			};
			f([], arr.slice());
			return res;
		};
		const perms = permute([...Array(k).keys()]);
		for (const p of perms) {
			const set = new Set();
			for (let i = 0; i < p.length; i++) {
				set.add(JSON.stringify(["row", i]));
				set.add(JSON.stringify(["col", p[i]]));
				set.add(JSON.stringify(["SE", i + p[i]]));
				set.add(JSON.stringify(["NE", i - p[i]]));
			}
			if (set.size === 4 * k) return p.map((j, i) => [i, j]);
		}
	}

	genRandom () {
		const m = Math.floor(this.random() * 6) + 4;
		const n = Math.floor(this.random() * 6) + 4;
		if (Math.min(m, n) <= 8) this.add({ m, n });
	}
}

export class MoreQueens extends PuzzleGenerator {
	static tags = [Tags.games, Tags.graphs, Tags.famous];
	static docstring = "Position min(m, n) > 8 queens on an m x n chess board so that no pair is attacking each other.";

	getExample () {
		return { m: 9, n: 9 };
	}

	/**
	 * Position min(m, n) > 8 queens on an m x n chess board so that no pair is attacking each other.
	 */
	static sat (squares, m = 9, n = 9) {
		const k = Math.min(m, n);
		if (!Array.isArray(squares) || squares.length !== k)
			throw new Error("wrong length");
		const rows = new Set(),
			cols = new Set(),
			se = new Set(),
			ne = new Set();
		for (const [i, j] of squares) {
			if (!(i >= 0 && i < m && j >= 0 && j < n)) throw new Error("off board");
			rows.add(i);
			cols.add(j);
			se.add(i + j);
			ne.add(i - j);
		}
		return rows.size === k && cols.size === k && se.size === k && ne.size === k;
	}

	/**
	 * Reference solution for MoreQueens.
	 */
	static sol (m, n) {
		// Handle dict input
		if (typeof m === "object" && !Array.isArray(m)) {
			({ m, n } = m);
		}
		let t = Math.min(m, n);
		const ans = [];
		if (t % 2 === 1) {
			ans.push([t - 1, t - 1]);
			t -= 1;
		}
		if (t % 6 === 2) {
			for (let i = 0; i < t / 2; i++)
				ans.push([i, (((2 * i + t / 2 - 1) % t) + t) % t]);
			for (let i = 0; i < t / 2; i++)
				ans.push([i + t / 2, (((2 * i - t / 2 + 2) % t) + t) % t]);
		} else {
			for (let i = 0; i < t / 2; i++) ans.push([i, 2 * i + 1]);
			for (let i = 0; i < t / 2; i++) ans.push([i + t / 2, 2 * i]);
		}
		return ans;
	}

	genRandom () {
		const m = Math.floor(this.random() * 6) + 4;
		const n = Math.floor(this.random() * 6) + 4;
		if (Math.min(m, n) > 8) this.add({ m, n });
	}
}

export class KnightsTour extends PuzzleGenerator {
	static tags = [Tags.games, Tags.graphs, Tags.hard, Tags.famous];
	static docstring = "Find an (open) tour of knight moves on an m x n chess-board that visits each square once.";

	getExample () {
		return { m: 8, n: 8 };
	}

	/**
	 * Find an (open) tour of knight moves on an m x n chess-board that visits each square once.
	 */
	static sat (tour, m = 8, n = 8) {
		if (!Array.isArray(tour)) throw new Error("bad");
		const moves = tour
			.slice(0, tour.length - 1)
			.map((p, i) => [tour[i], tour[i + 1]]);
		for (const [[i1, j1], [i2, j2]] of moves)
			if (
				!(
					new Set([Math.abs(i1 - i2), Math.abs(j1 - j2)]).size === 2 &&
					new Set([Math.abs(i1 - i2), Math.abs(j1 - j2)]).has(1) &&
					new Set([Math.abs(i1 - i2), Math.abs(j1 - j2)]).has(2)
				)
			)
				throw new Error("illegal move");
		const sorted = tour
			.slice()
			.map((x) => [x[0], x[1]])
			.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
		const all = [];
		for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) all.push([i, j]);
		return JSON.stringify(sorted) === JSON.stringify(all);
	}

	static sol (m, n) {
		for (let seed = 0; seed < 100; seed++) {
			const r = seed * 1103515245 + 12345; // simple LCG; deterministic
			const rand = (x) => {
				const v =
					((seed * 1103515245 + x * 1664525 + 1013904223) >>> 0) / 4294967295;
				return v;
			};
			const ans = [[0, 0]];
			const free = new Set();
			for (let i = 0; i < m; i++)
				for (let j = 0; j < n; j++) free.add(`${i}:${j}`);
			free.delete("0:0");
			const possible = (i, j) => {
				const moves = [
					[i + 1, j + 2],
					[i + 1, j - 2],
					[i - 1, j + 2],
					[i - 1, j - 2],
					[i + 2, j + 1],
					[i + 2, j - 1],
					[i - 2, j + 1],
					[i - 2, j - 1],
				];
				return moves.filter(([a, b]) => free.has(`${a}:${b}`));
			};
			while (true) {
				if (free.size === 0) return ans;
				const cands = possible(ans[ans.length - 1][0], ans[ans.length - 1][1]);
				if (!cands.length) break;
				cands.sort(
					(a, b) => possible(a[0], a[1]).length - possible(b[0], b[1]).length,
				);
				const pick = cands[Math.floor(rand(ans.length) * cands.length)];
				ans.push(pick);
				free.delete(`${pick[0]}:${pick[1]}`);
			}
		}
	}

	gen (target) {
		const sizes = [9, 8, 7, 6, 5].concat(
			Array.from({ length: 90 }, (_, i) => i + 10),
		);
		for (const n of sizes) {
			if (this.num_generated_so_far() >= target) return;
			const m = n;
			this.add({ m, n });
		}
	}

	genRandom () {
		const m = Math.floor(this.random() * 95) + 5;
		const n = Math.floor(this.random() * 95) + 5;
		if (Math.max(m / n, n / m) <= 2) this.add({ m, n });
	}
}

export class UncrossedKnightsPath extends PuzzleGenerator {
	static tags = [Tags.games, Tags.hard, Tags.famous];
	static docstring = "Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross.";

	constructor(seed = null) {
		super(seed);
		this.nxn_records = {
			3: 2,
			4: 5,
			5: 10,
			6: 17,
			7: 24,
			8: 35,
			9: 47,
			10: 61,
			11: 76,
			12: 94,
			13: 113,
			14: 135,
			15: 158,
			16: 183,
			17: 211,
			18: 238,
			19: 268,
			20: 302,
			21: 337,
			22: 375,
			23: 414,
		};
	}

	static sol (inputs) {
		// Unsolved problem - no solution provided
		return null;
	}

	/**
	 * Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross.
	 */
	static sat (path, m = 8, n = 8, target = 35) {
		const legal_move = (m) => {
			const [[a, b], [i, j]] = m;
			return (
				new Set([Math.abs(i - a), Math.abs(j - b)]).size === 2 &&
				new Set([Math.abs(i - a), Math.abs(j - b)]).has(1) &&
				new Set([Math.abs(i - a), Math.abs(j - b)]).has(2)
			);
		};
		const legal_quad = (m1, m2) => {
			const [[i1, j1], [i2, j2]] = m1;
			const [[a1, b1], [a2, b2]] = m2;
			if (
				new Set([
					[i1, j1].toString(),
					[i2, j2].toString(),
					[a1, b1].toString(),
					[a2, b2].toString(),
				]).size < 4
			)
				return true;
			if ((i1 - i2) * (b1 - b2) === (j1 - j2) * (a1 - a2)) return true;
			if (
				(Math.max(a1, a2, i1, i2) - Math.min(a1, a2, i1, i2)) *
				(Math.max(b1, b2, j1, j2) - Math.min(b1, b2, j1, j2)) >=
				5
			)
				return true;
			return false;
		};
		if (!Array.isArray(path)) throw new Error("bad");
		if (!path.every(([i, j]) => i >= 0 && i < m && j >= 0 && j < n))
			throw new Error("off board");
		const set = new Set(path.map((a) => `${a[0]}:${a[1]}`));
		if (set.size !== path.length) throw new Error("repeat");
		const moves = path.slice(0, -1).map((p, i) => [path[i], path[i + 1]]);
		for (const m of moves) if (!legal_move(m)) throw new Error("illegal");
		for (const m1 of moves)
			for (const m2 of moves)
				if (!legal_quad(m1, m2)) throw new Error("intersect");
		return path.length >= target;
	}

	gen (target) {
		for (const n of Object.keys(this.nxn_records)) {
			if (this.num_generated_so_far() >= target) return;
			this.add({ m: n, n: n, target: this.nxn_records[n] });
		}
	}

	genRandom () {
		const m = Math.floor(this.random() * 7) + 3;
		const n = Math.floor(this.random() * 7) + 3;
		const k = Math.min(m, n);
		if (k in this.nxn_records)
			this.add({
				m,
				n,
				target: Math.floor(this.random() * this.nxn_records[k]),
			});
	}
}

export class UNSOLVED_UncrossedKnightsPath extends PuzzleGenerator {
	static tags = [Tags.unsolved, Tags.games, Tags.famous];
	static docstring = "Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross.";

	constructor(seed = null) {
		super(seed);
		this.unsolved_nxn_records = {
			10: 61,
			11: 76,
			12: 94,
			13: 113,
			14: 135,
			15: 158,
			16: 183,
			17: 211,
			18: 238,
			19: 268,
			20: 302,
			21: 337,
			22: 375,
			23: 414,
		};
	}

	static sol (inputs) {
		// Unsolved problem - no solution provided
		return null;
	}

	/**
	 * Find a long (open) tour of knight moves on an m x n chess-board whose edges don't cross.
	 */
	static sat (path, m = 10, n = 10, target = 62) {
		return new UncrossedKnightsPath().sat(path, m, n, target);
	}
	gen (target) {
		for (const n of Object.keys(this.unsolved_nxn_records)) {
			if (this.num_generated_so_far() >= target) return;
			this.add({ m: n, n: n, target: this.unsolved_nxn_records[n] + 1 });
		}
	}
}
