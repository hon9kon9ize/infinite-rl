import { PuzzleGenerator } from "../puzzle_generator.js";

/**
 * BiPermutations - inspired by ICPC 2019 Problem A: Azulejos
 */
export class BiPermutations extends PuzzleGenerator {
	static docstring = "There are two rows of objects. Given the length-n integer arrays of prices and heights of objects in each\n        row, find a permutation of both rows so that the permuted prices are non-decreasing in each row and\n        so that the first row is taller than the second row.";
	getExample () {
		return {
			prices0: [7, 7, 9, 5, 3, 7, 1, 2],
			prices1: [5, 5, 5, 4, 2, 5, 1, 1],
			heights0: [2, 4, 9, 3, 8, 5, 5, 4],
			heights1: [1, 3, 8, 1, 5, 4, 4, 2],
		};
	}

	/**
	 * There are two rows of objects. Given the length-n integer arrays of prices and heights of objects in each
	 * row, find a permutation of both rows so that the permuted prices are non-decreasing in each row and
	 * so that the first row is taller than the second row.
	 */
	static sat (perms, inputs) {
		if (!Array.isArray(perms) || perms.length !== 2) return false;
		const [perm0, perm1] = perms;
		const { prices0, prices1, heights0, heights1 } = inputs;
		const n = prices0.length;

		// Check that they are permutations
		if (!Array.isArray(perm0) || !Array.isArray(perm1)) return false;
		const sorted0 = [...perm0].sort((a, b) => a - b);
		const sorted1 = [...perm1].sort((a, b) => a - b);
		for (let i = 0; i < n; i++) {
			if (sorted0[i] !== i || sorted1[i] !== i) return false;
		}

		// Check prices are non-decreasing
		for (let i = 0; i < n - 1; i++) {
			if (prices0[perm0[i]] > prices0[perm0[i + 1]]) return false;
			if (prices1[perm1[i]] > prices1[perm1[i + 1]]) return false;
		}

		// Check heights: row 0 > row 1
		for (let i = 0; i < n; i++) {
			if (heights0[perm0[i]] <= heights1[perm1[i]]) return false;
		}

		return true;
	}

	static sol (prices0, prices1, heights0, heights1) {
		// Complex algorithm with potential infinite loops - return null for now
		return null;
	}

	genRandom () {
		const choices = [10, 20, 100];
		const choice = choices[Math.floor(this.random() * choices.length)];
		const n = Math.floor(this.random() * (choice - 2 + 1)) + 2;

		// Generate prices (non-decreasing)
		let P = [];
		const maxPrice = 1 + Math.floor(n / 10);
		for (let i = 0; i < n; i++) {
			P.push(Math.floor(this.random() * (maxPrice + 1)));
		}
		P.sort((a, b) => a - b);

		// Generate heights
		let H = [];
		for (let i = 0; i < n; i++) {
			H.push(Math.floor(this.random() * 9) + 1);
		}

		// Create row 1
		const perm1 = [...Array(n).keys()];
		for (let i = perm1.length - 1; i > 0; i--) {
			const j = Math.floor(this.random() * (i + 1));
			[perm1[i], perm1[j]] = [perm1[j], perm1[i]];
		}
		const prices1 = perm1.map((i) => P[i]);
		const heights1 = perm1.map((i) => H[i]);

		// Create row 0 (with larger heights)
		P = [];
		for (let i = 0; i < n; i++) {
			P.push(Math.floor(this.random() * (maxPrice + 1)));
		}
		P.sort((a, b) => a - b);

		H = H.map((h) => h + Math.floor(this.random() * 5) + 1);

		const perm0 = [...Array(n).keys()];
		for (let i = perm0.length - 1; i > 0; i--) {
			const j = Math.floor(this.random() * (i + 1));
			[perm0[i], perm0[j]] = [perm0[j], perm0[i]];
		}
		const prices0 = perm0.map((i) => P[i]);
		const heights0 = perm0.map((i) => H[i]);

		this.add({
			prices0,
			prices1,
			heights0,
			heights1,
		});
	}
}

/**
 * OptimalBridges - inspired by ICPC 2019 Problem B: Bridges
 */
export class OptimalBridges extends PuzzleGenerator {
	static docstring = "You are to choose locations for bridge bases from among a given set of mountain peaks located at\n        `xs, ys`, where `xs` and `ys` are lists of n integers of the same length. Your answer should be a sorted\n        list of indices starting at 0 and ending at n-1. The goal is to minimize building costs such that the bridges\n        are feasible. The bridges are all semicircles placed on top of the pillars. The feasibility constraints are that:\n        * The bridges may not extend above a given height `H`. Mathematically, if the distance between the two xs\n        of adjacent pillars is d, then the semicircle will have radius `d/2` and therefore the heights of the\n        selected mountain peaks must both be at most `H - d/2`.\n        *  The bridges must clear all the mountain peaks, which means that the semicircle must lie above the tops of the\n        peak. See the code for how this is determined mathematically.\n        * The total cost of all the bridges must be at most `thresh`, where the cost is parameter alpha * (the sum of\n        all pillar heights) + beta * (the sum of the squared diameters)";
	getExample () {
		return {
			H: 60,
			alpha: 18,
			beta: 2,
			xs: [0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200],
			ys: [0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10],
			thresh: 26020,
		};
	}

	static sat (
		indices,
		H = 60,
		alpha = 18,
		beta = 2,
		xs = [0, 10, 20, 30, 50, 80, 100, 120, 160, 190, 200],
		ys = [0, 30, 10, 30, 50, 40, 10, 20, 20, 55, 10],
		thresh = 26020,
	) {
		if (!Array.isArray(indices)) return false;

		// Check that indices are sorted and include 0 and n-1
		const n = xs.length;
		const set = new Set(indices);
		if (
			!indices.every((v, i) => i === 0 || v > indices[i - 1]) ||
			!set.has(0) ||
			!set.has(n - 1)
		) {
			return false;
		}

		let cost = alpha * (H - ys[0]);

		for (let idx = 0; idx < indices.length - 1; idx++) {
			const i = indices[idx];
			const j = indices[idx + 1];
			const a = xs[i];
			const b = xs[j];
			const r = (b - a) / 2;

			// Bridge too tall?
			if (Math.max(ys[i], ys[j]) + r > H) return false;

			// Bridge too short? Check all intermediate peaks
			for (let k = i + 1; k < j; k++) {
				const maxY = H - r + Math.sqrt(r * r - (xs[k] - (a + b) / 2) ** 2);
				if (ys[k] > maxY) return false;
			}

			cost += alpha * (H - ys[j]) + beta * (b - a) * (b - a);
		}

		return cost <= thresh;
	}

	static sol (H, alpha, beta, xs, ys, thresh) {
		const n = xs.length;

		const cost = new Array(n).fill(-1);
		const prior = new Array(n).fill(n);

		cost[0] = alpha * (H - ys[0]);

		for (let i = 0; i < n; i++) {
			if (cost[i] === -1) continue;

			let minD = 0;
			let maxD = 2 * (H - ys[i]);

			for (let j = i + 1; j < n; j++) {
				const d = xs[j] - xs[i];
				const h = H - ys[j];

				if (d > maxD) break;

				if (2 * h <= d) {
					minD = Math.max(minD, 2 * d + 2 * h - Math.sqrt(8 * d * h));
				}
				maxD = Math.min(maxD, 2 * d + 2 * h + Math.sqrt(8 * d * h));

				if (minD > maxD) break;

				if (minD <= d && d <= maxD) {
					const newCost = cost[i] + alpha * h + beta * d * d;
					if (cost[j] === -1 || cost[j] > newCost) {
						cost[j] = newCost;
						prior[j] = i;
					}
				}
			}
		}

		const revAns = [n - 1];
		while (revAns[revAns.length - 1] !== 0) {
			revAns.push(prior[revAns[revAns.length - 1]]);
		}

		return revAns.reverse();
	}

	genRandom () {
		const H = 100000;
		const choices = [10, 20, 50, 100, 1000];
		const L = choices[Math.floor(this.random() * choices.length)];
		const n = Math.floor(this.random() * (L - 2)) + 2;
		const alpha = Math.floor(this.random() * L);
		const beta = Math.floor(this.random() * L);
		const m = Math.floor(this.random() * (n - 1)) + 1;

		const keys = [0];
		const samples = [];
		for (let i = 1; i < H; i++) {
			samples.push(i);
		}
		// Fisher-Yates shuffle
		for (let i = samples.length - 1; i > 0; i--) {
			const j = Math.floor(this.random() * (i + 1));
			[samples[i], samples[j]] = [samples[j], samples[i]];
		}
		for (let i = 0; i < m - 1; i++) {
			keys.push(samples[i]);
		}
		keys.push(H);
		keys.sort((a, b) => a - b);

		const dists = [];
		for (let i = 0; i < m; i++) {
			dists.push(keys[i + 1] - keys[i]);
		}

		const heights = [];
		for (let i = 0; i <= m; i++) {
			const maxDist = Math.max(i > 0 ? dists[i - 1] : 0, i < m ? dists[i] : 0);
			const max = H - Math.ceil((maxDist + 1) / 2);
			heights.push(Math.floor(this.random() * (max + 1)));
		}

		const xs = [];
		const ys = [];
		for (let i = 0; i <= m; i++) {
			xs.push(keys[i]);
			ys.push(heights[i]);

			const count = Math.floor(1 / (this.random() + 0.0001)); // Avoid division by zero
			for (let _ = 0; _ < count; _++) {
				if (
					i >= m ||
					xs.length + m + 1 - i >= L ||
					xs[xs.length - 1] === keys[i + 1]
				) {
					break;
				}
				const min = xs[xs.length - 1] + 1;
				const max = keys[i + 1] - 1;
				const x = Math.floor(this.random() * (max - min + 1)) + min;
				xs.push(x);
				const c = (keys[i + 1] + keys[i]) / 2;
				const r = (keys[i + 1] - keys[i]) / 2;
				const maxY = H - r + Math.sqrt(r * r - (x - c) ** 2);
				const y = Math.floor(this.random() * (Math.floor(maxY) + 1));
				ys.push(y);
			}
		}

		const indices = OptimalBridges.sol(H, alpha, beta, xs, ys, null);
		let cost = alpha * (H - ys[0]);
		for (let idx = 0; idx < indices.length - 1; idx++) {
			const i = indices[idx];
			const j = indices[idx + 1];
			const a = xs[i];
			const b = xs[j];
			cost += alpha * (H - ys[j]) + beta * (b - a) * (b - a);
		}
		const finalThresh = cost;

		this.add({ H, alpha, beta, xs, ys, thresh: finalThresh });
	}
}

/**
 * CheckersPosition - inspired by ICPC 2019 Problem C: Checks Post Facto
 */
export class CheckersPosition extends PuzzleGenerator {
	static docstring = "You are given a partial transcript a checkers game. Find an initial position such that the transcript\n        would be a legal set of moves. The board positions are [x, y] pairs with 0 <= x, y < 8 and x + y even.\n        There are two players which we call -1 and 1 for convenience, and player 1 must move first in transcript.\n        The initial position is represented as a list [x, y, piece] where piece means:\n        * 0 is empty square\n        * 1 or -1 is piece that moves only in the y = 1 or y = -1 dir, respectively\n        * 2 or -2 is king for player 1 or player 2 respectively\n\n        Additional rules:\n        * You must jump if you can, and you must continue jumping until one can't any longer.\n        * You cannot start the position with any non-kings on your last rank.\n        * Promotion happens after the turn ends";
	getExample () {
		return {
			transcript: [
				[
					[3, 3],
					[5, 5],
					[3, 7],
				],
				[
					[5, 3],
					[6, 4],
				],
			],
		};
	}

	static sat (answer, transcript) {
		if (!Array.isArray(answer)) return false;
		if (!Array.isArray(transcript)) return false;

		// Build board
		const board = new Map();
		for (let x = 0; x < 8; x++) {
			for (let y = 0; y < 8; y++) {
				if ((x + y) % 2 === 0) {
					board.set(`${x},${y}`, 0);
				}
			}
		}

		// Place pieces
		const seenSquares = new Set();
		for (const [x, y, p] of answer) {
			if (p < -2 || p > 2) return false;
			if (!board.has(`${x},${y}`)) return false;
			if (seenSquares.has(`${x},${y}`)) return false;
			seenSquares.add(`${x},${y}`);
			if (p === 1 && y === 0) return false;
			if (p === -1 && y === 7) return false;
			board.set(`${x},${y}`, p);
		}

		// Validate each move
		let sign = 1;
		for (const move of transcript) {
			if (!Array.isArray(move) || move.length < 2) return false;

			const [sx, sy] = move[0];
			const piece = board.get(`${sx},${sy}`);

			if (!piece || piece * sign <= 0) return false;

			board.set(`${sx},${sy}`, 0);

			for (let i = 1; i < move.length; i++) {
				const [x, y] = move[i];
				const [px, py] = move[i - 1];
				const dx = Math.abs(x - px);
				const dy = Math.abs(y - py);

				if (board.get(`${x},${y}`) !== 0) return false;

				if (dx === 2 && dy === 2) {
					const midX = Math.floor((px + x) / 2);
					const midY = Math.floor((py + y) / 2);
					const midKey = `${midX},${midY}`;
					const captured = board.get(midKey);
					if (!captured || captured * piece >= 0) return false;
					board.set(midKey, 0);
				} else if (dx !== 1 || dy !== 1) {
					return false;
				}
			}

			const [ex, ey] = move[move.length - 1];
			board.set(`${ex},${ey}`, piece);

			if (Math.abs(piece) === 1 && (ey === 0 || ey === 7)) {
				board.set(`${ex},${ey}`, piece * 2);
			}

			sign *= -1;
		}

		return true;
	}

	static sol (transcript) {
		// Complex backtracking problem - return null for now
		return null;
	}

	genRandom () {
		const transcript = this.randomGameTranscript();
		const n = transcript.length;
		const a = Math.floor(this.random() * (Math.floor(n / 2) + 1)) * 2; // randrange(0, n+1, 2)
		const b = Math.floor(this.random() * (n + 1));
		const subset = transcript.slice(a, b);
		this.add({ transcript: subset });
	}

	randomGameTranscript () {
		const board = new Map();
		for (let x = 0; x < 8; x++) {
			for (let y = 0; y < 8; y++) {
				if ((x + y) % 2 === 0) {
					const piece = y < 3 ? 1 : y > 4 ? -1 : 0;
					board.set(`${x},${y}`, piece);
				}
			}
		}

		const deltas = (x, y) => {
			const p = board.get(`${x},${y}`);
			const result = [];
			for (let dx = -1; dx <= 1; dx += 2) {
				for (let dy = -1; dy <= 1; dy += 2) {
					if (dy !== -p) result.push([dx, dy]);
				}
			}
			return result;
		};

		const getJumps = (x, y) => {
			const p = board.get(`${x},${y}`);
			const result = [];
			for (const [dx, dy] of deltas(x, y)) {
				const nx = x + 2 * dx;
				const ny = y + 2 * dy;
				if (board.has(`${nx},${ny}`) && board.get(`${nx},${ny}`) === 0) {
					if (
						board.has(`${x + dx},${y + dy}`) &&
						board.get(`${x + dx},${y + dy}`) * p < 0
					) {
						result.push([nx, ny]);
					}
				}
			}
			return result;
		};

		const transcript = [];
		let sign = 1;

		while (true) {
			const pieces = Array.from(board.keys())
				.map((k) => k.split(",").map(Number))
				.filter(([x, y]) => board.get(`${x},${y}`) * sign > 0);

			const jumps = [];
			for (const [x, y] of pieces) {
				for (const [nx, ny] of getJumps(x, y)) {
					jumps.push([
						[x, y],
						[nx, ny],
					]);
				}
			}

			let move;
			if (jumps.length > 0) {
				move = jumps[Math.floor(this.random() * jumps.length)];
				while (true) {
					const [x1, y1] = move[move.length - 2];
					const [x2, y2] = move[move.length - 1];
					const mid = [Math.floor((x1 + x2) / 2), Math.floor((y1 + y2) / 2)];
					board.set(`${x2},${y2}`, board.get(`${x1},${y1}`));
					board.set(`${x1},${y1}`, 0);
					board.set(`${mid[0]},${mid[1]}`, 0);

					const newJumps = getJumps(x2, y2);
					if (newJumps.length === 0) break;
					move.push(newJumps[Math.floor(this.random() * newJumps.length)]);
				}
			} else {
				const candidates = [];
				for (const [x, y] of pieces) {
					for (const [dx, dy] of deltas(x, y)) {
						const nx = x + dx;
						const ny = y + dy;
						if (board.has(`${nx},${ny}`) && board.get(`${nx},${ny}`) === 0) {
							candidates.push([
								[x, y],
								[nx, ny],
							]);
						}
					}
				}
				if (candidates.length === 0) break;
				move = candidates[Math.floor(this.random() * candidates.length)];
				const [x1, y1] = move[0];
				const [x2, y2] = move[1];
				board.set(`${x2},${y2}`, board.get(`${x1},${y1}`));
				board.set(`${x1},${y1}`, 0);
			}

			transcript.push(move);
			const [endX, endY] = move[move.length - 1];
			if (
				Math.abs(board.get(`${endX},${endY}`)) === 1 &&
				(endY === 0 || endY === 7)
			) {
				board.set(`${endX},${endY}`, board.get(`${endX},${endY}`) * 2);
			}

			sign *= -1;
		}

		return transcript;
	}
}

/**
 * MatchingMarkers - inspired by ICPC 2019 Problem D: Matching Markers
 */
export class MatchingMarkers extends PuzzleGenerator {
	static docstring = "The input is a string of start and end markers \"aaBAcGeg\" where upper-case characters indicate start markers\n        and lower-case characters indicate ending markers. The string indicates a ring (joined at the ends) and the goal is\n        to find a location to split the ring so that there are a maximal number of matched start/end chars where a character\n        (like \"a\"/\"A\") is matched if starting at the split and going around the ring, the start-end pairs form a valid\n        nesting like nested parentheses. Can you solve it in linear time?";
	getExample () {
		return {
			ring: "yRrsmOkLCHSDJywpVDEDsjgCwSUmtvHMefxxPFdmBIpM",
			lower: 5,
		};
	}

	static sat (answer, ring, lower) {
		if (typeof answer !== "number" || answer < 0 || answer >= ring.length)
			return false;

		const line = ring.slice(answer) + ring.slice(0, answer);
		const matches = {};
		for (const c of line.toLowerCase()) {
			matches[c] = 0;
		}

		for (const c of line) {
			const lc = c.toLowerCase();
			if (c === c.toLowerCase()) {
				matches[lc] -= matches[lc] > 0 ? 1 : line.length;
			} else {
				matches[lc] += 1;
			}
		}

		return Object.values(matches).filter((v) => v === 0).length >= lower;
	}

	static sol (ring, lower) {
		const cumulatives = {};
		for (const c of ring.toLowerCase()) {
			cumulatives[c] = [[0, 0]];
		}

		const n = ring.length;
		for (let i = 0; i < n; i++) {
			const c = ring[i];
			const lc = c.toLowerCase();
			const v = cumulatives[lc];
			const delta = c === lc ? -1 : 1;
			v.push([i, v[v.length - 1][1] + delta]);
		}

		const scores = Array(n).fill(0);
		for (const [c, v] of Object.entries(cumulatives)) {
			if (v[v.length - 1][1] !== 0) continue;
			const m = Math.min(...v.map(([i, t]) => t));
			for (let idx = 0; idx < v.length - 1; idx++) {
				const [i, t] = v[idx];
				const [i2, t2] = v[idx + 1];
				if (t === m) {
					for (let j = i + 1; j <= i2; j++) {
						scores[j % n] += 1;
					}
				}
			}
			const lastIdx = v.length - 1;
			const [i, t] = v[lastIdx];
			if (t === m) {
				for (let j = i + 1; j < n; j++) {
					scores[j % n] += 1;
				}
			}
		}

		const maxScore = Math.max(...scores);
		for (let i = 0; i < n; i++) {
			if (scores[i] === maxScore) return i;
		}
		return 0;
	}

	genRandom () {
		const poolSet = new Set();
		const poolSize = Math.floor(this.random() * 25) + 1; // randrange(1, 26)
		const alphabet = "abcdefghijklmnopqrstuvwxyz".split("");
		// Fisher-Yates shuffle for alphabet
		for (let i = alphabet.length - 1; i > 0; i--) {
			const j = Math.floor(this.random() * (i + 1));
			[alphabet[i], alphabet[j]] = [alphabet[j], alphabet[i]];
		}
		for (let i = 0; i < poolSize; i++) {
			poolSet.add(alphabet[i]);
		}
		const pool = Array.from(poolSet);
		for (const c of poolSet) {
			pool.push(c.toUpperCase());
		}

		const exp = Math.floor(this.random() * 11); // randrange(11)
		const length = Math.floor(this.random() * 2 ** exp) + 1; // randrange(1, 2^exp + 1)
		let ring = "";
		for (let i = 0; i < length; i++) {
			ring += pool[Math.floor(this.random() * pool.length)];
		}

		const cutPosition = MatchingMarkers.sol(ring, null);
		const line = ring.slice(cutPosition) + ring.slice(0, cutPosition);
		const matches = {};
		for (const c of line.toLowerCase()) {
			matches[c] = 0;
		}
		for (const c of line) {
			const lc = c.toLowerCase();
			if (c === c.toLowerCase()) {
				matches[lc] -= matches[lc] > 0 ? 1 : line.length;
			} else {
				matches[lc] += 1;
			}
		}
		const finalLower = Object.values(matches).filter((v) => v === 0).length;

		this.add({ ring, lower: finalLower });
	}
}
