import { PuzzleGenerator } from "../puzzle_generator.js";

export class LearnParity extends PuzzleGenerator {
	static docstring = "* Parity learning: Given binary vectors in a subspace, find the secret set S of indices such that: $\\sum_{i \in S} x_i = 1 (mod 2)$";
	getExample () {
		return {
			vecs: [
				169, 203, 409, 50, 37, 479, 370, 133, 53, 159, 161, 367, 474, 107, 82,
				447, 385,
			],
		};
	}

	static sat (inds, vecs) {
		/** Parity learning: Given binary vectors in a subspace, find the secret set S of indices such that: $\\sum_{i \in S} x_i = 1 (mod 2)$ */
		return vecs.every(
			(v) => inds.reduce((acc, i) => acc + ((v >> i) & 1), 0) % 2 === 1,
		);
	}

	static sol (vecs) {
		// Gaussian elimination over GF(2), mirroring Python version
		const vecsArray = vecs.slice();
		let m = Math.max(...vecsArray);
		let d = 0;
		while (m > 0) {
			m >>= 1;
			d += 1;
		}
		const rows = [];
		for (const n of vecsArray) {
			const arr = Array.from({ length: d }, (_, i) => (n >> i) & 1);
			arr.push(1);
			rows.push(arr);
		}
		// identity padding
		const pool = Array.from({ length: d }, (_, i) =>
			Array.from({ length: d + 1 }, (_, j) => (i === j ? 1 : 0)),
		).concat(rows);
		for (let i = 0; i < d; i++) {
			// find a row with bit i == 1
			let v = null;
			for (let r = d; r < pool.length; r++)
				if (pool[r][i] === 1) {
					v = pool[r];
					break;
				}
			if (!v) v = pool[i];
			const w = v.slice();
			for (let r = 0; r < pool.length; r++) {
				if (pool[r][i] === 1) {
					for (let j = 0; j < d + 1; j++) pool[r][j] ^= w[j];
				}
			}
		}
		const ans = [];
		for (let i = 0; i < d; i++) if (pool[i][d] === 1) ans.push(i);
		return ans;
	}

	randParityProblem (rand, d = 63) {
		// Sample unique random indices for the secret
		const secret = [];
		const pool = Array.from({ length: d }, (_, i) => i);
		for (let i = 0; i < Math.floor(d / 2); i++) {
			const idx = Math.floor(rand() * (pool.length - i));
			secret.push(pool[idx]);
			[pool[idx], pool[pool.length - 1 - i]] = [
				pool[pool.length - 1 - i],
				pool[idx],
			];
		}

		const num_vecs = d + 9;
		const vecs = [];
		for (let i = 0; i < num_vecs; i++) {
			const v = Array.from({ length: d }, () => Math.floor(rand() * 2));
			vecs.push(v);
		}

		// Set the parity bit: v[secret[0]] = (1 + sum of other secret indices) % 2
		for (const v of vecs) {
			let sum = 1;
			for (let i = 1; i < secret.length; i++) {
				sum += v[secret[i]];
			}
			v[secret[0]] = sum % 2;
		}

		return vecs.map((v) => v.reduce((acc, b, i) => acc + (b << i), 0));
	}

	gen (targetNum) {
		const vecs = this.randParityProblem(this.random, 63);
		this.add({ vecs }, { multiplier: 10 });
	}

	genRandom () {
		const d = Math.floor(this.random() * 100);
		const vecs = this.randParityProblem(
			this.random,
			Math.max(2, Math.floor(this.random() * 20)),
		);
		this.add({ vecs });
	}
}

export class LearnParityWithNoise extends PuzzleGenerator {
	static docstring = "* Learning parity with noise: Given binary vectors, find the secret set $S$ of indices such that, for at least 3/4 of the vectors, $$sum_{i \in S} x_i = 1 (mod 2)$$";
	getExample () {
		return {
			vecs: [
				26, 5, 32, 3, 15, 18, 31, 13, 24, 25, 34, 5, 15, 24, 16, 13, 0, 27, 37,
			],
		};
	}

	static sat (inds, vecs) {
		/** Learning parity with noise: Given binary vectors, find the secret set $S$ of indices such that, for at least 3/4 of the vectors, $$sum_{i \in S} x_i = 1 (mod 2)$$ */
		const successes = vecs.reduce(
			(acc, v) => acc + (inds.reduce((a, i) => a + ((v >> i) & 1), 0) % 2),
			0,
		);
		return successes >= (vecs.length * 3) / 4;
	}

	static sol (vecs) {
		// brute force-ish: try many random subsets with seeded random for determinism
		const vecsArray = vecs.slice();
		let m = Math.max(...vecsArray);
		let d = 0;
		while (m > 0) {
			m >>= 1;
			d += 1;
		}
		// Seeded random for deterministic behavior (matching Python random.Random(0))
		let seed = 0;
		const rand = () => {
			seed = (seed * 9301 + 49297) % 233280;
			return seed / 233280;
		};
		const target = (vecsArray.length * 3) / 4;
		const maxAttempts = 1e5;
		for (let t = 0; t < maxAttempts; t++) {
			const ans = [];
			for (let i = 0; i < d; i++) if (Math.floor(rand() * 2)) ans.push(i);
			const s = vecsArray.reduce(
				(acc, v) => acc + (ans.reduce((a, i) => a + ((v >> i) & 1), 0) % 2),
				0,
			);
			if (s >= target) return ans;
		}
	}

	randParityProblem (rand, d = 63) {
		// Sample unique random indices for the secret
		const secret = [];
		const pool = Array.from({ length: d }, (_, i) => i);
		for (let i = 0; i < Math.floor(d / 2); i++) {
			const idx = Math.floor(rand() * (pool.length - i));
			secret.push(pool[idx]);
			[pool[idx], pool[pool.length - 1 - i]] = [
				pool[pool.length - 1 - i],
				pool[idx],
			];
		}

		const num_vecs = 2 * d + 5;
		const vecs = [];
		for (let i = 0; i < num_vecs; i++) {
			const v = Array.from({ length: d }, () => Math.floor(rand() * 2));
			vecs.push(v);
		}

		// Set the parity bit: v[secret[0]] = (1 + sum of other secret indices) % 2
		for (const v of vecs) {
			let sum = 1;
			for (let i = 1; i < secret.length; i++) {
				sum += v[secret[i]];
			}
			v[secret[0]] = sum % 2;
		}

		// flip some mistakes
		const numErrors = Math.floor(rand() * vecs.length * 0.25);
		for (let i = 0; i < numErrors; i++) {
			vecs[i][secret[0]] ^= 1;
		}

		return vecs.map((v) => v.reduce((acc, b, i) => acc + (b << i), 0));
	}

	gen (targetNum) {
		const vecs = this.randParityProblem(this.random, 63);
		this.add({ vecs }, { test: false, multiplier: 1000 });
	}

	genRandom () {
		const d = Math.max(
			2,
			Math.floor(this.random() * this.randomChoice([11, 100])),
		);
		const vecs = this.randParityProblem(this.random, d);
		this.add({ vecs }, { test: d < 19 });
	}
}
