import { PuzzleGenerator } from "../puzzle_generator.js";

function mulberry32 (a) {
	return () => {
		a += 0x6d2b79f5;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export class BirthdayParadox extends PuzzleGenerator {
	static docstring = "Find n such that the probability of two people having the same birthday in a group of n is near 1/2.";
	constructor(seed = null) {
		super(seed);
	}

	/**
	 * Find n such that the probability of two people having the same birthday in a group of n is near 1/2.
	 */
	static sat (n, year_len = 365) {
		let prob = 1.0;
		for (let i = 0; i < n; i++) prob *= (year_len - i) / year_len;
		return (prob - 0.5) ** 2 <= 1.0 / year_len;
	}

	static sol (year_len) {
		if (typeof year_len === "object") ({ year_len } = year_len); // allow object param
		year_len = year_len === undefined ? 365 : year_len;
		let n = 1;
		let distinct_prob = 1.0;
		let best = [Math.abs(0.5 - distinct_prob), 1];
		while (distinct_prob > 0.5) {
			distinct_prob *= (year_len - n) / year_len;
			n += 1;
			const cand = [Math.abs(0.5 - distinct_prob), n];
			if (cand[0] < best[0] || (cand[0] === best[0] && cand[1] < best[1]))
				best = cand;
		}
		return best[1];
	}

	safe_add (inputs) {
		if (this.sat(this.sol(inputs.year_len), inputs.year_len)) this.add(inputs);
	}

	gen (target) {
		this.safe_add({ year_len: 60182 });
		for (let year_len = 2; year_len < target; year_len++)
			this.safe_add({ year_len });
	}
}

export class BirthdayParadoxMonteCarlo extends BirthdayParadox {
	static docstring = "A slower, Monte Carlo version of the above Birthday Paradox problem.";
	constructor(seed = null) {
		super(seed);
	}
	static sat (n, year_len = 365) {
		const rng = mulberry32(0);
		const K = 1000;
		let count = 0;
		for (let j = 0; j < K; j++) {
			const seen = new Set();
			for (let i = 0; i < n; i++) {
				const r = Math.floor(rng() * year_len);
				seen.add(r);
			}
			if (seen.size < n) count++;
		}
		const prob = count / K;
		return (prob - 0.5) ** 2 <= year_len;
	}
}

export class BallotProblem extends PuzzleGenerator {
	static docstring = "Suppose a list of m 1's and n -1's are permuted at random.\n        What is the probability that all of the cumulative sums are positive?\n        The goal is to find counts = [m, n] that make the probability of the ballot problem close to target_prob.";
	constructor(seed = null) {
		super(seed);
	}

	/**
	 * Suppose a list of m 1's and n -1's are permuted at random.
	 * What is the probability that all of the cumulative sums are positive?
	 * The goal is to find counts = [m, n] that make the probability of the ballot problem close to target_prob.
	 */
	static sat (counts, target_prob = 0.5) {
		const [m, n] = counts;
		let probs = [1.0].concat(Array(n).fill(0.0));
		for (let i = 2; i <= m; i++) {
			const old_probs = probs.slice();
			probs = [1.0].concat(Array(n).fill(0.0));
			for (let j = 1; j < Math.min(n + 1, i); j++) {
				probs[j] = (j / (i + j)) * probs[j - 1] + (i / (i + j)) * old_probs[j];
			}
		}
		return Math.abs(probs[n] - target_prob) < 1e-6;
	}

	static sol (target_prob) {
		if (typeof target_prob === "object") ({ target_prob } = target_prob);
		for (let m = 1; m < 10000; m++) {
			const n = Math.round((m * (1 - target_prob)) / (1 + target_prob));
			if (Math.abs(target_prob - (m - n) / (m + n)) < 1e-6) return [m, n];
		}
	}

	genRandom () {
		const m = this.random.randrange(
			1,
			this.random.choice([10, 100, 200, 300, 400, 500, 1000]),
		);
		const n = this.random.randrange(1, m + 1);
		const target_prob = (m - n) / (m + n);
		this.add({ target_prob });
	}
}

export class BinomialProbabilities extends PuzzleGenerator {
	static docstring = "Find counts = [a, b] so that the probability of a H's and b T's among a + b coin flips is ~ target_prob.";
	constructor(seed = null) {
		super(seed);
	}

	/**
	 * Find counts = [a, b] so that the probability of a H's and b T's among a + b coin flips is ~ target_prob.
	 */
	static sat (counts, p = 0.5, target_prob = 1 / 16.0) {
		const [a, b] = counts;
		const n = a + b;
		const prob = p ** a * (1 - p) ** b; // base p^a (1-p)^b
		// number of sequences with a successes is choose(n,a)
		const choose = (n, k) => {
			let res = 1;
			for (let i = 1; i <= k; i++) {
				res = (res * (n - i + 1)) / i;
			}
			return Math.round(res);
		};
		const tot = choose(n, a) * prob;
		return Math.abs(tot - target_prob) < 1e-6;
	}

	static sol (p, target_prob) {
		if (typeof p === "object") ({ p, target_prob } = p);
		let probs = [1.0];
		const q = 1 - p;
		while (probs.length < 20) {
			const newProbs = [];
			for (let i = 0; i <= probs.length; i++) {
				const left = i - 1 >= 0 ? probs[i - 1] : 0.0;
				const right = i < probs.length ? probs[i] : 0.0;
				newProbs.push(p * left + q * right);
			}
			probs = newProbs;
			const answers = probs
				.map((pv, idx) => (Math.abs(pv - target_prob) < 1e-6 ? idx : -1))
				.filter((x) => x >= 0);
			if (answers.length) return [answers[0], probs.length - 1 - answers[0]];
		}
	}

	genRandom () {
		let probs = [1.0];
		const p = this.random();
		const steps = this.random.randrange(1, 11);
		for (let k = 0; k < steps; k++) {
			const newProbs = Array(probs.length + 1).fill(0);
			for (let i = 0; i < probs.length; i++) {
				newProbs[i] += (1 - p) * probs[i];
				newProbs[i + 1] += p * probs[i];
			}
			probs = newProbs;
		}
		const target_prob = probs[Math.floor(this.random() * probs.length)];
		this.add({ p, target_prob });
	}
}

export class ExponentialProbability extends PuzzleGenerator {
	static docstring = "Find p_stop so that the probability of stopping in steps or fewer time steps is the given target_prob if you\n        stop each step with probability p_stop";
	constructor(seed = null) {
		super(seed);
	}

	/**
	 * Find p_stop so that the probability of stopping in steps or fewer time steps is the given target_prob if you
	 * stop each step with probability p_stop.
	 */
	static sat (p_stop, steps = 10, target_prob = 0.5) {
		let prob = 0.0;
		for (let t = 0; t < steps; t++) prob += p_stop * (1 - p_stop) ** t;
		return Math.abs(prob - target_prob) < 1e-6;
	}
	static sol (steps, target_prob) {
		return 1 - (1 - target_prob) ** (1.0 / steps);
	}
	genRandom () {
		const steps = this.random.randrange(1, 100);
		const target_prob = this.random();
		this.add({ steps, target_prob });
	}
}
