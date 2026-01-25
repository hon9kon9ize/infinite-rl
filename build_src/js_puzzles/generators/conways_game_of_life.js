import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

function neighborsOf (x, y) {
	return [
		[x + 1, y],
		[x - 1, y],
		[x, y + 1],
		[x, y - 1],
		[x + 1, y + 1],
		[x + 1, y - 1],
		[x - 1, y + 1],
		[x - 1, y - 1],
	];
}
function setFromArray (arr) {
	return new Set(arr.map(([x, y]) => `${x}:${y}`));
}
function arrayFromSet (s) {
	return [...s].map((e) => e.split(":").map(Number));
}

export class Oscillators extends PuzzleGenerator {
	static docstring = "Find a pattern in Conway's Game of Life https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life that repeats\n        with a certain period https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life";

	static tags = [Tags.games, Tags.famous];

	getExample () {
		return { period: 3 };
	}

	/**
	 * Find a pattern in Conway's Game of Life https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life that repeats with a certain period https://en.wikipedia.org/wiki/Oscillator_%28cellular_automaton%29#:~:text=Game%20of%20Life
	 */
	static sat (init, period = 3) {
		const target = setFromArray(init);
		let live = new Set(target);
		for (let t = 0; t < period; t++) {
			const visible = new Set();
			for (const z of live) {
				const [x, y] = z.split(":").map(Number);
				for (const [a, b] of neighborsOf(x, y)) visible.add(`${a}:${b}`);
			}
			const next = new Set();
			for (const v of visible) {
				const [a, b] = v.split(":").map(Number);
				let cnt = 0;
				for (const [na, nb] of neighborsOf(a, b))
					if (live.has(`${na}:${nb}`)) cnt++;
				if (live.has(v)) {
					if (cnt === 2 || cnt === 3) next.add(v);
				} else {
					if (cnt === 3) next.add(v);
				}
			}
			live = next;
			if (setEquals(live, target)) return t + 1 === period;
		}
	}

	static sol (period) {
		// randomized search similar to python
		const rand = mulberry32(1);
		const deltas = [
			[1, 0],
			[-1, 0],
			[0, 1],
			[0, -1],
			[1, 1],
			[1, -1],
			[-1, 1],
			[-1, -1],
		];
		const completes = Array.from({ length: 30 }, (_, n) => {
			const arr = [];
			for (let x = 0; x < n; x++) for (let y = 0; y < n; y++) arr.push([x, y]);
			return arr;
		});
		for (let attempt = 0; attempt < 100000; attempt++) {
			const n = 3 + Math.floor(rand() * 7);
			const m = 3 + Math.floor(rand() * (n * n - 2));
			const base = completes[n];
			const live = new Set();
			const perm = shuffle(base.slice(), rand);
			for (let i = 0; i < m; i++) live.add(`${perm[i][0]}:${perm[i][1]}`);
			if (Math.floor(rand() * 2))
				for (const s of Array.from(live)) {
					const [x, y] = s.split(":").map(Number);
					live.add(`${-x}:${-y}`);
				}
			if (Math.floor(rand() * 2))
				for (const s of Array.from(live)) {
					const [x, y] = s.split(":").map(Number);
					live.add(`${x}:${-y}`);
				}
			const memory = new Map();
			for (let step = 0; step < period * 10; step++) {
				const key = fingerprintSet(live);
				if (memory.has(key)) {
					if (memory.get(key) === step - period) return arrayFromSet(live);
					break;
				}
				memory.set(key, step);
				const visible = new Set();
				for (const s of live) {
					const [x, y] = s.split(":").map(Number);
					for (const [dx, dy] of deltas) visible.add(`${x + dx}:${y + dy}`);
				}
				const next = new Set();
				for (const v of visible) {
					const [a, b] = v.split(":").map(Number);
					let cnt = 0;
					for (const [dx, dy] of deltas)
						if (live.has(`${a + dx}:${b + dy}`)) cnt++;
					if (live.has(v)) {
						if (cnt === 2 || cnt === 3) next.add(v);
					} else {
						if (cnt === 3) next.add(v);
					}
				}
				live.clear();
				for (const x of next) live.add(x);
			}
		}
		return null;
	}

	gen (target) {
		for (let period = 1; period <= target; period++)
			this.add({ period }, period === 1 || period === 2 || period === 3);
	}
}

export class ReverseLifeStep extends PuzzleGenerator {
	static docstring = "Given a target pattern in Conway's Game of Life (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life ),\n        specified by [x,y] coordinates of live cells, find a position that leads to that pattern on the next step.";

	static tags = [Tags.games, Tags.famous];

	getExample () {
		return {
			target: [
				[1, 3],
				[1, 4],
				[2, 5],
			],
		};
	}

	/**
	 * Given a target pattern in Conway's Game of Life (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life ), specified by [x,y] coordinates of live cells, find a position that leads to that pattern on the next step.
	 */
	static sat (
		position,
		target = [
			[1, 3],
			[1, 4],
			[2, 5],
		],
	) {
		const live = setFromArray(position);
		const deltas = neighborsOf;
		const visible = new Set();
		for (const s of live) {
			const [x, y] = s.split(":").map(Number);
			for (const [a, b] of neighborsOf(x, y)) visible.add(`${a}:${b}`);
		}
		const next = new Set();
		for (const v of visible) {
			const [a, b] = v.split(":").map(Number);
			let cnt = 0;
			for (const [na, nb] of neighborsOf(a, b))
				if (live.has(`${na}:${nb}`)) cnt++;
			if (live.has(v)) {
				if (cnt === 2 || cnt === 3) next.add(v);
			} else {
				if (cnt === 3) next.add(v);
			}
		}
		return setEquals(next, setFromArray(target));
	}

	static sol (target) {
		// MC optimization (fixed-temperature)
		const TEMP = 0.05;
		const rand = mulberry32(0);
		const tgt = setFromArray(target);
		const deltas = neighborsOf;
		const distance = (liveSet) => {
			const visible = new Set();
			for (const s of liveSet) {
				const [x, y] = s.split(":").map(Number);
				for (const [a, b] of deltas(x, y)) visible.add(`${a}:${b}`);
			}
			const next = new Set();
			for (const v of visible) {
				const [a, b] = v.split(":").map(Number);
				let cnt = 0;
				for (const [na, nb] of deltas(a, b))
					if (liveSet.has(`${na}:${nb}`)) cnt++;
				if (liveSet.has(v)) {
					if (cnt === 2 || cnt === 3) next.add(v);
				} else {
					if (cnt === 3) next.add(v);
				}
			}
			return symmetricDifferenceCount(next, tgt);
		};
		let pos = new Set(tgt);
		let cur_dist = distance(pos);
		for (let step = 0; step < 100000; step++) {
			if (cur_dist === 0) return arrayFromSet(pos);
			const options = Array.from(
				new Set(
					Array.from(new Set([...pos, ...tgt])).flatMap((s) => {
						const [x, y] = s.split(":").map(Number);
						return neighborsOf(x, y).map(([a, b]) => `${a}:${b}`);
					}),
				),
			);
			const z = options[Math.floor(rand() * options.length)];
			const newPos = new Set(pos);
			if (newPos.has(z)) newPos.delete(z);
			else newPos.add(z);
			const dist = distance(newPos);
			if (Math.random() <= TEMP ** (dist - cur_dist)) {
				pos = newPos;
				cur_dist = dist;
			}
		}
		return null;
	}

	genRandom () {
		const n = Math.floor(this.random() * 10);
		const live = new Set();
		for (let i = 0; i < n; i++)
			live.add(`${this.randomChoiceCoords()}:${this.randomChoiceCoords()}`);
		const deltas = neighborsOf;
		const visible = new Set();
		for (const s of live) {
			const [x, y] = s.split(":").map(Number);
			for (const [a, b] of deltas(x, y)) visible.add(`${a}:${b}`);
		}
		const next = new Set();
		for (const v of visible) {
			const [a, b] = v.split(":").map(Number);
			let cnt = 0;
			for (const [na, nb] of deltas(a, b)) if (live.has(`${na}:${nb}`)) cnt++;
			if (live.has(v)) {
				if (cnt === 2 || cnt === 3) next.add(v);
			} else {
				if (cnt === 3) next.add(v);
			}
		}
		const target = arrayFromSet(next).sort(
			(a, b) => a[0] - b[0] || a[1] - b[1],
		);
		this.add({ target }, this.num_generated_so_far() < 10);
	}

	randomChoiceCoords () {
		return Math.floor(this.random() * 11) - 5;
	}
}

export class Spaceship extends PuzzleGenerator {
	static docstring = "Find a \"spaceship\" (see https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29 ) in Conway's\n        Game of Life see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life with a certain period";

	static tags = [Tags.games, Tags.famous];

	getExample () {
		return { period: 4 };
	}

	static sol (period) {
		// Unsolved problem - no solution provided
		return null;
	}

	/**
	 * Find a "spaceship" (see https://en.wikipedia.org/wiki/Spaceship_%28cellular_automaton%29 ) in Conway's Game of Life see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life with a certain period
	 */
	static sat (init, period = 4) {
		const live = setFromArray(init);
		const initTot = sumSet(live);
		const target = new Set(
			[...live].map((s) => {
				const [x, y] = s.split(":").map(Number);
				return `${x * live.size - initTot.x}:${y * live.size - initTot.y}`;
			}),
		);
		const deltas = neighborsOf;
		let cur = new Set(live);
		for (let t = 0; t < period; t++) {
			const visible = new Set();
			for (const s of cur) {
				const [x, y] = s.split(":").map(Number);
				for (const [a, b] of deltas(x, y)) visible.add(`${a}:${b}`);
			}
			const next = new Set();
			for (const v of visible) {
				const [a, b] = v.split(":").map(Number);
				let cnt = 0;
				for (const [na, nb] of deltas(a, b)) if (cur.has(`${na}:${nb}`)) cnt++;
				if (cur.has(v)) {
					if (cnt === 2 || cnt === 3) next.add(v);
				} else {
					if (cnt === 3) next.add(v);
				}
			}
			cur = next;
			const tot = sumSet(cur);
			const shifted = new Set(
				[...cur].map((s) => {
					const [x, y] = s.split(":").map(Number);
					return `${x * cur.size - tot.x}:${y * cur.size - tot.y}`;
				}),
			);
			if (setEquals(shifted, target) && cur.size !== live.size)
				return t + 1 === period;
		}
	}

	gen (target) {
		for (let period = 2; period <= target + 1; period++)
			this.add({ period }, period !== 33 && period !== 34);
	}
}

// Utilities
function setEquals (a, b) {
	if (a.size !== b.size) return false;
	for (const x of a) if (!b.has(x)) return false;
	return true;
}
function fingerprintSet (s) {
	const arr = Array.from(s).sort();
	const real = arr.join("|");
	let x = 0,
		y = 0;
	for (let i = 0; i < real.length; i++) {
		x += real.charCodeAt(i) * 17;
		y += real.charCodeAt(i) * 31;
	}
	return `${x}:${y}`;
}
function arraySumCoords (s) {
	let x = 0,
		y = 0;
	for (const a of s) {
		const [i, j] = a.split(":").map(Number);
		x += i;
		y += j;
	}
	return { x, y };
}
function sumSet (s) {
	return arraySumCoords(Array.from(s));
}
function symmetricDifferenceCount (a, b) {
	const A = new Set(a);
	const B = new Set(b);
	let cnt = 0;
	for (const x of A) if (!B.has(x)) cnt++;
	for (const x of B) if (!A.has(x)) cnt++;
	return cnt;
}

function shuffle (arr, rand) {
	for (let i = arr.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[arr[i], arr[j]] = [arr[j], arr[i]];
	}
	return arr;
}
function mulberry32 (a) {
	return () => {
		a += 0x6d2b79f5;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// helper for sampling
function randomChoice (arr, rand) {
	return arr[Math.floor(rand() * arr.length)];
}
