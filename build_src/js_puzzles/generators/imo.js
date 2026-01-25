import { PuzzleGenerator } from "../puzzle_generator.js";

export class ExponentialCoinMoves extends PuzzleGenerator {
	static docstring = "There are five boxes each having one coin initially. Two types of moves are allowed:\n        * (advance) remove `k > 0` coins from box `i` and add `2k` coins to box `i + 1`\n        * (swap) remove a coin from box `i` and swap the contents of boxes `i+1` and `i+2`\n        Given `0 <= n <= 16385`, find a sequence of states that result in 2^n coins in the last box.\n        Note that `n` can be as large as 16385 yielding 2^16385 coins (a number with 4,933 digits) in the last\n        box. Encode each state as a list of the numbers of coins in the five boxes.\n\n        Sample Input:\n        `n = 2`\n\n        Sample Output:\n        `[[1, 1, 1, 1, 1], [0, 3, 1, 1, 1], [0, 1, 5, 1, 1], [0, 1, 4, 1, 1], [0, 0, 1, 4, 1], [0, 0, 0, 1, 4]]`\n\n        The last box now has 2^2 coins. This is a sequence of two advances followed by three swaps.\n\n        states is encoded by lists of 5 coin counts";
	getExample () {
		// skip_example = true in Python, so we don't provide a simple example
		return { n: 2 };
	}

	static sat (states, n = 16385) {
		/**
		 * There are five boxes each having one coin initially. Two types of moves are allowed:
		 * * (advance) remove `k > 0` coins from box `i` and add `2k` coins to box `i + 1`
		 * * (swap) remove a coin from box `i` and swap the contents of boxes `i+1` and `i+2`
		 * Given `0 <= n <= 16385`, find a sequence of states that result in 2^n coins in the last box.
		 * Note that `n` can be as large as 16385 yielding 2^16385 coins (a number with 4,933 digits) in the last
		 * box. Encode each state as a list of the numbers of coins in the five boxes.
		 *
		 * Sample Input:
		 * `n = 2`
		 *
		 * Sample Output:
		 * `[[1, 1, 1, 1, 1], [0, 3, 1, 1, 1], [0, 1, 5, 1, 1], [0, 1, 4, 1, 1], [0, 0, 1, 4, 1], [0, 0, 0, 1, 4]]`
		 *
		 * The last box now has 2^2 coins. This is a sequence of two advances followed by three swaps.
		 *
		 * states is encoded by lists of 5 coin counts
		 */
		if (states.length === 0) return false;
		if (JSON.stringify(states[0]) !== JSON.stringify([1, 1, 1, 1, 1]))
			return false;
		if (!states.every((s) => s.length === 5 && s.every((i) => i >= 0)))
			return false;

		for (let idx = 0; idx < states.length - 1; idx++) {
			const prev = states[idx];
			const cur = states[idx + 1];

			let i = 0;
			for (; i < 5; i++) {
				if (cur[i] !== prev[i]) break;
			}

			if (i >= 5) return false; // No change detected
			if (cur[i] >= prev[i]) return false; // Must decrease

			// Check for advance: cur[i+1] - prev[i+1] == 2 * (prev[i] - cur[i])
			const advanceValid =
				i + 1 < 5 &&
				cur[i + 1] - prev[i + 1] === 2 * (prev[i] - cur[i]) &&
				JSON.stringify(cur.slice(i + 2)) === JSON.stringify(prev.slice(i + 2));

			// Check for swap: cur[i:i+3] == [prev[i]-1, prev[i+2], prev[i+1]]
			const swapValid =
				i + 2 < 5 &&
				cur[i] === prev[i] - 1 &&
				cur[i + 1] === prev[i + 2] &&
				cur[i + 2] === prev[i + 1] &&
				JSON.stringify(cur.slice(i + 3)) === JSON.stringify(prev.slice(i + 3));

			if (!advanceValid && !swapValid) return false;
		}

		// Check if last box has 2^n coins
		return states[states.length - 1][4] === 2 ** n;
	}

	static sol (n) {
		if (n < 1) return [];

		const ans = [
			[1, 1, 1, 1, 1],
			[0, 3, 1, 1, 1],
			[0, 2, 3, 1, 1],
			[0, 2, 2, 3, 1],
			[0, 2, 2, 0, 7],
			[0, 2, 1, 7, 0],
			[0, 2, 1, 0, 14],
			[0, 2, 0, 14, 0],
			[0, 1, 14, 0, 0],
		];

		const expMove = () => {
			const state = ans[ans.length - 1].slice();
			state[2] -= 1;
			state[3] += 2;
			ans.push(state.slice());

			while (state[2]) {
				const prevState3 = state[3];
				state[3] = 0;
				state[4] = 2 * prevState3;
				ans.push(state.slice());
				state[2] -= 1;
				state[3] = state[4];
				state[4] = 0;
				ans.push(state.slice());
			}
		};

		expMove();
		ans.push([0, 0, 2 ** 14, 0, 0]);

		if (n <= 16) {
			ans.push([0, 0, 0, 2 ** 15, 0]);
		} else {
			expMove();
		}

		const state = ans[ans.length - 1].slice();
		state[3] -= 2 ** (n - 1);
		state[4] = 2 ** n;
		ans.push(state);

		return ans;
	}

	genRandom () {
		const n = this.random.randint(1, 20);
		this.add({ n });
	}
}

export class NoRelativePrimes extends PuzzleGenerator {
	static docstring = "Let P(n) = n^2 + n + 1.\n\n        Given b>=6 and m>=1, find m non-negative integers for which the set {P(a+1), P(a+2), ..., P(a+b)} has\n        the property that there is no element that is relatively prime to every other element.\n\n        Sample input:\n        b = 6\n        m = 2\n\n        Sample output:\n        [195, 196]";
	getExample () {
		return { b: 7, m: 6 };
	}

	static gcd (i, j) {
		let r = Math.max(i, j);
		let s = Math.min(i, j);
		while (s >= 1) {
			[r, s] = [s, r % s];
		}
		return r;
	}

	static sat (nums, b = 7, m = 6) {
		/**
		 * Let P(n) = n^2 + n + 1.
		 *
		 * Given b>=6 and m>=1, find m non-negative integers for which the set {P(a+1), P(a+2), ..., P(a+b)} has
		 * the property that there is no element that is relatively prime to every other element.
		 *
		 * Sample input:
		 * b = 6
		 * m = 2
		 *
		 * Sample output:
		 * [195, 196]
		 */
		if (
			nums.length !== new Set(nums).size ||
			nums.length !== m ||
			Math.min(...nums) < 0
		)
			return false;

		for (const a of nums) {
			const p = Array.from(
				{ length: b },
				(_, i) => (a + i + 1) ** 2 + (a + i + 1) + 1,
			);
			if (
				!p.every((val) =>
					p.some(
						(other, idx) =>
							idx !== p.indexOf(val) && NoRelativePrimes.gcd(val, other) > 1,
					),
				)
			) {
				return false;
			}
		}

		return true;
	}

	static sol (b, m) {
		const gcd = (i, j) => {
			let r = Math.max(i, j);
			let s = Math.min(i, j);
			while (s >= 1) {
				[r, s] = [s, r % s];
			}
			return r;
		};

		const ans = [];
		const seen = new Set();
		const deltas = new Set();

		const go = (a) => {
			if (a < 0 || seen.has(a) || ans.length === m) return;
			seen.add(a);

			const p = Array.from(
				{ length: b },
				(_, i) => (a + i + 1) ** 2 + (a + i + 1) + 1,
			);
			if (
				p.every((val) =>
					p.some((other, idx) => idx !== p.indexOf(val) && gcd(val, other) > 1),
				)
			) {
				ans.push(a);
				for (const a2 of ans.slice(0, -1)) {
					const delta = Math.abs(a - a2);
					if (!deltas.has(delta)) {
						deltas.add(delta);
						go(a2 + delta);
						go(a2 - delta);
					}
				}
				for (const delta of Array.from(deltas).sort((x, y) => x - y)) {
					go(a + delta);
				}
			}
		};

		let a = 0;
		while (ans.length < m) {
			go(a);
			a += 1;
		}

		return ans;
	}

	genRandom () {
		const b = this.random.randrange(6, 20);
		const m = this.random.randrange(1, 100);
		this.add({ b, m });
	}
}

export class FindRepeats extends PuzzleGenerator {
	static docstring = "Find a repeating integer in an infinite sequence of integers, specifically the indices for which the same value\n        occurs 1000 times. The sequence is defined by a starting value a_0 and each subsequent term is:\n        a_{n+1} = the square root of a_n if the a_n is a perfect square, and a_n + 3 otherwise.\n\n        For a given a_0 (that is a multiple of 3), the goal is to find 1000 indices where the a_i's are all equal.\n\n        Sample input:\n        9\n\n        Sample output:\n        [0, 3, 6, ..., 2997]";
	getExample () {
		return { a0: 123 };
	}

	static sat (indices, a0 = 123) {
		/**
		 * Find a repeating integer in an infinite sequence of integers, specifically the indices for which the same value
		 * occurs 1000 times. The sequence is defined by a starting value a_0 and each subsequent term is:
		 * a_{n+1} = the square root of a_n if the a_n is a perfect square, and a_n + 3 otherwise.
		 *
		 * For a given a_0 (that is a multiple of 3), the goal is to find 1000 indices where the a_i's are all equal.
		 *
		 * Sample input:
		 * 9
		 *
		 * Sample output:
		 * [0, 3, 6, ..., 2997]
		 *
		 * The sequence starting with a0=9 is [9, 3, 6, 9, 3, 6, 9, ...] thus a_n at where n is a multiple of 3 are
		 * all equal in this case.
		 */
		if (a0 < 0 || a0 % 3 !== 0) return false;
		if (new Set(indices).size !== indices.length || indices.length !== 1000)
			return false;
		if (Math.min(...indices) < 0) return false;

		const s = [a0];
		for (let i = 0; i < Math.max(...indices); i++) {
			const sq = Math.floor(Math.sqrt(s[s.length - 1]));
			s.push(sq * sq === s[s.length - 1] ? sq : s[s.length - 1] + 3);
		}

		const values = new Set(indices.map((i) => s[i]));
		return values.size === 1;
	}

	static sol (a0) {
		let n = a0;
		const ans = [];
		let i = 0;
		// Safety limit: for valid a0 (multiple of 3), the sequence enters a 3->6->9 cycle
		// and finds 1000 occurrences of 3 in ~3000 iterations. This limit prevents
		// infinite loops if called with invalid inputs.
		const MAX_ITERATIONS = 100000;

		while (ans.length < 1000 && i < MAX_ITERATIONS) {
			if (n === 3) ans.push(i);
			const sq = Math.floor(Math.sqrt(n));
			n = sq * sq === n ? sq : n + 3;
			i += 1;
		}

		// Return null if we couldn't find 1000 indices (indicates invalid input)
		return ans.length === 1000 ? ans : null;
	}

	genRandom () {
		const a0 = 3 * this.random.randrange(1, 10 ** 6);
		this.add({ a0 });
	}
}

export class PickNearNeighbors extends PuzzleGenerator {
	static docstring = "Given a permutation of the integers up to n(n+1) as a list, choose 2n numbers to keep (in the same order)\n        so that the remaining list of numbers satisfies:\n        * its largest number is next to its second largest number\n        * its third largest number is next to its fourth largest number\n        ...\n        * its second smallest number is next to its smallest number\n\n        Sample input:\n        [4, 0, 5, 3, 1, 2]\n        n = 2\n\n        Sample output:\n        [True, False, True, False, True, True]\n\n        Keeping these indices results in the sublist [4, 5, 1, 2] where 4 and 5 are adjacent as are 1 and 2.";
	getExample () {
		return {
			heights: [
				10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4,
			],
		};
	}

	static sat (
		keep,
		heights = [
			10, 2, 14, 1, 8, 19, 16, 6, 12, 3, 17, 0, 9, 18, 5, 7, 11, 13, 15, 4,
		],
	) {
		/**
		 * Given a permutation of the integers up to n(n+1) as a list, choose 2n numbers to keep (in the same order)
		 * so that the remaining list of numbers satisfies:
		 * * its largest number is next to its second largest number
		 * * its third largest number is next to its fourth largest number
		 * ...
		 * * its second smallest number is next to its smallest number
		 *
		 * Sample input:
		 * [4, 0, 5, 3, 1, 2]
		 * n = 2
		 *
		 * Sample output:
		 * [True, False, True, False, True, True]
		 *
		 * Keeping these indices results in the sublist [4, 5, 1, 2] where 4 and 5 are adjacent as are 1 and 2.
		 */
		const n = Math.floor(Math.sqrt(heights.length));
		if (
			heights.sort((a, b) => a - b).some((v, i) => v !== i) ||
			n * (n + 1) !== heights.length
		)
			return false;

		const kept = heights.filter((_, i) => keep[i]);
		if (kept.length !== 2 * n) return false;

		const pi = Array.from({ length: 2 * n }, (_, i) => i).sort(
			(i, j) => kept[i] - kept[j],
		);
		return Array.from(
			{ length: n },
			(_, i) => Math.abs(pi[2 * i] - pi[2 * i + 1]) === 1,
		).every((v) => v);
	}

	static sol (heights) {
		const n = Math.floor(Math.sqrt(heights.length));
		const groups = heights.map((h) => Math.floor(h / (n + 1)));
		const ans = new Array(heights.length).fill(false);
		const usedGroups = new Set();

		let a = 0;
		while (ans.filter((v) => v).length < 2 * n) {
			const groupTracker = {};
			let b = a;
			while (
				!Object.hasOwn(groupTracker, groups[b]) ||
				usedGroups.has(groups[b])
			) {
				groupTracker[groups[b]] = b;
				b += 1;
			}
			ans[groupTracker[groups[b]]] = true;
			ans[b] = true;
			usedGroups.add(groups[b]);
			a = b + 1;
		}

		return ans;
	}

	genRandom () {
		const n = this.random.randrange(1, 10);
		const heights = Array.from({ length: n * (n + 1) }, (_, i) => i);
		this.random.shuffle(heights);
		this.add({ heights });
	}
}

export class FindProductiveList extends PuzzleGenerator {
	static docstring = "Given n, find n integers such that li[i] * li[i+1] + 1 == li[i+2], for i = 0, 1, ..., n-1\n        where indices >= n \"wrap around\". Note: only n multiples of 3 are given since this is only possible for n\n        that are multiples of 3 (as proven in the IMO problem).\n\n        Sample input:\n        6\n\n        Sample output:\n        [_, _, _, _, _, _]\n\n        (Sample output hidden because showing sample output would give away too much information.)";
	getExample () {
		return { n: 18 };
	}

	static sat (li, n = 18) {
		/**
		 * Given n, find n integers such that li[i] * li[i+1] + 1 == li[i+2], for i = 0, 1, ..., n-1
		 * where indices >= n "wrap around". Note: only n multiples of 3 are given since this is only possible for n
		 * that are multiples of 3 (as proven in the IMO problem).
		 *
		 * Sample input:
		 * 6
		 *
		 * Sample output:
		 * [_, _, _, _, _, _]
		 *
		 * (Sample output hidden because showing sample output would give away too much information.)
		 */
		if (n % 3 !== 0) return false;
		if (li.length !== n) return false;
		return Array.from(
			{ length: n },
			(_, i) => li[(i + 2) % n] === 1 + li[(i + 1) % n] * li[i],
		).every((v) => v);
	}

	static sol (n) {
		const cycle = [-1, -1, 2];
		const ans = [];
		for (let i = 0; i < n / 3; i++) {
			ans.push(...cycle);
		}
		return ans;
	}

	gen (targetNumInstances) {
		for (let n = 3; n < 3 * targetNumInstances + 3; n += 3) {
			this.add({ n });
		}
	}
}

export class HalfTag extends PuzzleGenerator {
	static docstring = "The input tags is a list of 4n integer tags each in range(n) with each tag occurring 4 times.\n        The goal is to find a subset (list) li of half the indices such that:\n        * The sum of the indices equals the sum of the sum of the missing indices.\n        * The tags of the chosen indices contains exactly each number in range(n) twice.\n\n        Sample input:\n        n = 3\n        tags = [0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]\n\n        Sample output:\n        [0, 3, 5, 6, 8, 11]\n\n        Note the sum of the output is 33 = (0+1+2+...+11)/2 and the selected tags are [0, 0, 1, 1, 2, 2]";
	getExample () {
		return {
			tags: [3, 0, 3, 2, 0, 1, 0, 3, 1, 1, 2, 2, 0, 2, 1, 3],
		};
	}

	static sat (li, tags = [3, 0, 3, 2, 0, 1, 0, 3, 1, 1, 2, 2, 0, 2, 1, 3]) {		/**
		 * The input tags is a list of 4n integer tags each in range(n) with each tag occurring 4 times.
		 * The goal is to find a subset (list) li of half the indices such that:
		 * * The sum of the indices equals the sum of the sum of the missing indices.
		 * * The tags of the chosen indices contains exactly each number in range(n) twice.
		 *
		 * Sample input:
		 * n = 3
		 * tags = [0, 1, 2, 0, 0, 1, 1, 1, 2, 2, 0, 2]
		 *
		 * Sample output:
		 * [0, 3, 5, 6, 8, 11]
		 *
		 * Note the sum of the output is 33 = (0+1+2+...+11)/2 and the selected tags are [0, 0, 1, 1, 2, 2]
		 */		const n = Math.max(...tags) + 1;
		const fourN = 4 * n;
		if (new Set(li).size !== li.length || Math.min(...li) < 0) return false;

		const sum = li.reduce((a, b) => a + b, 0);
		const totalSum = (fourN * (fourN - 1)) / 2;
		if (sum * 2 !== totalSum) return false;

		const selectedTags = li.map((i) => tags[i]).sort((a, b) => a - b);
		const expectedTags = Array.from({ length: 2 * n }, (_, i) =>
			Math.floor(i / 2),
		);
		return JSON.stringify(selectedTags) === JSON.stringify(expectedTags);
	}

	static sol (tags) {
		const n = Math.max(...tags) + 1;
		const fourN = 4 * n;
		const pairs = new Set();
		const pairsList = []; // Keep a separate list for ordering
		for (let i = 0; i < 2 * n; i++) {
			const pair = [i, fourN - i - 1];
			pairs.add(JSON.stringify(pair));
			pairsList.push(pair);
		}

		const byTag = {};
		for (let tag = 0; tag < n; tag++) {
			byTag[tag] = [];
		}

		for (const pair of pairsList) {
			const [a, b] = pair;
			const tagA = tags[a];
			const tagB = tags[b];
			byTag[tagA].push(pair);
			byTag[tagB].push(pair);
		}

		const cycles = [];
		let cycle = [];
		let tag; // Declare tag here so it's available in the while loop

		while (pairs.size > 0) {
			if (cycle.length === 0) {
				const pStr = Array.from(pairs)[0];
				const [a] = JSON.parse(pStr);
				tag = tags[a];
			}

			if (!byTag[tag] || byTag[tag].length === 0) {
				// Tag exhausted, move to next available pair
				if (pairs.size === 0) break;
				const pStr = Array.from(pairs)[0];
				const [a] = JSON.parse(pStr);
				tag = tags[a];
				cycle = [];
			}

			const pair = byTag[tag].pop();
			if (!pair) break; // Safety check
			const [a, b] = pair;
			const pStr = JSON.stringify(pair);
			const tagA = tags[a];
			const tagB = tags[b];
			tag = tagA === tag ? tagB : tagA;
			byTag[tag] = byTag[tag].filter((p) => JSON.stringify(p) !== pStr);
			cycle.push(tag === tagB ? [a, b] : [b, a]);
			pairs.delete(pStr);

			if (byTag[tag].length === 0) {
				cycles.push(cycle);
				cycle = [];
			}
		}

		while (cycles.some((c) => c.length % 2 !== 0)) {
			let merged = false;
			for (let i = 0; i < cycles.length && !merged; i++) {
				for (let j = 0; j < i; j++) {
					const tagsI = new Set();
					const tagsJ = new Set();
					cycles[i].forEach(([a, b]) => {
						tagsI.add(tags[a]);
						tagsI.add(tags[b]);
					});
					cycles[j].forEach(([a, b]) => {
						tagsJ.add(tags[a]);
						tagsJ.add(tags[b]);
					});

					const intersection = Array.from(tagsI).find((t) => tagsJ.has(t));
					if (intersection) {
						const cycleI = cycles.splice(i, 1)[0];
						const i1 = cycleI.findIndex(([a]) => tags[a] === intersection);
						const j1 = cycles[j].findIndex(([a]) => tags[a] === intersection);
						cycles[j].splice(
							j1,
							0,
							...cycleI.slice(i1),
							...cycleI.slice(0, i1),
						);
						merged = true;
						break;
					}
				}
			}
		}

		const ans = [];
		for (const c of cycles) {
			for (let i = 1; i < c.length; i += 2) {
				ans.push(...c[i]);
			}
		}

		return ans;
	}

	genRandom () {
		const n = this.random.randrange(1, 10);
		const tags = Array.from({ length: 4 * n }, (_, i) => Math.floor(i / 4));
		this.random.shuffle(tags);
		this.add({ tags });
	}
}
