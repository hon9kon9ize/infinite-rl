/**
 * Each PuzzleGenerator has one or more Instances corresponding to different inputs. A "simple" problem like
 * checking if a string equals "Hello world" when concatenated has just one instance.
 */

import seedrandom from "./seedrandom.js";

const DEFAULT_CHARSET =
	"0123456789abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ,.:|/;?[]<>-=()+*&^%$#@!";

export const Tags = {
	brute_force: "brute_force", // can be solved by brute force
	codeforces: "codeforces", // inspired by a codeforces.com problem
	data_structures: "data_structures", // needs fancy data structures to solve
	dp: "dp", // dynamic programming
	famous: "famous", // roughly at the level that there is a Wikipedia page about the topic
	games: "games", // puzzle describes a game
	graphs: "graphs", // solution can use graph algorithms like depth-first search, etc.
	greedy: "greedy", // solution can use a greedy algorithm
	hard: "hard", // challenging
	human_eval: "human_eval", // inspired by a problem from the Human Eval dataset
	unsolved: "unsolved", // unsolved, involves open some unsolved puzzles
	math: "math", // mainly mathematical reasoning
	strings: "strings", // involves constructing a string
	trivial: "trivial", // trivial *solution* even if it may require some work to understand what the puzzle is asking
};

const buildRandom = (seed) => {
	const rng = seedrandom(String(seed));
	const random = () => rng();
	random.random = random;
	const clampRange = (start, stop) => {
		if (stop <= start) return start;
		return start + Math.floor(random() * (stop - start));
	};
	random.randrange = (start, stop) => {
		if (stop === undefined) {
			stop = start;
			start = 0;
		}
		if (stop <= start) return start;
		return clampRange(start, stop);
	};
	random.uniform = (min, max) => min + random() * (max - min);
	random.choice = (seq) => {
		if (!seq || seq.length === 0) {
			throw new Error("Cannot choose from empty sequence");
		}
		return seq[Math.floor(random() * seq.length)];
	};
	random.shuffle = (arr) => {
		for (let i = arr.length - 1; i > 0; i--) {
			const j = random.randrange(i + 1);
			[arr[i], arr[j]] = [arr[j], arr[i]];
		}
		return arr;
	};
	random.sample = (arr, k) => {
		const copy = [...arr];
		random.shuffle(copy);
		return copy.slice(0, k);
	};
	random.char = (chars = DEFAULT_CHARSET) => random.choice(chars);
	random.pseudo_word = (minLen = 1, maxLen = 20) => {
		const consonants = [
			"text",
			"th",
			"ch",
			"qu",
			"b",
			"c",
			"d",
			"f",
			"g",
			"h",
			"j",
			"k",
			"l",
			"m",
			"n",
			"p",
			"r",
			"s",
			"t",
			"v",
			"w",
			"x",
			"z",
		];
		const vowels = "aeiyou";
		let word = "";
		for (let i = 0; i < 1 + Math.floor(maxLen / 2); i++) {
			word += random.choice(consonants);
			word += random.choice(vowels);
		}
		const upper = random.randrange(minLen, maxLen + 1);
		return word.slice(0, upper);
	};
	// Camelcase aliases
	random.pseudoWord = random.pseudo_word;
	random.string = (...args) => {
		// Handle both calling styles: string(options) and string(minLen, maxLen)
		let minLen = 0,
			maxLen = 10,
			chars = DEFAULT_CHARSET;
		const options = args[0];

		if (typeof options === "number") {
			// Called as string(minLen, maxLen) - second arg is maxLen
			minLen = options;
			maxLen = args[1] !== undefined ? args[1] : 10;
		} else if (typeof options === "object" && options !== null) {
			// Called as string({ minLen, maxLen, chars })
			minLen = options.minLen ?? 0;
			maxLen = options.maxLen ?? 10;
			chars = options.chars ?? DEFAULT_CHARSET;
		}

		const len = random.randrange(minLen, maxLen + 1);
		let result = "";
		for (let i = 0; i < len; i++) {
			result += random.char(chars);
		}
		return result;
	};
	return random;
};

export class PuzzleGenerator {
	/**
	 * PuzzleGenerator is an abstract class for a puzzle generator which builds 1 or more instances.
	 * Each problem MUST OVERRIDE sat. Examples:
	 * 
	 * class HelloWorld extends PuzzleGenerator {
	 *   static tags = [Tags.strings];
	 *   static docstring = "Find a string that when concatenated onto 'Hello ' gives 'Hello world'.";
	 *   
	 *   static sat(s) {
	 *     return "Hello " + s === "Hello world";
	 *   }
	 * }
	 * 
	 * class BackWorlds extends PuzzleGenerator {
	 *   static tags = [Tags.strings];
	 *   static docstring = "Find a string that when reversed and concatenated gives 'Hello world'.";
	 *   
	 *   static sat(s) {
	 *     return s.split('').reverse().join('') + 'world' === 'Hello world';
	 *   }
	 *   
	 *   static sol() {
	 *     return 'dlroW olleH'.split('').reverse().join(''); // 'Hello World'
	 *   }
	 * }
	 */
	constructor(seed = null) {
		this.name = this.constructor.name;
		this.tags = this.constructor.tags || [];
		this.docstring = this.constructor.docstring || "";
		this.instances = [];
		this._inputs = [];
		this._seen = new Set();
		this._initSeed = seed ?? this.name;
		this.random = buildRandom(this._initSeed);
	}

	num_generated_so_far () {
		return this._inputs.length;
	}

	// Default example: override in subclass
	getExample () {
		return {};
	}

	/**
	 * Must override. Returns true if the answer satisfies the puzzle constraints.
	 * @param {any} answer - The proposed solution
	 * @param {...any} args - The puzzle inputs
	 * @returns {boolean} - True if the answer is correct
	 */
	sat (answer, ...args) {
		// If the class has a static sat method, use that
		if (this.constructor.sat) {
			// If a single argument is a plain inputs object, try to match the static method's parameters
			if (
				args.length === 1 &&
				typeof args[0] === "object" &&
				args[0] !== null &&
				!Array.isArray(args[0]) &&
				Object.getPrototypeOf(args[0]) === Object.prototype
			) {
				// Spread the object values
				return this.constructor.sat(answer, ...Object.values(args[0]));
			}
			return this.constructor.sat(answer, ...args);
		}
		throw new Error("sat must be implemented by subclass");
	}

	/**
	 * Reference solution; must override. Returns the correct answer for the given inputs.
	 * @param {...any} args - The puzzle inputs
	 * @returns {any} - The correct solution
	 */
	sol (...args) {
		// If the class has a static sol method, use that
		if (this.constructor.sol) {
			// If a single argument is a plain inputs object, try to match the static method's parameters
			if (
				args.length === 1 &&
				typeof args[0] === "object" &&
				args[0] !== null &&
				!Array.isArray(args[0]) &&
				Object.getPrototypeOf(args[0]) === Object.prototype
			) {
				// Spread the object values
				return this.constructor.sol(...Object.values(args[0]));
			}
			return this.constructor.sol(...args);
		}
		throw new Error("sol must be implemented by subclass");
	}

	/**
	 * Adds a new input instance to the generator, avoiding duplicates.
	 * @param {Object} inp - The input parameters for a puzzle instance
	 */
	add (inp) {
		const key = JSON.stringify(inp, (key, value) =>
			typeof value === "bigint" ? value.toString() : value,
		);
		if (this._seen.has(key)) return;
		this._seen.add(key);
		this._inputs.push(inp);
	}

	// Called by build to generate instances
	gen (target_num_instances) { }
	genRandom () { }

	reseed (seed = null) {
		if (seed !== null) this._initSeed = seed;
		this.random = buildRandom(this._initSeed);
	}

	/**
	 * Builds puzzle instances by generating inputs and computing solutions.
	 * @param {number} targetNumInstances - Number of instances to generate (default: 1)
	 * @param {number} maxRandomAttempts - Max attempts for random generation (default: 100)
	 * @returns {Array} - Array of puzzle instances with inputs and solutions
	 */
	build (targetNumInstances = 1, maxRandomAttempts = 100) {
		this._seen = new Set();
		this._inputs = [];
		this.reseed();

		// include example by default (unless skipExample is set)
		if (!this.skipExample) {
			this.add(this.getExample());
		}

		// Always try gen() first, like Python does
		if (this._inputs.length < targetNumInstances) {
			this.gen(targetNumInstances - this._inputs.length);
		}

		while (this._inputs.length < targetNumInstances) {
			const before = this._inputs.length;
			for (let i = 0; i < maxRandomAttempts; i++) {
				this.genRandom();
				if (this._inputs.length !== before) break;
			}
			if (this._inputs.length === before) break; // give up
		}

		this.instances = this._inputs.map((inp, idx) => ({
			name: `${this.name}:${idx}`,
			inputs: inp,
			solHeader: this.createSolHeader(inp),
			solution: this.sol(inp),
		}));

		return this.instances;
	}

	createSolHeader (inp) {
		const parts = Object.entries(inp).map(
			([k, v]) =>
				`${k}=${JSON.stringify(typeof v === "bigint" ? v.toString() : v)}`,
		);
		return `function sol(${parts.join(", ")}) {}`;
	}

	// Helpers used by generators
	/**
	 * Generates a random word for testing purposes.
	 * @param {number} maxLen - Maximum length of the word (default: 6)
	 * @returns {string} - A random lowercase word
	 */
	randomChoiceWords (maxLen = 6) {
		const len = Math.floor(this.random() * maxLen) + 1;
		const chars = "abcdefghijklmnopqrstuvwxyz";
		let w = "";
		for (let i = 0; i < len; i++)
			w += chars[Math.floor(this.random() * chars.length)];
		return w;
	}
}
