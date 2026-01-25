import { PuzzleGenerator } from "../puzzle_generator.js";

export class Tutorial1 extends PuzzleGenerator {
	static docstring = "* Find a string that when concatenated onto 'Hello ' gives 'Hello world'.";
	constructor(seed = null) {
		super(seed);
	}
	getExample () {
		return {};
	}
	static sat (s) {
		/** Find a string that when concatenated onto 'Hello ' gives 'Hello world'. */
		return "Hello " + s === "Hello world";
	}
	static sol () {
		return "world";
	}
}

export class Tutorial2 extends PuzzleGenerator {
	static docstring = "* Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'.";
	constructor(seed = null) {
		super(seed);
	}
	static sat (s) {
		/** Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'. */
		return "Hello " + s.split("").reverse().join("") === "Hello world";
	}
	static sol () {
		return "world".split("").reverse().join("");
	}
}

export class Tutorial3 extends PuzzleGenerator {
	static docstring = "* Find a list of two integers whose sum is 3.";
	constructor(seed = null) {
		super(seed);
	}
	static sat (x) {
		/** Find a list of two integers whose sum is 3. */
		return Array.isArray(x) && x.length === 2 && x[0] + x[1] === 3;
	}
	static sol () {
		return [1, 2];
	}
}

export class Tutorial4 extends PuzzleGenerator {
	static docstring = "* Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'.";
	constructor(seed = null) {
		super(seed);
	}
	static sat (s) {
		/** Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'. */
		if (!Array.isArray(s) || new Set(s).size !== 1000) return false;
		return s.every(
			(x) =>
				x.split("a").length - 1 > x.split("b").length - 1 && x.includes("b"),
		);
	}
	static sol () {
		return Array.from({ length: 1000 }, (_, i) => "a".repeat(i + 2) + "b");
	}
}

export class Tutorial5 extends PuzzleGenerator {
	static docstring = "* Find an integer whose perfect square begins with 123456789 in its decimal representation.";
	constructor(seed = null) {
		super(seed);
	}
	static sat (n) {
		/** Find an integer whose perfect square begins with 123456789 in its decimal representation. */
		return String(n * n).startsWith("123456789");
	}
	static sol () {
		return Math.floor(Math.sqrt(Number("123456789" + "0".repeat(9)))) + 1;
	}
}
