import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

export class ConcatStrings extends PuzzleGenerator {
	static docstring = "Concatenate the list of characters in s";

	static tags = [Tags.strings];

	getExample () {
		return { s: ["a", "b", "c", "d", "e", "f"], n: 4 };
	}

	/**
	 * Concatenate the list of characters in s
	 */
	static sat (answer, s, n) {
		if (typeof answer !== "string") return false;
		if (answer.length !== n) return false;
		for (let i = 0; i < n; i++) if (answer[i] !== s[i]) return false;
		return true;
	}

	static sol (s, n) {
		return s.slice(0, n).join("");
	}

	genRandom () {
		const n = Math.floor(this.random() * 10);
		const extra = Math.floor(this.random() * 10);
		const s = [];
		const chars = "abcdefghijklmnopqrstuvwxyz";
		for (let i = 0; i < n + extra; i++)
			s.push(chars[Math.floor(this.random() * chars.length)]);
		this.add({ s, n });
	}
}
