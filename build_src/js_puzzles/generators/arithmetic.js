import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

export class ArithmeticSequence extends PuzzleGenerator {
	static tags = [Tags.math];
	static docstring = "Create a list that is a subrange of an arithmetic sequence.";

	getExample () {
		return { a: 7, s: 5, e: 200 };
	}

	/**
	 * Create a list that is a subrange of an arithmetic sequence.
	 */
	static sat (answer, a, s, e) {
		if (!Array.isArray(answer) || answer.length === 0) return false;
		if (answer[0] !== a) return false;
		const last = answer[answer.length - 1];
		if (last > e) return false;
		if (last + s <= e) return false;
		for (let i = 0; i < answer.length - 1; i++) {
			if (answer[i] + s !== answer[i + 1]) return false;
		}
		return true;
	}

	static sol (a, s, e) {
		const ans = [];
		for (let x = a; x <= e; x += s) ans.push(x);
		return ans;
	}

	genRandom () {
		const a = Math.floor(this.random() * 1000) - 500;
		const e = a + Math.floor(this.random() * 1000);
		const s = Math.floor(this.random() * 10) + 1;
		this.add({ a, s, e });
	}
}
