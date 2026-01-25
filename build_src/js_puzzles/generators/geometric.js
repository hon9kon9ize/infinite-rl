import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

export class GeometricSequence extends PuzzleGenerator {
	static docstring = "Create a list that is a subrange of an gemoetric sequence.";

	static tags = [Tags.math];

	getExample () {
		return { a: 8, r: 2, l: 50 };
	}

	static sat (answer, a, r, l) {
		/** Create a list that is a subrange of an gemoetric sequence. */
		if (!Array.isArray(answer) || answer.length !== l) return false;
		if (answer[0] !== a) return false;
		for (let i = 0; i < l - 1; i++)
			if (answer[i] * r !== answer[i + 1]) return false;
		return true;
	}

	static sol (a, r, l) {
		const ans = [];
		let current = a;
		for (let i = 0; i < l; i++) {
			ans.push(current);
			current = current * r;
		}
		return ans;
	}

	genRandom () {
		const a = Math.floor(this.random() * 2000) - 1000;
		const r = Math.floor(this.random() * 5) + 1;
		const l = Math.floor(this.random() * 100) + 1;
		this.add({ a, r, l });
	}
}
