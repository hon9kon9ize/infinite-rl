import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

/**
 * Roots of polynomials
 */

function polyEval (coeffs, x) {
	// coeffs are [a,b,c] for quadratic or [a,b,c,d] for cubic (descending powers)
	return coeffs.reduce(
		(acc, c, i) => acc + c * x ** (coeffs.length - 1 - i),
		0,
	);
}

function findSignChangeRoot (coeffs, left = -1e3, right = 1e3, maxIter = 60) {
	// coarse scan to find an interval with sign change
	const steps = 400;
	let prevX = left;
	let prevV = polyEval(coeffs, prevX);
	for (let i = 1; i <= steps; i++) {
		const x = left + (right - left) * (i / steps);
		const v = polyEval(coeffs, x);
		if (Math.abs(v) < 1e-12) return x;
		if (prevV * v < 0) {
			// bisection
			let a = prevX,
				b = x,
				fa = prevV,
				fb = v;
			for (let it = 0; it < maxIter; it++) {
				const m = 0.5 * (a + b);
				const fm = polyEval(coeffs, m);
				if (Math.abs(fm) < 1e-12) return m;
				if (fa * fm <= 0) {
					b = m;
					fb = fm;
				} else {
					a = m;
					fa = fm;
				}
			}
			return 0.5 * (a + b);
		}
		prevX = x;
		prevV = v;
	}
	// fallback: try Newton from 0
	try {
		let x = 0;
		for (let it = 0; it < 100; it++) {
			const f = polyEval(coeffs, x);
			if (Math.abs(f) < 1e-9) return x;
			// derivative
			let df = 0;
			for (let i = 0; i < coeffs.length - 1; i++)
				df +=
					coeffs[i] * (coeffs.length - 1 - i) * x ** (coeffs.length - 2 - i);
			if (df === 0) break;
			x -= f / df;
		}
		if (Math.abs(polyEval(coeffs, x)) < 1e-6) return x;
	} catch (e) { }
	return null;
}

export class QuadraticRoot extends PuzzleGenerator {
	static tags = [Tags.math, Tags.famous];
	static docstring = "Find any (real) solution to:  a x^2 + b x + c where coeffs = [a, b, c].\n        For example, since x^2 - 3x + 2 has a root at 1, sat(x = 1., coeffs = [1., -3., 2.]) is True.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math, Tags.famous];
	}

	getExample () {
		return { coeffs: [2.5, 1.3, -0.5] };
	}

	/**
	 * Find any (real) solution to: a x^2 + b x + c where coeffs = [a, b, c].
	 * For example, since x^2 - 3x + 2 has a root at 1, sat(x = 1., coeffs = [1., -3., 2.]) is True.
	 */
	static sat (x, coeffs = [2.5, 1.3, -0.5]) {
		return Math.abs(polyEval(coeffs, x)) < 1e-6;
	}

	static sol (coeffs) {
		if (typeof coeffs === "object" && !Array.isArray(coeffs))
			({ coeffs } = coeffs);
		const [a, b, c] = coeffs;
		if (Math.abs(a) < 1e-12) {
			return b !== 0 ? -c / b : 0.0;
		}
		const disc = b * b - 4 * a * c;
		if (disc < 0) return null;
		return (-b + Math.sqrt(disc)) / (2 * a);
	}
}

export class AllQuadraticRoots extends PuzzleGenerator {
	static tags = [Tags.math, Tags.famous];
	static docstring = "Find all (real) solutions to: x^2 + b x + c (i.e., factor into roots), here coeffs = [b, c]";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math, Tags.famous];
	}
	getExample () {
		return { coeffs: [1.3, -0.5] };
	}

	/**
	 * Find all (real) solutions to: x^2 + b x + c (i.e., factor into roots), here coeffs = [b, c].
	 */
	static sat (roots, coeffs = [1.3, -0.5]) {
		const [b, c] = coeffs;
		const [r1, r2] = roots;
		return Math.abs(r1 + r2 + b) + Math.abs(r1 * r2 - c) < 1e-6;
	}

	/**
	 * Reference solution for AllQuadraticRoots.
	 */
	static sol (coeffs) {
		// Handle dict input like {coeffs: [b, c]} or direct array input
		if (typeof coeffs === "object" && !Array.isArray(coeffs)) {
			({ coeffs } = coeffs);
		}
		const [b, c] = coeffs;
		const delta = Math.sqrt(b * b - 4 * c);
		return [(-b + delta) / 2, (-b - delta) / 2];
	}
}

export class CubicRoot extends PuzzleGenerator {
	static docstring = "See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula).";
	static tags = [Tags.math, Tags.famous];

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math, Tags.famous];
	}
	getExample () {
		return { coeffs: [2.0, 1.0, 0.0, 8.0] };
	}

	/**
	 * Find any (real) solution to: a x^3 + b x^2 + c x + d where coeffs = [a, b, c, d].
	 * For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6, sat(x = 1., coeffs = [-6., 11., -6.]) is True.
	 */
	static sat (x, coeffs = [2.0, 1.0, 0.0, 8.0]) {
		return Math.abs(polyEval(coeffs, x)) < 1e-6;
	}

	/**
	 * Reference solution for CubicRoot.
	 */
	static sol (coeffs) {
		if (typeof coeffs === "object" && !Array.isArray(coeffs))
			({ coeffs } = coeffs);
		// find any real root using sign changes / bisection
		return findSignChangeRoot(coeffs);
	}

	genRandom () {
		const x = this.randomRange(-10, 10);
		const a = this.random() * 2 - 1;
		const b = this.random() * 2 - 1;
		const c = this.random() * 2 - 1;
		const d = -(a * x * x * x + b * x * x + c * x);
		const coeffs = [a, b, c, d];
		if (this.sol(coeffs) !== null) this.add({ coeffs });
	}
}

export class AllCubicRoots extends PuzzleGenerator {
	static docstring = "See [cubic equation](https://en.wikipedia.org/wiki/Cubic_formula).";
	static tags = [Tags.math, Tags.famous];

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.math, Tags.famous];
	}
	getExample () {
		return { coeffs: [1.0, -2.0, -1.0] };
	}

	/**
	 * Find all 3 distinct real roots of x^3 + a x^2 + b x + c, i.e., factor into (x-r1)(x-r2)(x-r3).
	 * coeffs = [a, b, c]. For example, since (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6,
	 * sat(roots = [1., 2., 3.], coeffs = [-6., 11., -6.]) is True.
	 */
	static sat (roots, coeffs = [1.0, -2.0, -1.0]) {
		const [r1, r2, r3] = roots;
		const [a, b, c] = coeffs;
		return (
			Math.abs(r1 + r2 + r3 + a) +
			Math.abs(r1 * r2 + r1 * r3 + r2 * r3 - b) +
			Math.abs(r1 * r2 * r3 + c) <
			1e-4
		);
	}

	/**
	 * Reference solution for AllCubicRoots.
	 */
	static sol (coeffs) {
		// Handle dict input like {coeffs: [...]} or direct array input
		if (typeof coeffs === "object" && !Array.isArray(coeffs)) {
			({ coeffs } = coeffs);
		}
		// find up to 3 real roots by scanning intervals and bisection
		const roots = new Set();
		const steps = 600;
		const L = -500,
			R = 500;
		let prevX = L,
			prevV = polyEval(coeffs, L);
		for (let i = 1; i <= steps; i++) {
			const x = L + (R - L) * (i / steps);
			const v = polyEval(coeffs, x);
			if (Math.abs(v) < 1e-9) roots.add(Number(x.toFixed(9)));
			if (prevV * v < 0) {
				// root in (prevX,x)
				let a = prevX,
					b = x,
					fa = prevV,
					fb = v;
				for (let it = 0; it < 80; it++) {
					const m = (a + b) / 2;
					const fm = polyEval(coeffs, m);
					if (Math.abs(fm) < 1e-9) {
						a = m;
						b = m;
						break;
					}
					if (fa * fm <= 0) {
						b = m;
						fb = fm;
					} else {
						a = m;
						fa = fm;
					}
				}
				roots.add(Number(((a + b) / 2).toFixed(9)));
			}
			prevX = x;
			prevV = v;
			if (roots.size >= 3) break;
		}
		const out = Array.from(roots).map(Number).slice(0, 3);
		if (out.length === 3) return out;
		return null;
	}
}
