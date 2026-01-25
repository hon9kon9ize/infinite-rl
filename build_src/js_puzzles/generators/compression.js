/**
 * Puzzles relating to de/compression.
 */

import { PuzzleGenerator, Tags } from "../puzzle_generator.js";

// Helpers
function compressLZW (text) {
	const index = new Map();
	for (let i = 0; i < 256; i++) index.set(String.fromCharCode(i), i);
	const seq = [];
	let buffer = "";
	for (const c of text) {
		if (index.has(buffer + c)) {
			buffer += c;
			continue;
		}
		seq.push(index.get(buffer));
		index.set(buffer + c, index.size + 1);
		buffer = c;
	}
	if (text !== "") seq.push(index.get(buffer));
	return seq;
}

function decompressLZW (seq) {
	const index = Array.from({ length: 256 }, (_, i) => String.fromCharCode(i));
	const pieces = [""];
	for (const i of seq) {
		pieces.push(
			i === index.length
				? pieces[pieces.length - 1] + pieces[pieces.length - 1][0]
				: index[i],
		);
		index.push(pieces[pieces.length - 2] + pieces[pieces.length - 1][0]);
	}
	return pieces.join("");
}

export class LZW extends PuzzleGenerator {
	static docstring = "Find a (short) compression that decompresses to the given string for the provided implementation of the\n        Lempel-Ziv decompression algorithm from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.strings, Tags.famous];
	}
	/**
	 * Find a (short) compression that decompresses to the given string for the provided implementation of the Lempel-Ziv decompression algorithm from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
	 */
	static sat (
		seq,
		compressed_len = 17,
		text = "Hellooooooooooooooooooooo world!",
	) {
		/**
		 * Find a (short) compression that decompresses to the given string for the provided implementation of the
		 * Lempel-Ziv decompression algorithm from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
		 */
		const index = Array.from({ length: 256 }, (_, i) => String.fromCharCode(i));
		const pieces = [""];
		for (const i of seq) {
			pieces.push(
				i === index.length
					? pieces[pieces.length - 1] + pieces[pieces.length - 1][0]
					: index[i],
			);
			index.push(pieces[pieces.length - 2] + pieces[pieces.length - 1][0]);
		}
		return pieces.join("") === text && seq.length <= compressed_len;
	}

	/**
	 * Reference solution for LZW.
	 */
	static sol (compressed_len, text) {
		if (typeof compressed_len === "object") {
			({ compressed_len, text } = compressed_len);
		}
		text = text || "Hellooooooooooooooooooooo world!";
		return compressLZW(text);
	}
	gen () {
		this.add({ text: "", compressed_len: 0 });
		this.add({
			text: "c".repeat(1000),
			compressed_len: compressLZW("c".repeat(1000)).length,
		});
	}
	genRandom () {
		const max_len = this.randomChoice([10, 100, 1000]);
		const text = this.pseudoWord(0, max_len);
		this.add({ text, compressed_len: compressLZW(text).length });
	}
}

export class PackingHam extends PuzzleGenerator {
	static docstring = "Pack a certain number of binary strings so that they have a minimum hamming distance between each other.";

	constructor(seed = null) {
		super(seed);
		this.tags = [Tags.strings, Tags.famous];
	}
	/**
	 * Pack a certain number of binary strings so that they have a minimum hamming distance between each other.
	 */
	static sat (words, num = 100, bits = 100, dist = 34) {
		/**
		 * Pack a certain number of binary strings so that they have a minimum hamming distance between each other.
		 */
		if (!Array.isArray(words) || words.length !== num) throw new Error("bad");
		for (const w of words)
			if (w.length !== bits || /[^01]/.test(w)) throw new Error("bad");
		for (let i = 0; i < num; i++)
			for (let j = 0; j < i; j++) {
				const d = hamming(words[i], words[j]);
				if (d < dist) return false;
			}
		return true;
	}

	/**
	 * Reference solution for PackingHam.
	 */
	static sol (num, bits, dist) {
		if (typeof num === "object") ({ num, bits, dist } = num);
		num = num || 10;
		bits = bits || 6;
		dist = dist || 1;
		const r = mulberry32(0);
		while (true) {
			const seqs = Array.from({ length: num }, () =>
				Math.floor(r() * 2 ** bits),
			);
			if (
				seqs.every((v, i) =>
					seqs.slice(0, i).every((w) => hammingBin(w, v) >= dist),
				)
			) {
				return seqs.map((s) => s.toString(2).padStart(bits, "0"));
			}
		}
	}
	genRandom () {
		const bits = this.randomRange(1, this.randomChoice([10, 100]));
		const num = this.randomRange(2, this.randomChoice([10, 100]));
		const candidates = Array.from({ length: 5 }, () =>
			Array.from({ length: num }, () => Math.floor(this.random() * 2 ** bits)),
		);
		const score = (arr) =>
			Math.min(
				...arr.flatMap((a, i) => arr.slice(0, i).map((b) => hammingBin(a, b))),
			);
		const best = candidates.reduce((a, b) => (score(a) <= score(b) ? a : b));
		const dist = score(best);
		if (dist > 0) this.add({ num, bits, dist });
	}
}

function hamming (a, b) {
	return hammingBin(parseInt(a, 2), parseInt(b, 2));
}
function hammingBin (x, y) {
	return (x ^ y).toString(2).replace(/0/g, "").length;
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
