export function makeLLMInstruction(problem, inputs, longThreshold = 200) {
	const body = {
		problem: problem.name,
		description: problem.description || "",
		inputs,
		answer_type: problem.answer_type || "unknown",
	};

	return `You are given a programming puzzle. Return a single JSON object (one line) with keys describing the answer.\n- If the answer is a very long string (> ${longThreshold} chars), encode it using base64 and set \"encoding\" to \"base64\".\n- Include \"answer_preview\" (first 50 chars + '...' + last 50 chars) and \"answer_length\" (int).\n- If encoding is not needed, set \"encoding\" to null and return the \"answer\" as the raw JSON value.\n\nProblem metadata:\n${JSON.stringify(body, null, 2)}\n`;
}

export function validateLLMAnswer(problem, inputs, llmOutput) {
	let obj;
	try {
		obj = JSON.parse(llmOutput);
	} catch (e) {
		return { ok: false, error: `Invalid JSON: ${e}` };
	}

	if (!("answer" in obj)) return { ok: false, error: "Missing 'answer' key" };

	const enc = obj.encoding || null;
	let val = obj.answer;

	if (enc === "base64") {
		try {
			val = Buffer.from(obj.answer, "base64").toString("utf8");
		} catch (e) {
			return { ok: false, error: `Base64 decode failed: ${e}` };
		}
	}

	// Prepare preview and length metadata
	const preview =
		obj.answer_preview ||
		(typeof val === "string"
			? val.slice(0, 50) + (val.length > 50 ? "..." : "")
			: JSON.stringify(val).slice(0, 50));
	const length =
		obj.answer_length || (typeof val === "string" ? val.length : null);

	try {
		const satOk = problem.sat(val, inputs);
		if (!satOk)
			return { ok: false, error: "sat returned false", preview, length };
	} catch (e) {
		return { ok: false, error: `Verification error: ${e}`, preview, length };
	}

	return { ok: true, preview, length };
}
