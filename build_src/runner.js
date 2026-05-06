const input = JSON.parse(readStdin());
const { puzzle, code, inputs, sat } = input;

const result = evalPuzzle(puzzle, code, inputs, sat);
writeOutput(result);

// =======================
// Puzzle evaluation
// =======================

function evalPuzzle (puzzle, code, inputs, satSource) {
  try {
    // Eval the user's code in global scope.
    globalThis.__infiniteRlSol = undefined;
    globalThis.__infiniteRlSat = undefined;

    const modifiedCode = code.replace(
      /function\s+sol\s*\(/,
      'globalThis.__infiniteRlSol = function ('
    );
    globalThis.eval(modifiedCode);

    if (typeof globalThis.__infiniteRlSol !== 'function') {
      return { error: 'sol function not defined in submitted code' };
    }

    // Call sol with inputs unpacked
    const result = globalThis.__infiniteRlSol(...Object.values(inputs || {}));

    if (!satSource) {
      return { result };
    }

    const modifiedSat = satSource.replace(
      /function\s+sat\s*\(/,
      'globalThis.__infiniteRlSat = function ('
    );
    globalThis.eval(modifiedSat);

    if (typeof globalThis.__infiniteRlSat !== 'function') {
      return { error: 'sat function not defined for JavaScript puzzle' };
    }

    const isCorrect = globalThis.__infiniteRlSat(
      result,
      ...Object.values(inputs || {})
    ) === true;

    return { result, isCorrect };
  } catch (err) {
    return {
      error: String(err),
      stack: err && err.stack ? err.stack : null
    };
  }
}

// =======================
// I/O
// =======================

function readStdin () {
  const CHUNK = 1024;
  const chunks = [];
  let total = 0;

  while (true) {
    const buf = new Uint8Array(CHUNK);
    const n = Javy.IO.readSync(0, buf);
    if (n === 0) break;
    chunks.push(buf.subarray(0, n));
    total += n;
  }

  const all = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    all.set(c, off);
    off += c.length;
  }

  return new TextDecoder().decode(all);
}

function writeOutput (value) {
  const out = new TextEncoder().encode(
    JSON.stringify(value)
  );
  Javy.IO.writeSync(1, out);
}
