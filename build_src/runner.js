const input = JSON.parse(readStdin());
const { puzzle, code, inputs } = input;

const result = evalPuzzle(puzzle, code, inputs);
writeOutput(result);

// =======================
// Puzzle evaluation
// =======================

function evalPuzzle (puzzle, code, inputs) {
  try {
    // Eval the user's code in global scope
    const modifiedCode = code.replace('function sol', 'globalThis.sol = function');
    globalThis.eval(modifiedCode);

    // Call sol with inputs unpacked
    const result = globalThis.sol(...Object.values(inputs));

    return { result };
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