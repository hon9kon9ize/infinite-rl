const source = readStdin();
const result = evalUserCode(source);
writeOutput(result);

// =======================
// Eval logic
// =======================

function evalUserCode (code) {
  /*
    The evaluated code SHOULD return a value, e.g.:

    ({ x: 1 + 2 })
    or
    (() => 42)()
  */

  try {
    return eval(code);
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