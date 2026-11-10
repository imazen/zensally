// One bounded browser parity check; no performance or crop-quality claim.
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { gzipSync, brotliCompressSync, constants } from 'node:zlib';

const [rootArg, playwrightRoot, chromiumPath] = process.argv.slice(2);
const root = path.resolve(rootArg);
const { chromium } = await import(pathToFileURL(path.join(playwrightRoot, 'index.mjs')));
const manifest = JSON.parse(await fs.readFile(path.join(root, 'inputs.json')));
const hash = b => createHash('sha256').update(b).digest('hex');
const server = http.createServer(async (req, res) => {
  try {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    const name = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
    if (name === '/') {
      res.setHeader('Content-Type', 'text/html');
      res.end('<!doctype html><title>ORT parity</title>');
      return;
    }
    const file = path.resolve(root, '.' + name);
    if (!file.startsWith(root + path.sep)) throw new Error('outside root');
    res.setHeader('Content-Type', file.endsWith('.mjs') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
    res.end(await fs.readFile(file));
  } catch { res.statusCode = 404; res.end(); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(3982, '127.0.0.1', resolve); });
let browser;
try {
  browser = await chromium.launch({ executablePath: chromiumPath, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const all = {};
  await fs.mkdir(path.join(root, 'outputs'));
  for (const arm of ['stock', 'reduced']) {
    const page = await browser.newPage();
    await page.goto('http://127.0.0.1:3982/');
    all[arm] = await page.evaluate(async ({ arm, fixtures }) => {
      const ort = await import('/stock/ort.wasm.bundle.min.mjs');
      ort.env.wasm.wasmPaths = new URL('/' + arm + '/', location.href).href;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      const model = await (await fetch('/yunet.ort')).arrayBuffer();
      const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'disabled' });
      const rows = [];
      for (const f of fixtures) {
        const data = new Float32Array(await (await fetch('/' + f.file)).arrayBuffer());
        const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', data, f.shape) });
        const outputs = [];
        for (const [name, tensor] of Object.entries(result)) {
          const bytes = new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
          let text = '';
          for (let i = 0; i < bytes.length; i += 16384) text += String.fromCharCode(...bytes.subarray(i, i + 16384));
          outputs.push({ name, dims: tensor.dims, base64: btoa(text) });
        }
        rows.push({ id: f.id, outputs });
      }
      await session.release();
      return rows;
    }, { arm, fixtures: manifest.fixtures });
    for (let i = 0; i < all[arm].length; i++) {
      for (const t of all[arm][i].outputs) {
        const name = `${arm}-${i}-${t.name}.f32`;
        const bytes = Buffer.from(t.base64, 'base64');
        await fs.writeFile(path.join(root, 'outputs', name), bytes);
        t.bytes = bytes.length; t.sha256 = hash(bytes); t.file = 'outputs/' + name;
        delete t.base64;
      }
    }
    await page.close();
  }
  const comparisons = [];
  for (let i = 0; i < manifest.fixtures.length; i++) {
    for (const a of all.stock[i].outputs) {
      const b = all.reduced[i].outputs.find(v => v.name === a.name);
      if (!b || JSON.stringify(a.dims) !== JSON.stringify(b.dims)) throw new Error('output shape/name mismatch');
      const aa = await fs.readFile(path.join(root, a.file));
      const bb = await fs.readFile(path.join(root, b.file));
      let maxAbs = 0, violations = 0;
      for (let j = 0; j < aa.length; j += 4) {
        const x = aa.readFloatLE(j), y = bb.readFloatLE(j);
        if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error('nonfinite output');
        const delta = Math.abs(x-y);
        maxAbs = Math.max(maxAbs, delta);
        violations += delta > 1e-5 + 1e-5 * Math.abs(x);
      }
      comparisons.push({ id: manifest.fixtures[i].id, output: a.name, scalars: aa.length/4,
        bitsExact: aa.equals(bb), maxAbs, toleranceViolations: violations });
    }
  }
  const sizes = [];
  for (const arm of ['stock', 'reduced']) {
    for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
      const bytes = await fs.readFile(path.join(root, arm, name));
      const gz = gzipSync(bytes, { level: 9 });
      const br = brotliCompressSync(bytes, { params: { [constants.BROTLI_PARAM_QUALITY]: 9 } });
      await fs.writeFile(path.join(root, arm, name + '.gz'), gz);
      await fs.writeFile(path.join(root, arm, name + '.br'), br);
      sizes.push({ arm, name, bytes: bytes.length, gzip9: gz.length, brotli9: br.length, sha256: hash(bytes) });
    }
  }
  const result = { ortVersion: manifest.ort_version, browser: await browser.version(), comparisons, sizes,
    outputs: all, apiCalls: 0, scope: 'three fixed BGR tensors through same ORT-format model; no GPU, detector postprocessing or latency qualification' };
  await fs.writeFile(path.join(root, 'browser-results.json'), JSON.stringify(result, null, 2)+'\n');
  if (comparisons.some(r => r.toleranceViolations)) throw new Error('raw numeric diagnostic tolerance failed');
  console.log(JSON.stringify({ outputs: comparisons.length, allBitsExact: comparisons.every(r => r.bitsExact), maxAbs: Math.max(...comparisons.map(r => r.maxAbs)), sizes }));
} catch (error) {
  await fs.writeFile(path.join(root, 'browser-failure.json'), JSON.stringify({ error: String(error.stack || error) }, null, 2));
  throw error;
} finally {
  if (browser) await browser.close();
  server.close();
}
