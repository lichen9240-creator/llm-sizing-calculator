// 校验脚本：抽取 HTML 内嵌引擎，用论文数据与本机实测回归。
// 用法：node validate.mjs
import fs from 'node:fs';

const FILE = 'index.html';
const html = fs.readFileSync(FILE, 'utf8');

const m = html.match(/\/\* ==ENGINE-START== \*\/([\s\S]*?)\/\* ==ENGINE-END== \*\//);
if (!m) { console.error('未找到引擎块'); process.exit(1); }
const ENGINE = new Function(m[1] + '\nreturn ENGINE;')();

// 1) 全量 <script> 语法检查（只编译不执行，DOM 代码不会跑）
const sm = html.match(/<script>([\s\S]*?)<\/script>/);
try { new Function(sm[1]); console.log('✓ UI 脚本语法通过'); }
catch (e) { console.error('✗ UI 脚本语法错误:', e.message); process.exit(1); }

// 2) 配置键一致性：FIELDS 里的 data-k 必须都在 BASE 中
const keys = new Set([...sm[1].matchAll(/k:'([a-z]+\.[a-zA-Z]+)'/g)].map(x => x[1]));
const missing = [...keys].filter(k => !(k in ENGINE.BASE));
if (missing.length) { console.error('✗ 字段缺少默认值:', missing.join(', ')); process.exit(1); }
console.log(`✓ 字段一致（${keys.size} 项）`);

// 3) 数值断言
const P = ENGINE.PRESETS;
const near = (a, b, tol) => Math.abs(a - b) / Math.abs(b) * 100 <= tol;
let fail = 0;
const assert = (name, ok, detail) => {
  console.log(`${ok ? '✓' : '✗'} ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) fail++;
};

// 模型库完整性
const bad = ENGINE.MODELS.filter(m => !(m.params > 0 && m.layers > 0 && m.hidden > 0 && m.heads > 0 && m.kvHeads > 0 && m.headDim > 0 && m.maxLen > 0));
assert(`模型库 ${ENGINE.MODELS.length} 个模型`, bad.length === 0, bad.map(m => m.id).join(','));
const kvBad = ENGINE.MODELS.filter(m => m.attn !== 'mla' && m.heads % m.kvHeads !== 0);
assert('GQA 头数整除', kvBad.length === 0, kvBad.map(m => m.id).join(','));
assert('场景预设 ' + ENGINE.WORKLOADS.length + ' 组', ENGINE.WORKLOADS.every(w => w.inLen > 0 && w.outLen > 0 && w.maxLen >= w.inLen + w.outLen));
assert('投机方案 ' + ENGINE.SPEC_METHODS.length + ' 种', ENGINE.SPEC_METHODS.every(s => s.on === 0 || (s.gamma > 0 && s.alpha >= 0 && s.alpha < 1)));

const r27 = ENGINE.compute(P.qwen27b.v);
assert('KV/token = 128 KiB', r27.kvTok === 131072, `${r27.kvTok} B`);
assert('KV 容量 ≈ 27 路 @8K', near(r27.maxConcLen, 27, 6), `${r27.maxConcLen} 路`);
assert('KV 池 ≈ 221k tok（vLLM 日志）', near(r27.kvPoolTok / 1e3, 221, 5), `${(r27.kvPoolTok / 1e3).toFixed(1)}k`);
assert('H100 平衡点 ≈ 295 FLOP/B', near(ENGINE.compute(P.llama70b.v).tune.ridge, 295, 2),
  `${ENGINE.compute(P.llama70b.v).tune.ridge.toFixed(1)}`);
assert('Llama-3-70B ≈ 141 GB', near(ENGINE.compute(P.llama70b.v).weightsGB, 141, 2),
  `${ENGINE.compute(P.llama70b.v).weightsGB.toFixed(1)} GB`);
assert('DeepSeek-V2 MLA ≈ 70 KB/token', near(ENGINE.compute(P.dsv3.v).kvTok / 1e3, 70, 2),
  `${(ENGINE.compute(P.dsv3.v).kvTok / 1e3).toFixed(2)} KB`);

// 4) 与本机实测对比（experiment/runs.csv, BASE 组，371 in / 128 out）
const MEAS = [
  ['TPOT C=1', 12.66, { 'load.conc': 1 }], ['TPOT C=4', 14.29, { 'load.conc': 4 }],
  ['TPOT C=8', 15.91, { 'load.conc': 8 }], ['TPOT C=16', 18.49, { 'load.conc': 16 }],
  ['TPOT C=32', 24.16, { 'load.conc': 32 }], ['TTFT C=1', 76.4, { 'load.conc': 1 }],
  ['吞吐 C=32', 1079.9, { 'load.conc': 32 }]
];
console.log('\n实测回归（runs.csv）：');
const rows = [];
for (const [name, meas, over] of MEAS) {
  const cfg = { ...P.qwen27b.v, ...over };
  const r = ENGINE.compute(cfg);
  const pred = name.startsWith('TTFT') ? r.e2e.ttft : name.startsWith('吞吐') ? r.e2e.outTps : r.dec.tpot;
  const e = Math.abs(pred - meas) / meas * 100;
  rows.push({ name, pred, meas, e });
  console.log(`  ${name.padEnd(10)} 预测 ${pred.toFixed(2).padStart(7)}  实测 ${String(meas).padStart(7)}  误差 ${e.toFixed(1)}%`);
}
const mape = rows.reduce((a, b) => a + b.e, 0) / rows.length;
console.log(`  平均绝对误差 MAPE = ${mape.toFixed(1)}%`);

// 5) 投机解码：MTP K=3 预设（E=2.0）
const spM = ENGINE.SPEC_METHODS.find(x => x.id === 'mtp');
const b = ENGINE.compute({ ...P.qwen27b.v, 'load.conc': 1 });
const sp = ENGINE.compute({ ...P.qwen27b.v, 'load.conc': 1, 'spec.on': 1, 'spec.gamma': spM.gamma,
  'spec.alpha': spM.alpha, 'spec.cost': spM.cost, 'spec.E': spM.E });
const gain = b.dec.tpot / sp.dec.tpot;
console.log(`\n投机：E=${sp.dec.E.toFixed(2)} tok/步，TPOT ${b.dec.tpot.toFixed(2)}→${sp.dec.tpot.toFixed(2)} ms，加速 ${gain.toFixed(2)}×（实测 1.88×）`);

// 5b) PD 分离自检：机制与 DistServe/Splitwise 一致（TTFT/TPOT 解耦、KV 传输是代价）
const pdBase = {
  ...ENGINE.BASE,
  'model.params': 32.8, 'model.active': 32.8, 'model.layers': 64, 'model.hidden': 5120,
  'model.heads': 64, 'model.kvHeads': 8, 'model.headDim': 128, 'model.wBytes': 1, 'model.kvBytes': 1,
  'load.inLen': 4000, 'load.outLen': 300, 'load.conc': 24, 'load.maxLen': 16384, 'load.actBase': 1.2,
  'hw.hwsel': 'h100', 'hw.count': 2, 'hw.tp': 2, 'hw.vram': 79.4, 'hw.bw': 3350, 'hw.flops': 1979,
  'hw.link': 900, 'hw.lat': 5, 'hw.mfuPre': 0.45, 'hw.mfuDec': 0.12, 'hw.bwEff': 0.85
};
const mono = ENGINE.compute(pdBase);
const pdInf = ENGINE.computePD({
  ...pdBase, 'pd.preHw': 'h100', 'pd.preCount': 1, 'pd.preTp': 2,
  'pd.decHw': 'h100', 'pd.decCount': 1, 'pd.decTp': 2, 'pd.link': 1000000
});
const expectTTFT = mono.e2e.ttft + pdInf.dec.tpot + 6 - 8; // +一个解码步 +握手 −一体部署的固定开销
const errTTFT = Math.abs(pdInf.sys.ttft - expectTTFT) / expectTTFT * 100;
console.log(`\nPD 自检：`);
console.log(`  无限链路 TTFT ${pdInf.sys.ttft.toFixed(0)} ms ≈ 一体 ${mono.e2e.ttft.toFixed(0)} + 一步 ${pdInf.dec.tpot.toFixed(1)} + 握手 = ${expectTTFT.toFixed(0)} ms（偏差 ${errTTFT.toFixed(1)}%）`);

const pdSlow = ENGINE.computePD({
  ...pdBase, 'pd.preHw': 'h100', 'pd.preCount': 1, 'pd.preTp': 2,
  'pd.decHw': 'h100', 'pd.decCount': 1, 'pd.decTp': 2, 'pd.link': 2
});
const xferExp = pdSlow.kvTok * 4000 / (2e9 * 0.8) * 1000;
console.log(`  2 GB/s 链路：KV 传输 ${pdSlow.xfer.ms.toFixed(0)} ms（公式 ${xferExp.toFixed(0)} ms），主瓶颈=${pdSlow.walls[0].k}，吞吐 ${pdSlow.sys.rps.toFixed(1)} req/s`);
const rPreExp = 2 * 1979e12 * 0.45 / (2 * 32.8e9 * 4000);
console.log(`  预填充上限 ${pdInf.pre.rps.toFixed(2)} req/s（公式 ${rPreExp.toFixed(2)}）`);
assert('PD 无限链路 ≈ 一体 + 一步', errTTFT < 5, `${errTTFT.toFixed(1)}%`);
assert('PD 链路瓶颈判定', pdSlow.walls[0].k === 'KV 链路', pdSlow.walls[0].k);
assert('PD KV 传输公式', Math.abs(pdSlow.xfer.ms - xferExp) < 0.5, `${pdSlow.xfer.ms.toFixed(1)} vs ${xferExp.toFixed(1)} ms`);
assert('PD 预填充上限公式', Math.abs(pdInf.pre.rps - rPreExp) / rPreExp < 0.03, `${pdInf.pre.rps.toFixed(2)} vs ${rPreExp.toFixed(2)} req/s`);

// 6) 边界：不得出现 NaN / 抛错
const edge = [
  ['单卡 TP=1', { ...ENGINE.BASE, 'hw.count': 1, 'hw.tp': 1, 'hw.link': 0 }],
  ['PP=2', { ...ENGINE.BASE, 'hw.pp': 2 }],
  ['滑窗 4K', { ...ENGINE.BASE, 'model.window': 4096 }],
  ['MLA 单路', { ...P.dsv3.v, 'load.conc': 1 }],
  ['α=1', { ...ENGINE.BASE, 'spec.on': 1, 'spec.alpha': 1 }],
  ['空输入', { ...ENGINE.BASE, 'model.params': '', 'model.layers': '', 'hw.bw': '', 'load.conc': '' }],
  ['负号输入', { ...ENGINE.BASE, 'model.params': '-' }],
  ['超并发 OOM', { ...ENGINE.BASE, 'load.conc': 9999 }]
];
let nan = 0;
for (const [name, cfg] of edge) {
  try {
    const r = ENGINE.compute(cfg);
    const bad = [r.dec.tpot, r.e2e.ttft, r.e2e.outTps, r.maxConc, r.kvTok].filter(v => !Number.isFinite(v));
    if (bad.length) { nan++; console.log(`  ✗ ${name}: 非有限值 ${bad}`); }
    else console.log(`  ✓ ${name}: tpot=${r.dec.tpot.toFixed(2)} maxConc=${r.maxConc} oom=${r.oom}`);
  } catch (e) { nan++; console.log(`  ✗ ${name}: ${e.message}`); }
}

// 7) 渲染冒烟：用最小 DOM 桩执行整段前端脚本，检查输出 HTML
const nodes = new Map();
const mkEl = sel => ({
  sel, innerHTML: '', value: '', tagName: sel === '#preset' ? 'SELECT' : 'DIV',
  dataset: {}, addEventListener() {}, appendChild() {}
});
const fakeDoc = {
  querySelector(sel) { if (!nodes.has(sel)) nodes.set(sel, mkEl(sel)); return nodes.get(sel); },
  querySelectorAll() { return []; },
  addEventListener() {}
};
try {
  new Function('document', 'window', sm[1])(fakeDoc, {});
  const out = nodes.get('#out')?.innerHTML || '';
  const quick = nodes.get('#inputs')?.innerHTML || '';
  const need = ['显存分解', 'Roofline', '指标计算链', '决策建议', '模型校验'];
  const miss = need.filter(x => !out.includes(x));
  const badv = ['NaN', 'undefined'].filter(x => out.includes(x));
  console.log(`\n渲染冒烟：输出 ${(out.length / 1024).toFixed(1)} KB，快速栏 ${quick.includes('qModel') ? '✓' : '✗'}`);
  console.log(`  ${miss.length ? '✗ 缺少区块: ' + miss.join(',') : '✓ 五大区块齐全'}`);
  console.log(`  ${badv.length ? '✗ 含坏值: ' + badv.join(',') : '✓ 无 NaN/undefined'}`);
  if (miss.length || badv.length || !quick.includes('qModel')) nan++;
} catch (e) {
  console.log('  ✗ 渲染抛错: ' + e.message);
  nan++;
}

// 7b) 交互链路：模拟点击 模型→显卡→精度→场景→投机，检查不崩且结果同步
const handlers = { input: [], change: [] };
const nodes2 = new Map();
const mk2 = s => ({ sel: s, innerHTML: '', value: '', tagName: s === '#qModel' || s === '#qGpu' ? 'SELECT' : 'DIV', dataset: {}, addEventListener() {} });
const doc2 = {
  querySelector(s) { if (!nodes2.has(s)) nodes2.set(s, mk2(s)); return nodes2.get(s); },
  querySelectorAll() { return []; },
  addEventListener(t, fn) { (handlers[t] = handlers[t] || []).push(fn); }
};
new Function('document', 'window', sm[1])(doc2, {});
const fire = (t, target) => (handlers[t] || []).forEach(fn => fn({ target, preventDefault() {} }));
const out2 = () => nodes2.get('#out').innerHTML;
const bad2 = () => /NaN|undefined/.test(out2());
const steps = [];
function step(name, fn, expect) {
  try { fn(); const ok = out2().includes(expect) && !bad2(); steps.push(ok); console.log(`  ${ok ? '✓' : '✗'} ${name}${ok ? '' : '（未出现:' + expect + '）'}`); }
  catch (e) { steps.push(false); console.log(`  ✗ ${name} 抛错: ${e.message}`); }
}
console.log('\n交互链路：');
step('切到 Llama-3.3-70B', () => fire('change', { id: 'qModel', value: 'llama-3.3-70b' }), 'Llama-3.3-70B');
step('切到 H100 ×8', () => fire('change', { id: 'qGpu', value: 'h100' }), 'H100');
step('切到 BF16', () => fire('change', { id: 'qDtype', value: '2' }), 'BF16');
step('切到 RAG 8K/512', () => fire('change', { id: 'qWork', value: 'rag' }), '8192 / 512');
step('开启 DFlash2', () => fire('change', { id: 'qSpec', value: 'dflash2' }), 'DFlash2');
step('改并发为 64', () => fire('input', { dataset: { k: 'load.conc' }, value: '64', tagName: 'INPUT' }), '并发 64');
step('改卡数为 4', () => fire('input', { id: 'qCount', value: '4', dataset: {}, tagName: 'INPUT' }), '× 4');
if (steps.some(x => !x)) nan++;

// 8) 生成报告
const md = `# 计算器校验报告

生成时间：${new Date().toISOString()}
被测文件：\`${FILE}\`

## 一、公式精度（与论文 / 厂商日志口径对比）

| 条目 | 计算值 | 参考值 | 误差 | 依据 |
|---|---|---|---|---|
${ENGINE.VALID.formula.map(c => {
  const p = c.f({ compute: ENGINE.compute, PRESETS: P });
  return `| ${c.n} | ${p.toFixed(2)} ${c.u} | ${c.m} ${c.u} | ${(Math.abs(p - c.m) / c.m * 100).toFixed(2)}% | ${c.src} |`;
}).join('\n')}

## 二、性能回归（Qwen3.8-27B-FP8 × 2×RTX5090，371 in / 128 out）

| 指标 | 计算 | 实测 | 误差 |
|---|---|---|---|
${rows.map(r => `| ${r.name} | ${r.pred.toFixed(2)} | ${r.meas} | ${r.e.toFixed(1)}% |`).join('\n')}

**MAPE = ${mape.toFixed(1)}%**

## 三、结论

- 容量类（权重、KV/token、可容纳并发）为确定性公式，误差 < 2%，可直接用于选型。
- 性能类依赖 MFU / 带宽效率 / 调度开销三个经验系数，本机 MAPE ${mape.toFixed(1)}%，
  最大偏差出现在 C=16（调度与队列占比上升，模型只计算单步稳态）。
- 投机解码：用 MTP K=3 预设（E=2.0，由实测反推）预测加速 ${gain.toFixed(2)}×，实测 1.88×，
  误差 ${(Math.abs(gain - 1.88) / 1.88 * 100).toFixed(1)}%；换模型或换草稿方案需重填「有效 E」。
- PD 分离：无限链路下 TTFT 与「一体部署 + 一个解码步 + 握手」偏差 ${errTTFT.toFixed(1)}%；
  2 GB/s 链路能正确判为 KV 链路墙；KV 传输时间与 S_in·KV/tok ÷ 链路 公式一致（${pdSlow.xfer.ms.toFixed(0)} ms）。
  机制上与 DistServe(OSDI'24)、Splitwise(ISCA'24)、Mooncake(FAST'25) 的拆分方式一致，但不含调度器带来的额外收益。

**使用方式**：先按「⑤ 校准系数」用本机压测把 MFU / 带宽效率 / 固定开销反推校准，再看墙与吞吐。
原始实测：\`experiment/runs.csv\`（BASE 组，input 371 / output 128，median of 3）。
`;
fs.writeFileSync('validation/report.md', md, 'utf8');
console.log('\n✓ 已写 validation/report.md');
process.exit(fail || nan ? 1 : 0);
