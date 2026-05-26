'use strict';
/**
 * Run Python simplification on a trajectory (points as JSON).
 */
const { execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.join(__dirname, '..', '..');

const PYTHON = (() => {
  const candidates = [
    path.join(ROOT, 'venv', 'bin', 'python3'),
    path.join(ROOT, 'venv', 'bin', 'python'),
    'python3',
    'python',
  ];
  return candidates.find(p => { try { fs.accessSync(p); return true; } catch { return false; } }) || 'python3';
})();

/**
 * @param {Array<{lat:number,lon:number,alt?:number,timestamp?:string}>} points
 * @param {string} algorithm
 * @param {number} compressionRatio
 * @returns {{ simplified: Array<{lat:number,lon:number}>, budget: number }}
 */
function simplifyPoints(points, algorithm, compressionRatio) {
  if (!points || points.length < 2) {
    return { simplified: points || [], budget: points?.length || 0 };
  }

  if (algorithm === 'original' || algorithm === 'none') {
    return {
      simplified: points.map(p => ({ lat: p.lat, lon: p.lon })),
      budget: points.length,
    };
  }

  const payload = JSON.stringify({ points, algorithm, compression_ratio: compressionRatio });

  const script = `
import json, sys
sys.path.insert(0, '.')
data = json.loads(sys.stdin.read())
pts = data['points']
algo = data['algorithm']
cr = float(data['compression_ratio'])
import pandas as pd
df = pd.DataFrame(pts)
if 'timestamp' not in df.columns:
    df['timestamp'] = None
budget = max(2, int(len(df) / cr))
if algo == 'proposed':
    from src.algorithms.proposed_method import proposed_simplification
    simp, idx = proposed_simplification(df, budget)
    out = [{'lat': float(p[0]), 'lon': float(p[1])} for p in simp]
else:
    from src.algorithms.baseline_algorithms import simplify_with_budget
    simp = simplify_with_budget(df, algorithm=algo, budget=budget)
    out = [{'lat': float(p[0]), 'lon': float(p[1])} for p in simp]
print(json.dumps({'simplified': out, 'budget': budget}))
`;

  try {
    const out = execFileSync(PYTHON, ['-c', script], {
      cwd: ROOT,
      input: payload,
      timeout: 120000,
      maxBuffer: 50 * 1024 * 1024,
    }).toString();
    return JSON.parse(out);
  } catch (err) {
    throw new Error(`Simplification failed: ${err.message}`);
  }
}

module.exports = { simplifyPoints };
