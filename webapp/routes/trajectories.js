/**
 * /api/trajectories  — GPS point data for the map
 *
 * GET /api/trajectories/list?user_id=000    → trajectory metadata
 * GET /api/trajectories/points?trajectory_id=0&algorithm=proposed&compression_ratio=5
 */

const express       = require('express');
const { execFileSync } = require('child_process');
const path          = require('path');
const router        = express.Router();

const ROOT       = path.join(__dirname, '..', '..');
const PICKLE     = path.join(ROOT, 'data', 'processed', 'trajectories.pkl');
const PROPERTIES = path.join(ROOT, 'data', 'processed', 'trajectory_properties.csv');

const PYTHON = (() => {
  const candidates = [
    path.join(ROOT, 'venv', 'bin', 'python3'),
    path.join(ROOT, 'venv', 'bin', 'python'),
    'python3',
    'python',
  ];
  const fs = require('fs');
  return candidates.find(p => { try { fs.accessSync(p); return true; } catch { return false; } })
    || 'python3';
})();

function runPython(script, args = []) {
  return execFileSync(PYTHON, ['-c', script, ...args], { cwd: ROOT, timeout: 60000 }).toString();
}

// ── list trajectories ─────────────────────────────────────────────────────────
router.get('/list', (req, res) => {
  const { user_id, max = 200 } = req.query;
  const userFilter = user_id ? `str(t.get('user_id','?')).zfill(3) == '${String(user_id).padStart(3,'0')}'` : 'True';

  const script = `
import pickle, json, sys
sys.path.insert(0,'.')
with open(r'${PICKLE}','rb') as f:
    trajs=pickle.load(f)
out=[]
for i,t in enumerate(trajs):
    if not hasattr(t,'columns'): continue
    uid=str(t['user_id'].iloc[0]).zfill(3) if 'user_id' in t.columns else '?'
    fid=str(t['file_id'].iloc[0]) if 'file_id' in t.columns else '?'
    if not (${userFilter}): continue
    out.append({'trajectory_id':i,'user_id':uid,'file_id':fid,'num_points':len(t),
        'mean_lat':round(float(t['lat'].mean()),6),'mean_lon':round(float(t['lon'].mean()),6)})
    if len(out)>=${max}: break
print(json.dumps(out))
`;
  try {
    const result = JSON.parse(runPython(script));
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── GPS points (original + simplified) ───────────────────────────────────────
router.get('/points', (req, res) => {
  const { trajectory_id = 0, algorithm = 'proposed', compression_ratio = 5.0 } = req.query;
  const tid = parseInt(trajectory_id);
  const cr  = parseFloat(compression_ratio);

  const script = `
import pickle, json, sys, math
sys.path.insert(0,'.')
with open(r'${PICKLE}','rb') as f:
    trajs=pickle.load(f)
t=trajs[${tid}]
original=[{'lat':float(r['lat']),'lon':float(r['lon'])} for _,r in t.iterrows()]

# simplify
budget=max(2,int(len(t)/${cr}))
algo='${algorithm}'
try:
    if algo=='proposed':
        from src.algorithms.proposed_method import proposed_simplification
        simp,idx=proposed_simplification(t,budget)
        simplified=[{'lat':float(p[0]),'lon':float(p[1]),'idx':int(i)} for p,i in zip(simp,idx)]
    else:
        from src.algorithms.baseline_algorithms import simplify_with_budget
        simp=simplify_with_budget(t,algo,budget)
        simplified=[{'lat':float(p[0]),'lon':float(p[1])} for p in simp]
except Exception as e:
    simplified=[]

# metrics
try:
    import numpy as np
    orig_pts=t[['lat','lon']].values
    simp_pts=simp if isinstance(simp[0],(list,)) else simp
    def hausdorff(a,b):
        from src.algorithms.baseline_algorithms import haversine_distance
        d=max(min(haversine_distance(tuple(p),tuple(q)) for q in b) for p in a)
        return round(d,2)
    h=hausdorff(orig_pts[:50],simp if hasattr(simp,'__len__') else [])
except:
    h=None

uid=str(t['user_id'].iloc[0]) if 'user_id' in t.columns else '?'
fid=str(t['file_id'].iloc[0]) if 'file_id' in t.columns else '?'
print(json.dumps({'original':original,'simplified':simplified,
    'meta':{'trajectory_id':${tid},'user_id':uid,'file_id':fid,
            'n_original':len(original),'n_simplified':len(simplified),
            'compression_ratio':${cr},'budget':budget}}))
`;

  try {
    const result = JSON.parse(runPython(script));
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

module.exports = router;
