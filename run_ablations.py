"""Matched CIFAR-100 mechanism study; no held-out-driven setting selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'csr_k10': ('centroid_soft', 10.),
    'normalized_centroid': ('centroid_normalized', 10.),
    'shuffled_pairs': ('centroid_shuffled', 10.),
    'csr_k1': ('centroid_soft', 1.),
    'csr_k100': ('centroid_soft', 100.),
}

def summary():
    rows = []
    for p in sorted((ROOT/'results').glob('*/*/result.json')):
        x = json.loads(p.read_text()); c = x['config']
        row = dict(stage=c['experiment_stage'], variant=c['config_variant'], seed=c['seed'])
        for kind, key in [('linear', 'linear_probe'), ('knn', 'knn')]:
            for split in ['val', 'test']:
                v = x[key][split+'_accuracy']
                row[split+'_'+kind] = None if v is None else 100*v
        row['collapsed'] = x['geometry']['collapsed'] or x['geometry']['unit_projector']['collapsed']
        rows.append(row)
        print(row)
    if rows:
        with (ROOT/'results/ablation_summary.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    else:
        print('No completed runs yet.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'smoke', 'run', 'summary'])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--data-root', type=Path, default=Path('./data'))
    p.add_argument('--cpu', action='store_true', help='CPU smoke check only')
    args = p.parse_args(); os.chdir(ROOT)
    if args.stage == 'summary': summary(); return
    if args.cpu and args.stage != 'smoke': p.error('--cpu is only for smoke checks')
    smoke = args.stage == 'smoke'
    stage = 'smoke' if smoke else 'cifar100_mechanism'
    base = json.loads((ROOT/'base_config.json').read_text())
    base['root'] = str(args.data_root.resolve())
    for seed in args.seeds if args.seeds is not None else ([0] if smoke else [0,1,2]):
        for variant in args.variants:
            method, kappa = VARIANTS[variant]
            cfg = base | dict(method=method, kappa=kappa, seed=seed, experiment_stage=stage,
                config_variant=variant, output=f'results/{stage}/{variant}_s{seed}')
            if smoke:
                cfg.update(epochs=1, train_size=400, val_size=200, batch_size=32, workers=0,
                    diagnostics_size=64, probe_epochs=2, include_test=False,
                    device='cpu' if args.cpu else 'cuda', amp=not args.cpu)
            path = ROOT/'configs'/stage/f'{variant}_s{seed}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and json.loads(path.read_text()) != cfg:
                raise ValueError(f'Existing configuration differs: {path}')
            path.write_text(json.dumps(cfg, indent=2)+'\n')
            if args.stage == 'prepare':
                print(path); continue
            out = ROOT/cfg['output']; out.mkdir(parents=True, exist_ok=True)
            print(f'Starting {variant}, seed {seed}; log: {out / "process.log"}', flush=True)
            with (out/'process.log').open('a') as log:
                r = subprocess.run([sys.executable, '-u', 'train_ssl.py', '--config', str(path)],
                    stdout=log, stderr=subprocess.STDOUT)
            if r.returncode:
                print((out/'process.log').read_text()[-6000:]); raise SystemExit(r.returncode)
    if args.stage != 'prepare': summary()

if __name__ == '__main__': main()
