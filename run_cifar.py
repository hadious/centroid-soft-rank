"""Run the recorded 30-epoch CIFAR comparisons sequentially."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=['cifar10','cifar100'],default=['cifar10','cifar100'])
    p.add_argument('--methods',nargs='+',choices=['csr_k10','mmcr','vicreg'],default=['csr_k10','mmcr','vicreg'])
    p.add_argument('--seeds',nargs='+',type=int,choices=[0,1,2],default=[0,1,2])
    p.add_argument('--data-root',type=Path,default=Path('./data'))
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();data_root=args.data_root.resolve();os.chdir(ROOT)
    for ds in args.datasets:
        for seed in args.seeds:
            for method in args.methods:
                template=ROOT/'configs'/f'{ds}_final'/f'{method}_s{seed}.json'
                cfg=json.loads(template.read_text());cfg['root']=str(data_root)
                path=ROOT/'generated_configs'/f'{ds}_final'/template.name
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists() and json.loads(path.read_text())!=cfg:
                    raise ValueError(f'Configuration changed: {path}')
                path.write_text(json.dumps(cfg,indent=2)+'\n')
                if args.prepare_only:print(path);continue
                out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
                print(f'Starting {ds}/{method}/seed {seed}; log: {out / "process.log"}',flush=True)
                with (out/'process.log').open('a') as log:
                    proc=subprocess.run([sys.executable,'-u','train_ssl.py','--config',str(path)],stdout=log,stderr=subprocess.STDOUT)
                if proc.returncode:
                    print((out/'process.log').read_text()[-6000:]);raise SystemExit(proc.returncode)

if __name__=='__main__':main()
