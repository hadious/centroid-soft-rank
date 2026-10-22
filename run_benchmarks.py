"""Run matched ImageNet-subset pilots/finals without altering previous studies."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS={"csr":dict(method="centroid_soft",kappa=10.),"mmcr":dict(method="mmcr"),"vicreg":dict(method="vicreg")}
DATA=Path("./data")


def preflight(root, datasets):
    from datasets.folder_benchmarks import FolderBenchmark
    from datasets.vision import build_datasets
    records=[]
    for ds in datasets:
        train=FolderBenchmark(root,ds,True);heldout=FolderBenchmark(root,ds,False)
        expected_classes=200 if ds=="tinyimagenet" else 10
        assert len(train.classes)==expected_classes
        assert train.class_to_idx==heldout.class_to_idx
        if ds=="tinyimagenet":
            assert len(train)==100000 and len(heldout)==10000
            assert np.all(np.bincount(train.targets)==500)
            assert np.all(np.bincount(heldout.targets)==50)
        for collection in [train,heldout]:
            missing=[p for p,_ in collection.samples if not Path(p).is_file()]
            if missing:raise FileNotFoundError(missing[:3])
            for index in [0,len(collection)//2,len(collection)-1]:collection[index]
        bundle=build_datasets(ds,root=root,train_size=None,val_size=5000 if ds=="tinyimagenet" else 1000,include_test=False)
        assert bundle["metadata"]["train_validation_disjoint"] and bundle["test_eval"] is None
        a,b=bundle["ssl_train"][0];x,_=bundle["val_eval"][0]
        size=64 if ds=="tinyimagenet" else 128
        assert tuple(a.shape)==tuple(b.shape)==tuple(x.shape)==(3,size,size)
        record=dict(dataset=ds,classes=expected_classes,official_train=len(train),tuning=len(bundle["val_eval"]),
                    ssl_train=len(bundle["ssl_train"]),heldout_official_val=len(heldout),input_size=size,
                    train_manifest=train.sample_manifest_sha256,heldout_manifest=heldout.sample_manifest_sha256)
        records.append(record);print(json.dumps(record),flush=True)
    return records


def config(args,ds,method,seed):
    smoke=args.stage=="smoke"
    return dict(dataset=ds,root=str(args.data_root.resolve()),method=METHODS[method]["method"],kappa=10.,
        seed=seed,split_seed=1729,train_size=(400 if ds=="tinyimagenet" else 256) if smoke else None,
        val_size=(400 if ds=="tinyimagenet" else 100) if smoke else (5000 if ds=="tinyimagenet" else 1000),
        include_test=args.stage=="final",epochs={"smoke":1,"pilot":10,"final":100}[args.stage],
        batch_size=64 if smoke else 256,workers=0 if smoke else 4,threads=4,device="cuda",amp=True,
        lr=.001,weight_decay=0.,warmup_epochs=1,grad_clip=None,loss_device="cpu",cuda_memory_fraction=.6,
        diagnostics_size=64 if smoke else 1024,diagnostics_every=1 if smoke else 5,
        probe_epochs=2 if smoke else 120,probe_device="cpu",save_features=True,
        output=f"results/{args.stage}/{ds}/{method}_s{seed}",
        experiment_stage=args.stage,config_variant=method,
        evaluation_note="test_* means official validation held out from tuning; pilots never evaluate it",
        comparison_note="Shared learning rate; no gradient clipping for any method; not separately tuned published baselines")


def summarize():
    rows=[]
    for p in sorted((ROOT/"results").glob("*/*/*/result.json")):
        r=json.loads(p.read_text());cfg=r["config"];g=r["geometry"]
        rows.append(dict(stage=cfg["experiment_stage"],dataset=cfg["dataset"],method=cfg["config_variant"],seed=cfg["seed"],
            epochs=cfg["epochs"],tuning_linear=100*r["linear_probe"]["val_accuracy"],tuning_knn=100*r["knn"]["val_accuracy"],
            heldout_linear=None if r["linear_probe"]["test_accuracy"] is None else 100*r["linear_probe"]["test_accuracy"],
            heldout_knn=None if r["knn"]["test_accuracy"] is None else 100*r["knn"]["test_accuracy"],
            collapsed=g["collapsed"] or g["unit_projector"]["collapsed"],training_seconds=r["training_seconds"],
            peak_allocated_mib=r["maximum_cuda_memory_mb"]))
    if not rows:print("No completed runs yet.");return
    (ROOT/"results/benchmark_summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (ROOT/"results/benchmark_summary.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("stage dataset method seed tuning_linear tuning_knn heldout_linear collapsed")
    for r in rows:
        print(f"{r['stage']:5} {r['dataset']:12} {r['method']:6} {r['seed']} {r['tuning_linear']:7.2f} {r['tuning_knn']:7.2f} {str(r['heldout_linear']):>8} {r['collapsed']}")
    print("Saved results/benchmark_summary.csv and benchmark_summary.json. Smoke scores are plumbing checks only.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["check","smoke","pilot","final","summary"])
    parser.add_argument("--datasets",nargs="+",choices=["tinyimagenet","imagenette"],default=["tinyimagenet","imagenette"])
    parser.add_argument("--methods",nargs="+",choices=list(METHODS),default=list(METHODS))
    parser.add_argument("--seeds",nargs="+",type=int,default=None)
    parser.add_argument("--data-root",type=Path,default=DATA)
    args=parser.parse_args();os.chdir(ROOT)
    if args.stage=="summary":summarize();return
    preflight(args.data_root,args.datasets)
    if args.stage=="check":return
    import torch
    if not torch.cuda.is_available():raise RuntimeError("CUDA is not available; refusing accidental CPU training.")
    seeds=args.seeds if args.seeds is not None else ([0,1,2] if args.stage=="final" else [0])
    for ds in args.datasets:
        for seed in seeds:
            for method in args.methods:
                cfg=config(args,ds,method,seed)
                p=ROOT/"configs"/args.stage/ds/f"{method}_s{seed}.json"
                p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists() and json.loads(p.read_text())!=cfg:
                    raise ValueError(f"Existing configuration differs: {p}; preserve old results and use a separate study directory")
                p.write_text(json.dumps(cfg,indent=2)+"\n")
                out=ROOT/cfg["output"];out.mkdir(parents=True,exist_ok=True)
                print(f"Starting {args.stage}: {ds} / {method} / seed {seed}; log: {out/'process.log'}",flush=True)
                with (out/"process.log").open("a") as log:
                    completed=subprocess.run([sys.executable,"-u","train_ssl.py","--config",str(p)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if completed.returncode:
                    print((out/"process.log").read_text()[-6000:],file=sys.stderr)
                    raise SystemExit(completed.returncode)
                summarize()


if __name__=="__main__":main()
