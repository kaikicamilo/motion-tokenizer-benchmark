"""Collect the table metrics of one dataset from its training logs.

Takes, per method, the evaluation with the lowest FID (the reported checkpoint)
and the validation-pass perplexity logged at that epoch; model size is measured by instantiating
the config. R-Precision here is a single noisy pass; the paper reports the mean
of eval_rprecision_std.py, and MPJPE comes from eval_mpjpe_humanml3d.py.

    python scripts/collect_metrics.py --dataset snapmogen --logs logs/snapmogen --out metrics_snapmogen.csv
    python scripts/collect_metrics.py --dataset humanml3d --logs logs/humanml3d --out metrics_humanml3d.csv --mpjpe mpjpe_humanml3d.csv
"""
import os, re, csv, sys, argparse
from os.path import join as pjoin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import bootstrap, build_model, config_path, METHODS, TOKENS_PER_STEP
bootstrap()
from config.load_config import load_config

EVAL = re.compile(r"Eva. Ep (\d+):, FID\. ([\d.]+), Diversity Real\. ([\d.]+), Diversity\. ([\d.]+), "
                  r"R_precision_real\. \([\d., ]+\), R_precision\. \(([\d.]+), ([\d.]+), ([\d.]+)\)"
                  r"(?:.*mpjpe\. ([\d.]+))?")
PPL = re.compile(r"^Validation epoch:\s*(\d+)\D.*perplexity: ([\d.]+)")


def best_eval(log):
    best = None
    for line in open(log, errors='ignore'):
        m = EVAL.search(line)
        if m and (best is None or float(m.group(2)) < best['FID']):
            best = dict(epoch=int(m.group(1)), FID=float(m.group(2)), Div_real=float(m.group(3)),
                        Div=float(m.group(4)), T1=float(m.group(5)), T3=float(m.group(7)),
                        MPJPE=float(m.group(8)) if m.group(8) else None)
    if best:
        ppl = [float(m.group(2)) for l in open(log, errors='ignore')
               if (m := PPL.search(l)) and int(m.group(1)) == best['epoch']]
        best['perplexity'] = sum(ppl) / len(ppl) if ppl else float('nan')
    return best


def size_M(cfg):
    net = build_model(cfg)
    return (sum(p.numel() for p in net.parameters()) + sum(b.numel() for b in net.buffers())) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['snapmogen', 'humanml3d'])
    ap.add_argument('--logs', required=True, help='directory with <method>.log files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mpjpe', help='CSV (method,MPJPE_cm) from eval_mpjpe_humanml3d.py; '
                                    'SnapMoGen logs already contain MPJPE')
    a = ap.parse_args()
    mpjpe = {r['method']: float(r['MPJPE_cm']) for r in csv.DictReader(open(a.mpjpe))} if a.mpjpe else {}

    rows = []
    for stem, name in METHODS:
        log = pjoin(a.logs, f'{stem}.log')
        b = best_eval(log) if os.path.exists(log) else None
        if b is None:
            print(f"  skip {name}: no evaluation entries in {log}"); continue
        cfg = load_config(config_path(a.dataset, stem))
        rows.append(dict(method=name, tokens_step=TOKENS_PER_STEP[stem], size_M=round(size_M(cfg), 2),
                         best_epoch=b['epoch'], FID=round(b['FID'], 4), R_Prec_Top1=round(b['T1'], 4),
                         R_Prec_Top3=round(b['T3'], 4),
                         MPJPE_cm=round(mpjpe.get(name, b['MPJPE'] or float('nan')), 2),
                         Diversity=round(b['Div'], 2),
                         Div_real=round(b['Div_real'], 2), perplexity=round(b['perplexity'], 1)))
        print(f"  {name:<18} ep {b['epoch']:>4}  FID {b['FID']:.4f}")
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} methods)")


if __name__ == '__main__':
    main()
