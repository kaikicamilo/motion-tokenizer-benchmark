"""MPJPE (cm) on HumanML3D, computed from each method's reported checkpoint.

Global (non root-relative) error over recovered 22-joint positions, masked by
sequence length. HumanML3D only: the SnapMoGen trainer already logs MPJPE.

    python scripts/eval_mpjpe_humanml3d.py --out mpjpe_humanml3d.csv
"""
import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import bootstrap, config_path, load_trained, METHODS
bootstrap()

import numpy as np, torch
from os.path import join as pjoin
from config.load_config import load_config
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from utils.get_opt import get_opt
from utils.motion_process_hml import recover_from_ric
from utils.metrics import calculate_mpjpe

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(stems, out):
    rows = []
    base = load_config(config_path('humanml3d', 'base'))
    os.environ.setdefault('HUMANML3D_ROOT', base.data.root_dir)
    opt = get_opt(base.evaluator, DEV)
    mean = torch.from_numpy(np.load(pjoin(opt.meta_dir, 'mean.npy'))).float().to(DEV)
    std = torch.from_numpy(np.load(pjoin(opt.meta_dir, 'std.npy'))).float().to(DEV)
    loader, _ = get_dataset_motion_loader(base.evaluator, 32, 'test', device=DEV)

    print(f"{'method':<18}{'epoch':>6}{'MPJPE (cm)':>12}")
    for stem, name in METHODS:
        if stems and stem not in stems:
            continue
        try:
            net, cfg = load_trained('humanml3d', stem, DEV)
        except FileNotFoundError:
            print(f"{name:<18}{'—':>6}{'(no checkpoint)':>16}"); continue
        nf, err, cnt = cfg.data.dim_pose, 0.0, 0
        with torch.no_grad():
            for _, _, _, _, motion, m_len, _ in loader:
                motion, m_len = motion.to(DEV).float(), m_len.to(DEV).long()
                _, codes = net.encode(motion[..., :nf], m_len.clone())
                rec = net.decode(codes, m_len.clone())
                gt_j = recover_from_ric(motion[..., :nf] * std + mean, 22)
                rc_j = recover_from_ric(rec[..., :nf] * std + mean, 22)
                mask = torch.arange(motion.shape[1], device=DEV)[None] < m_len[:, None]
                e, c = calculate_mpjpe(rc_j, gt_j, mask=mask, only_local=False)
                err += float(e); cnt += int(c)
        ep = torch.load(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.exp.name, 'model', 'net_best_fid.tar'),
                        map_location='cpu', weights_only=False).get('ep', '?')
        print(f"{name:<18}{ep:>6}{err / cnt * 100:>12.2f}")
        rows.append(dict(method=name, MPJPE_cm=round(err / cnt * 100, 2)))
    if out:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['method', 'MPJPE_cm']); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('methods', nargs='*', help='config stems; default: all')
    ap.add_argument('--out', help='write method,MPJPE_cm CSV for collect_metrics.py --mpjpe')
    a = ap.parse_args()
    main(a.methods, a.out)
