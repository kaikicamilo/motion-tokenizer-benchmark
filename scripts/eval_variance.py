"""Mean/std of FID and R-Precision over repeated evaluation rounds (one dataset).

Every round resamples captions, temporal crops and retrieval pools, so the metrics
fluctuate; all methods (and the ground truth) share each round, which also allows
paired comparisons. Per-round values are written to JSON.

    python scripts/eval_variance.py --dataset humanml3d --rounds 20 --out variance_humanml3d.json
"""
import os, sys, json, argparse
from os.path import join as pjoin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import bootstrap, config_path, load_trained, METHODS
bootstrap()

import numpy as np, torch
from torch.utils.data import DataLoader
from config.load_config import load_config
from utils.metrics import calculate_R_precision, calculate_activation_statistics, calculate_frechet_distance

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GT = 'Ground truth'


def fid(a, b):
    return calculate_frechet_distance(*calculate_activation_statistics(a), *calculate_activation_statistics(b))


def rounds_humanml3d(base, nets, n_rounds):
    from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
    from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
    from utils.get_opt import get_opt
    os.environ.setdefault('HUMANML3D_ROOT', base.data.root_dir)
    ew = EvaluatorModelWrapper(get_opt(base.evaluator, DEV))
    loader, _ = get_dataset_motion_loader(base.evaluator, 32, 'test', device=DEV)
    for _ in range(n_rounds):
        emb = {k: [] for k in [GT] + list(nets)}; rp = {k: np.zeros(3) for k in emb}; N = 0
        with torch.no_grad():
            for we, po, cap, sl, motion, ml, tok in loader:
                motion, ml = motion.to(DEV).float(), ml.to(DEV).long()
                et, em = ew.get_co_embeddings(we, po, sl, motion, ml)
                et_np = et.cpu().numpy()
                emb[GT].append(em.cpu().numpy()); rp[GT] += calculate_R_precision(et_np, em.cpu().numpy(), 3, True)
                for k, (net, nf) in nets.items():
                    _, codes = net.encode(motion[..., :nf], ml.clone())
                    _, ep = ew.get_co_embeddings(we, po, sl, net.decode(codes, ml.clone()), ml)
                    emb[k].append(ep.cpu().numpy()); rp[k] += calculate_R_precision(et_np, ep.cpu().numpy(), 3, True)
                N += motion.shape[0]
        yield emb, rp, N


def rounds_snapmogen(base, nets, n_rounds):
    from dataset.dataset import TextMotionDataset
    from model.evaluator.evaluator_wrapper import EvaluatorWrapper
    root = base.data.root_dir
    base.data.feat_dir = pjoin(root, 'renamed_feats')
    split = pjoin(root, 'data_split_info')
    mean, std = np.load(pjoin(root, 'meta_data', 'mean.npy')), np.load(pjoin(root, 'meta_data', 'std.npy'))
    ds = TextMotionDataset(base, mean, std, pjoin(split, 'val_fnames.txt'), pjoin(split, 'val_ids.txt'),
                           pjoin(root, 'all_caption_clean.json'))
    ecfg = load_config(pjoin(base.evaluator, 'evaluator.yaml'))
    ecfg.exp.root_ckpt_dir, ecfg.data.name = './checkpoint_dir', 'snapmogen'
    ew = EvaluatorWrapper(ecfg, device=DEV)
    loader = DataLoader(ds, batch_size=ecfg.matching_pool_size, drop_last=True, num_workers=4, shuffle=True)
    for _ in range(n_rounds):
        emb = {k: [] for k in [GT] + list(nets)}; rp = {k: np.zeros(3) for k in emb}; N = 0
        with torch.no_grad():
            for texts, motions, mlens in loader:
                motions, ml = motions.to(DEV).float(), mlens.to(DEV).long()
                et, _ = ew.encode_text(texts, sample_mean=True); et_np = et.cpu().numpy()
                f_gt, em, _ = ew.encode_motion(motions[..., :148], ml, sample_mean=True)
                emb[GT].append(f_gt.cpu().numpy())
                rp[GT] += calculate_R_precision(et_np, em.cpu().numpy(), 3, True, is_cosine_sim=True)
                for k, (net, nf) in nets.items():
                    _, codes = net.encode(motions[..., :nf], ml.clone())
                    f_p, ep, _ = ew.encode_motion(net.decode(codes, ml.clone())[..., :148], ml, sample_mean=True)
                    emb[k].append(f_p.cpu().numpy())
                    rp[k] += calculate_R_precision(et_np, ep.cpu().numpy(), 3, True, is_cosine_sim=True)
                N += motions.shape[0]
        yield emb, rp, N


ROUNDS = {'humanml3d': rounds_humanml3d, 'snapmogen': rounds_snapmogen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(ROUNDS))
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    base = load_config(config_path(a.dataset, 'base'))
    nets = {}
    for stem, name in METHODS:
        try:
            net, cfg = load_trained(a.dataset, stem, DEV); nets[name] = (net, cfg.data.dim_pose)
        except FileNotFoundError:
            print(f"  skip {name}: no checkpoint")
    out = {k: {'fid': [], 'top1': [], 'top3': []} for k in [GT] + list(nets)}
    for r, (emb, rp, N) in enumerate(ROUNDS[a.dataset](base, nets, a.rounds), 1):
        gt = np.concatenate(emb[GT])
        for k in out:
            out[k]['top1'].append(float(rp[k][0] / N)); out[k]['top3'].append(float(rp[k][2] / N))
            if k != GT:
                out[k]['fid'].append(float(fid(gt, np.concatenate(emb[k]))))
        json.dump(out, open(a.out, 'w'), indent=1)
        print(f"  round {r}/{a.rounds}", flush=True)
    for k, v in out.items():
        f = f"  FID {np.mean(v['fid']):.4f} ± {np.std(v['fid']):.4f}" if v['fid'] else ''
        print(f"{k:<18} Top-1 {np.mean(v['top1']):.4f} ± {np.std(v['top1']):.4f}   "
              f"Top-3 {np.mean(v['top3']):.4f} ± {np.std(v['top3']):.4f}{f}")


if __name__ == '__main__':
    main()
