"""Shared helpers: path bootstrap, method registry, model construction.

(Named `benchmark` rather than `common` to avoid shadowing SnapMoGen's `common` package.)"""
import os, sys
from os.path import join as pjoin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = pjoin(ROOT, 'third_party', 'snapmogen')


def bootstrap():
    """Put the vendored SnapMoGen code on sys.path and run from the repo root,
    so ./data, ./checkpoint_dir, ./log and ./glove resolve consistently."""
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# (config stem, name used in the paper)
METHODS = [('vqvae', 'VQ-VAE'), ('hrvqvae', 'MS-RVQ [2,1]'), ('hrvqvae_8421', 'MS-RVQ [8,4,2,1]'),
           ('rvq4', 'Shared RVQ x4'), ('rvq_independent4', 'RVQ x4'), ('rvq', 'Shared RVQ x6'),
           ('rvq_independent', 'RVQ x6'), ('fsq', 'FSQ'), ('bsq', 'BSQ'), ('residual_fsq', 'Res-FSQ'),
           ('lfq', 'LFQ')]
TOKENS_PER_STEP = {'vqvae': 1, 'fsq': 1, 'bsq': 1, 'lfq': 1, 'hrvqvae': 1.5, 'hrvqvae_8421': 1.875,
                   'rvq4': 4, 'rvq_independent4': 4, 'rvq': 6, 'rvq_independent': 6, 'residual_fsq': 6}


def config_path(dataset, stem):
    return pjoin(ROOT, 'configs', dataset, f'{stem}.yaml')


def build_model(cfg):
    from model.vq.rvq_model import HRVQVAE
    return HRVQVAE(cfg, cfg.data.dim_pose, cfg.model.down_t, cfg.model.stride_t, cfg.model.width,
                   cfg.model.depth, cfg.model.dilation_growth_rate, cfg.model.vq_act,
                   cfg.model.use_attn, cfg.model.vq_norm)


def checkpoint_path(cfg, which='net_best_fid'):
    return pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.exp.name, 'model', f'{which}.tar')


def load_trained(dataset, stem, device):
    """Instantiate a method from its config and load the reported checkpoint."""
    import torch
    from config.load_config import load_config
    cfg = load_config(config_path(dataset, stem))
    net = build_model(cfg).to(device)
    sd = torch.load(checkpoint_path(cfg), map_location=device, weights_only=False)
    net.load_state_dict(sd.get('vq_model', sd.get('model')))
    net.eval()
    return net, cfg
