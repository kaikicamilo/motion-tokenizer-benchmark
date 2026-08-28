"""Train a motion tokenizer. Both the dataset and the quantizer are set by --config.

    python scripts/train.py --config configs/snapmogen/rvq_independent.yaml
    python scripts/train.py --config configs/humanml3d/rvq_independent.yaml
"""
import os, sys, shutil, argparse, time, datetime, csv
from os.path import join as pjoin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import ROOT, bootstrap, build_model
bootstrap()

import numpy as np
import torch
from torch.utils.data import DataLoader
from config.load_config import load_config
from trainers.ae_trainer import VQTokenizerTrainer
from utils.fixseeds import fixseed
from utils.utils import plot_3d_motion


def setup_snapmogen(cfg, device):
    from dataset.dataset import TextMotionDataset
    from model.evaluator.evaluator_wrapper import EvaluatorWrapper
    from utils import bvh_io
    from common.skeleton import Skeleton
    from utils.motion_process_bvh import recover_pos_from_rot
    from utils.paramUtil import kinematic_chain

    root = cfg.data.root_dir
    cfg.data.feat_dir = pjoin(root, 'renamed_feats')
    split, captions = pjoin(root, 'data_split_info'), pjoin(root, 'all_caption_clean.json')
    mean = np.load(pjoin(root, 'meta_data', 'mean.npy'))
    std = np.load(pjoin(root, 'meta_data', 'std.npy'))
    make = lambda s: TextMotionDataset(cfg, mean, std, pjoin(split, f'{s}_fnames.txt'),
                                       pjoin(split, f'{s}_ids.txt'), captions)
    train_ds, val_ds = make('train'), make('val')

    eval_cfg = load_config(pjoin(cfg.evaluator, 'evaluator.yaml'))
    eval_cfg.exp.root_ckpt_dir, eval_cfg.data.name = './checkpoint_dir', 'snapmogen'  # weights next to the yaml
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)
    eval_loader = DataLoader(val_ds, batch_size=eval_cfg.matching_pool_size, drop_last=True,
                             num_workers=8, shuffle=True, pin_memory=True)

    anim = bvh_io.load(pjoin(root, 'renamed_bvhs', 'm_ep2_00086.bvh'))
    skeleton = Skeleton(anim.offsets, anim.parents, device=device)

    def fk(data):
        return recover_pos_from_rot(train_ds.inv_transform(data), joints_num=cfg.data.joint_num, skeleton=skeleton)

    def plot(data, save_dir):
        pos = fk(data).detach().cpu().numpy()
        for i in range(len(pos)):
            plot_3d_motion(pjoin(save_dir, '%02d.mp4' % i), kinematic_chain, pos[i], title="None", fps=30, radius=100)

    return train_ds, val_ds, eval_loader, eval_wrapper, plot, fk


def setup_humanml3d(cfg, device):
    from dataset.humanml3d_dataset import Text2MotionDataset
    from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
    from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
    from utils.get_opt import get_opt
    from utils.motion_process_hml import recover_from_ric
    from utils.paramUtil import t2m_kinematic_chain

    os.environ.setdefault('HUMANML3D_ROOT', cfg.data.root_dir)
    opt = get_opt(cfg.evaluator, device)
    eval_wrapper = EvaluatorModelWrapper(opt)
    # normalization statistics from the evaluator, so evaluation and training agree
    mean, std = np.load(pjoin(opt.meta_dir, 'mean.npy')), np.load(pjoin(opt.meta_dir, 'std.npy'))
    train_ds = Text2MotionDataset(opt, mean, std, pjoin(cfg.data.root_dir, 'train.txt'))
    val_ds = Text2MotionDataset(opt, mean, std, pjoin(cfg.data.root_dir, 'val.txt'))
    eval_loader, _ = get_dataset_motion_loader(cfg.evaluator, 32, 'test', device=device)

    def plot(data, save_dir):
        data = train_ds.inv_transform(data.cpu().detach().numpy())
        for i in range(len(data)):
            joint = recover_from_ric(torch.from_numpy(data[i]).float(), 22).numpy()
            plot_3d_motion(pjoin(save_dir, '%02d.mp4' % i), t2m_kinematic_chain, joint, title="None", fps=20, radius=4)

    return train_ds, val_ds, eval_loader, eval_wrapper, plot, None


SETUP = {'snapmogen': setup_snapmogen, 'humanml3d': setup_humanml3d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='configs/<dataset>/<method>.yaml')
    ap.add_argument('--continue_train', action='store_true', help='resume from latest.tar')
    args = ap.parse_args()

    cfg = load_config(os.path.abspath(args.config))
    cfg.exp.is_continue = cfg.exp.is_continue or args.continue_train
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'vq', cfg.exp.name)
    if cfg.exp.is_continue:
        saved = load_config(pjoin(cfg.exp.checkpoint_dir, os.path.basename(args.config)))
        saved.exp.is_continue, saved.exp.device, saved.exp.checkpoint_dir = True, cfg.exp.device, cfg.exp.checkpoint_dir
        cfg = saved
    else:
        os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
        shutil.copy(args.config, cfg.exp.checkpoint_dir)

    fixseed(cfg.exp.seed)
    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)
    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, 'animation')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'vq', cfg.exp.name)
    for d in (cfg.exp.model_dir, cfg.exp.eval_dir, cfg.exp.log_dir):
        os.makedirs(d, exist_ok=True)

    if os.environ.get('WANDB_DISABLED', '0') != '1':  # optional; mirrors TensorBoard scalars
        import wandb
        kw = dict(project=os.environ.get('WANDB_PROJECT', 'motion-tokenizer-benchmark'),
                  name=cfg.exp.name, group=cfg.data.name, config=dict(cfg), sync_tensorboard=True)
        if os.environ.get('WANDB_ENTITY'):
            kw['entity'] = os.environ['WANDB_ENTITY']
        try:
            wandb.init(mode=os.environ.get('WANDB_MODE', 'offline'), **kw)
        except Exception as e:
            print(f"[wandb] disabled ({e})")

    train_ds, val_ds, eval_loader, eval_wrapper, plot, fk = SETUP[cfg.data.name](cfg, device)
    loader = lambda ds: DataLoader(ds, batch_size=cfg.training.batch_size, drop_last=True,
                                   num_workers=8, shuffle=True, pin_memory=True)

    net = build_model(cfg)
    print(f"{cfg.data.name} / {cfg.exp.name}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params on {device}")
    trainer = VQTokenizerTrainer(cfg, vq_model=net, device=device)

    t0, start = time.time(), datetime.datetime.now().isoformat(timespec='seconds')
    status = 'completed'
    try:
        trainer.train(loader(train_ds), loader(val_ds), eval_loader, eval_wrapper, plot, fk)
    except BaseException as e:
        status = f'interrupted:{type(e).__name__}'
        raise
    finally:
        path = pjoin(ROOT, f'training_times_{cfg.data.name}.csv')
        row = dict(exp_name=cfg.exp.name, config=os.path.basename(args.config), start=start,
                   end=datetime.datetime.now().isoformat(timespec='seconds'),
                   duration_h=round((time.time() - t0) / 3600, 3), max_epoch=cfg.training.max_epoch, status=status)
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row)


if __name__ == '__main__':
    main()
