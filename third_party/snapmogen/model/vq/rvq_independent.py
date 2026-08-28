"""RVQ with independent EMA codebooks, one per residual stage.

Each stage is an HRQuantizeEMAResetV2 with scales=[1], so the only difference
from the shared-codebook variant is the number of codebooks.
"""

import torch
import torch.nn as nn
from einops import rearrange

from model.vq.quantizer import HRQuantizeEMAResetV2, length_to_mask


class RVQIndependent(nn.Module):
    """Residual VQ with N independent EMA codebooks."""

    def __init__(self, nb_code, code_dim, mu, num_stages, **kwargs):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.num_stages = num_stages
        self.stages = nn.ModuleList([
            HRQuantizeEMAResetV2(nb_code=nb_code, code_dim=code_dim, mu=mu,
                                 scales=[1], share_quant_resi=4, quant_resi=0)
            for _ in range(num_stages)
        ])

    def forward(self, x, temperature=0., m_lens=None, start_drop=1, quantize_dropout_prob=0.):
        N, width, T = x.shape
        residual = x.clone()
        f_hat = torch.zeros_like(x)
        total_loss = 0.
        perplexities = []
        if m_lens is not None:
            full_mask = length_to_mask(m_lens, T)
        for stage in self.stages:
            if m_lens is not None:
                residual = residual * full_mask.unsqueeze(1)
            x_d, loss_i, perp_i = stage(residual, temperature=temperature, m_lens=m_lens,
                                        start_drop=start_drop, quantize_dropout_prob=quantize_dropout_prob)
            f_hat = f_hat + x_d
            residual = x - f_hat
            total_loss = total_loss + loss_i
            perplexities.append(perp_i)
        mean_loss = total_loss / self.num_stages
        mean_perplexity = sum(perplexities) / len(perplexities)
        return f_hat, mean_loss, mean_perplexity

    def quantize_all(self, x, m_lens=None, return_latent=False):
        N, width, T = x.shape
        residual = x.clone()
        f_hat = torch.zeros_like(x)
        code_idx_list = []
        if m_lens is not None:
            full_mask = length_to_mask(m_lens, T)
        for stage in self.stages:
            if m_lens is not None:
                residual = residual * full_mask.unsqueeze(1)
            idx, codes = stage.quantize_all(residual, m_lens=m_lens, return_latent=True)
            code_idx_list.append(idx)
            f_hat = f_hat + codes
            residual = x - f_hat
        if return_latent:
            return code_idx_list, f_hat
        return code_idx_list

    def get_codes_from_indices(self, indices_list):
        f_hat = None
        for stage, indices in zip(self.stages, indices_list):
            codes = stage.get_codes_from_indices(indices)
            f_hat = codes if f_hat is None else f_hat + codes
        return f_hat
