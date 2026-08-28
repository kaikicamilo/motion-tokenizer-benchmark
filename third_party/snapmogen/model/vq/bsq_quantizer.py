"""BSQ adapter, following the official BinarySphericalQuantizer (nvidia/QLIP).

With this conv+attention backbone, `l2norm_input=True` is required to avoid
early collapse; the official module relies only on the output 1/sqrt(d) scale.
"""

import torch
import torch.nn as nn
from einops import rearrange, reduce

_EPS = 1e-8


class BSQMotionQuantizer(nn.Module):
    """BSQ with SnapMoGen's quantizer call interface.

    Parameters
    ----------
    code_dim : int
        Encoder latent dimension (512). Linear projections
        code_dim <-> embed_dim implement the 'project' boxes of BSQ Fig. 1.
    embed_dim : int
        Number of BSQ bits; vocabulary = 2^embed_dim (11 -> 2048, matching
        VQ's nb_code). Must be divisible by group_size.
    group_size : int
        Group size for the tractable entropy computation (official code).
    beta : float
        Commit-loss weight. Default 0.0 (paper's best config).
    gamma_0, gamma_1 : float
        Entropy weights: gamma_0*E[H(q)] - gamma_1*H[E[q]]. Defaults 1.0.
    inv_temperature : float
        Inverse temperature (official: 100.0).
    l2norm_input : bool
        Normalize the input onto the unit sphere before sign(); see module
        docstring.
    """

    def __init__(self, code_dim: int, embed_dim: int = 11, group_size: int = 11,
                 beta: float = 0.0, gamma_0: float = 1.0, gamma_1: float = 1.0,
                 inv_temperature: float = 100.0, l2norm_input: bool = False):
        super().__init__()
        assert embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
        self.code_dim = code_dim
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.beta = beta
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.inv_temperature = inv_temperature
        self.l2norm_input = l2norm_input
        self.codebook_size = 2 ** embed_dim

        # 'project' boxes of BSQ Fig. 1: code_dim <-> embed_dim (bits)
        self.proj_in = nn.Linear(code_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, code_dim)

        # basis for code<->index conversion (official code)
        self.register_buffer("basis", 2 ** torch.arange(embed_dim - 1, -1, -1), persistent=False)
        self.register_buffer("group_basis", 2 ** torch.arange(group_size - 1, -1, -1), persistent=False)
        group_codes = torch.arange(2 ** group_size)
        group_codebook = self._indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer("group_codebook", group_codebook, persistent=False)

    # ---------- BSQ core (faithful to the official code) ----------
    def _quantize(self, z):
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        return z + (zhat - z).detach()  # STE

    def _indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self.basis), 2)
        return codes_non_centered * 2 - 1

    def _codes_to_indexes(self, zhat):
        return ((zhat.int() + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def _soft_entropy_loss(self, z):
        # group-wise approximation (official code)
        group_codebook = self.group_codebook / (self.embed_dim ** 0.5)
        divided_z = rearrange(z, "... (g c) -> ... g c", c=self.group_size)
        distance = -2 * torch.einsum("... g c, d c -> ... g d", divided_z, group_codebook)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        persample_entropy = torch.special.entr(prob + _EPS).sum((-1, -2)).mean()
        avg_prob = reduce(prob, "... g d -> g d", "mean")
        cb_entropy = torch.special.entr(avg_prob + _EPS).sum()
        return persample_entropy, cb_entropy

    # ---------- SnapMoGen pipeline interface ----------
    def forward(self, x, temperature=None, m_lens=None,
                start_drop=None, quantize_dropout_prob=None):
        """x: (B, C, T). Returns (x_q, loss, perplexity).

        loss = commit(beta) + entropy_penalty / inv_temperature (official).
        """
        x_btc = x.transpose(1, 2)                    # (B, T, code_dim)
        v = self.proj_in(x_btc)                      # (B, T, embed_dim)
        if self.l2norm_input:
            v = v / (v.norm(dim=-1, keepdim=True) + _EPS)

        zq = self._quantize(v)                       # STE, {-1,+1} with grad
        indices = self._codes_to_indexes(zq.detach())

        persample_entropy, cb_entropy = self._soft_entropy_loss(v)
        entropy_penalty = self.gamma_0 * persample_entropy - self.gamma_1 * cb_entropy

        q_scale = 1.0 / (self.embed_dim ** 0.5)      # spherical output scale
        zq = zq * q_scale
        commit_loss = self.beta * torch.mean(((zq.detach() - v) ** 2).sum(dim=-1))

        z_out = self.proj_out(zq)                    # (B, T, code_dim)
        x_q = z_out.transpose(1, 2)                  # (B, C, T)

        loss = commit_loss + entropy_penalty / self.inv_temperature
        perplexity = self._perplexity(indices)
        return x_q, loss, perplexity

    @torch.no_grad()
    def _perplexity(self, indices):
        # .int() on NaN/Inf activations can yield out-of-range indices.
        idx = indices.reshape(-1).long()
        idx = idx[(idx >= 0) & (idx < self.codebook_size)]
        if idx.numel() == 0:
            return torch.ones((), device=indices.device)
        counts = torch.bincount(idx, minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp(min=1)
        nz = probs > 0
        return (-(probs[nz] * probs[nz].log()).sum()).exp()

    def quantize_all(self, x, m_lens=None, return_latent=False):
        """x (B, C, T) -> (code_idx (B, T), all_codes (B, C, T))."""
        x_btc = x.transpose(1, 2)
        v = self.proj_in(x_btc)
        if self.l2norm_input:
            v = v / (v.norm(dim=-1, keepdim=True) + _EPS)
        zq = self._quantize(v)
        indices = self._codes_to_indexes(zq.detach())
        zq = zq * (1.0 / (self.embed_dim ** 0.5))
        codes = self.proj_out(zq).transpose(1, 2)
        if return_latent:
            return indices, codes
        return indices

    def get_codes_from_indices(self, indices):
        """indices (B, T) -> latent (B, C, T)."""
        zq = self._indexes_to_codes(indices).float()   # (B, T, embed_dim) in {-1,+1}
        zq = zq * (1.0 / (self.embed_dim ** 0.5))
        return self.proj_out(zq).transpose(1, 2)
