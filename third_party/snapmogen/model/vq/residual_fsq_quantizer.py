"""Residual FSQ adapter, wrapping `ResidualFSQ` from vector_quantize_pytorch.

Codebook-free analogue of RVQ. Indices carry a stage dimension: (B, T, N).
"""

import torch
import torch.nn as nn

try:
    from vector_quantize_pytorch import ResidualFSQ
except Exception as e:  # pragma: no cover
    raise ImportError(
        "residual_fsq_quantizer requires vector-quantize-pytorch. "
        "Install with: pip install vector-quantize-pytorch"
    ) from e


class ResidualFSQMotionQuantizer(nn.Module):
    """Residual FSQ with SnapMoGen's quantizer call interface.

    Parameters
    ----------
    code_dim : int
        Encoder latent dimension (512). ResidualFSQ projects
        code_dim -> len(levels) -> code_dim internally.
    levels : list[int]
        FSQ levels per stage; prod(levels) is the per-stage vocabulary
        ([8,8,8,4] -> 2048, matching RVQ's nb_code).
    num_quantizers : int
        Number of residual stages (matches the RVQ stage count).
    """

    def __init__(self, code_dim: int, levels, num_quantizers: int):
        super().__init__()
        self.code_dim = code_dim
        self.levels = list(levels)
        self.num_quantizers = num_quantizers

        self.vocab_per_stage = 1
        for l in self.levels:
            self.vocab_per_stage *= l

        self.rfsq = ResidualFSQ(
            dim=code_dim,
            levels=self.levels,
            num_quantizers=num_quantizers,
        )

    def forward(self, x, temperature=None, m_lens=None,
                start_drop=None, quantize_dropout_prob=None):
        """x: (B, C, T). Returns (x_q, loss, perplexity).

        temperature/m_lens/dropout are accepted for signature compatibility;
        padding is masked outside the quantizer, as for VQ/FSQ.
        """
        x_btc = x.transpose(1, 2)                    # (B, T, C)
        quantized, indices = self.rfsq(x_btc)        # (B, T, C), (B, T, N)
        x_q = quantized.transpose(1, 2)              # (B, C, T)

        loss = torch.zeros((), device=x.device, dtype=x.dtype)  # no auxiliary loss
        perplexity = self._perplexity(indices)
        return x_q, loss, perplexity

    @torch.no_grad()
    def _perplexity(self, indices):
        """exp(entropy) over ALL stage indices pooled together (per batch)."""
        idx = indices.reshape(-1)
        counts = torch.bincount(idx, minlength=self.vocab_per_stage).float()
        probs = counts / counts.sum().clamp(min=1)
        nz = probs > 0
        entropy = -(probs[nz] * probs[nz].log()).sum()
        return entropy.exp()

    # ---- names expected by HRVQVAE.encode/forward_decoder ----

    def quantize_all(self, x, m_lens=None, return_latent=False):
        """x (B, C, T) -> (code_idx (B, T, N), all_codes (B, C, T))."""
        x_btc = x.transpose(1, 2)
        quantized, indices = self.rfsq(x_btc)
        codes = quantized.transpose(1, 2)
        if return_latent:
            return indices, codes
        return indices

    def get_codes_from_indices(self, indices):
        """indices (B, T, N) -> latent (B, C, T)."""
        codes = self.rfsq.get_output_from_indices(indices)
        return codes.transpose(1, 2)
