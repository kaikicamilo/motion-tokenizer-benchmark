"""FSQ adapter, wrapping `FSQ` from vector_quantize_pytorch.

No learned codebook and no auxiliary loss, so forward returns aux_loss = 0.
Padding is masked outside the quantizer, identically for all methods.
"""

import torch
import torch.nn as nn

try:
    from vector_quantize_pytorch import FSQ
except Exception as e:  # pragma: no cover
    raise ImportError(
        "fsq_quantizer requires vector-quantize-pytorch. "
        "Install with: pip install vector-quantize-pytorch"
    ) from e


class FSQMotionQuantizer(nn.Module):
    """FSQ with SnapMoGen's quantizer call interface.

    Parameters
    ----------
    code_dim : int
        Encoder latent dimension (512). FSQ internally projects
        code_dim -> len(levels) -> code_dim.
    levels : list[int]
        FSQ levels per dimension; prod(levels) is the implicit vocabulary.
        [8,8,8,4] -> 2048, matching VQ's nb_code.
    """

    def __init__(self, code_dim: int, levels):
        super().__init__()
        self.code_dim = code_dim
        self.levels = list(levels)
        self.codebook_size = 1
        for l in self.levels:
            self.codebook_size *= l

        self.fsq = FSQ(levels=self.levels, dim=code_dim)

    def forward(self, x, temperature=None, m_lens=None,
                start_drop=None, quantize_dropout_prob=None):
        """x: (B, C, T). Returns (x_q, loss, perplexity).

        temperature/m_lens/dropout are accepted for signature compatibility;
        plain FSQ does not use them (padding is masked outside the quantizer,
        as for VQ, keeping mask handling identical across methods).
        """
        # The library expects (B, T, C); the pipeline provides (B, C, T).
        x_btc = x.transpose(1, 2)
        xhat, indices = self.fsq(x_btc)
        x_q = xhat.transpose(1, 2)

        loss = torch.zeros((), device=x.device, dtype=x.dtype)  # no auxiliary loss in FSQ
        perplexity = self._perplexity(indices)
        return x_q, loss, perplexity

    @torch.no_grad()
    def _perplexity(self, indices):
        """exp(entropy) of the batch index distribution — codebook-usage metric.

        Per-batch estimate; aggregate indices over the dataset for global usage.
        """
        idx = indices.reshape(-1)
        counts = torch.bincount(idx, minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp(min=1)
        nz = probs > 0
        entropy = -(probs[nz] * probs[nz].log()).sum()
        return entropy.exp()

    # ---- token-level encode/decode ----

    def encode_indices(self, x):
        """x (B, C, T) -> indices (B, T)."""
        x_btc = x.transpose(1, 2)
        _, indices = self.fsq(x_btc)
        return indices

    def decode_indices(self, indices):
        """indices (B, T) -> x_q (B, C, T)."""
        codes = self.fsq.indices_to_codes(indices)
        return codes.transpose(1, 2)

    # ---- names expected by HRVQVAE.encode/forward_decoder ----

    def quantize_all(self, x, m_lens=None, return_latent=False):
        """Single-stage analogue of the VQ quantize_all. x (B, C, T)."""
        idx = self.encode_indices(x)
        codes = self.decode_indices(idx)
        if return_latent:
            return idx, codes
        return idx

    def get_codes_from_indices(self, indices):
        return self.decode_indices(indices)
