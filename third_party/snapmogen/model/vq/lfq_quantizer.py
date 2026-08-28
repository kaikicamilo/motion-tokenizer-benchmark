"""LFQ adapter, wrapping `LFQ` from vector_quantize_pytorch.

The entropy auxiliary loss is returned in the commit-loss slot, so
training.lambda_commit re-weights a term already scaled by
entropy_loss_weight. Negative values are expected (per-sample minus batch
entropy), not a bug.
"""

import torch
import torch.nn as nn

try:
    from vector_quantize_pytorch import LFQ
except Exception as e:  # pragma: no cover
    raise ImportError(
        "lfq_quantizer requires vector-quantize-pytorch. "
        "Install with: pip install vector-quantize-pytorch"
    ) from e


class LFQMotionQuantizer(nn.Module):
    """LFQ with SnapMoGen's quantizer call interface.

    Parameters
    ----------
    code_dim : int
        Encoder latent dimension (512). LFQ projects code_dim ->
        log2(codebook_size) bits and back internally.
    codebook_size : int
        Implicit vocabulary size; must be a power of 2 (2048 = 2^11).
    entropy_loss_weight : float
        Entropy regularization weight (library default 0.1).
    diversity_gamma : float
        Diversity weight inside the entropy loss (library default 1.0).
    inv_temperature : float
        Inverse temperature in the LFQ forward (library default 100);
        controls the softness of the entropy computation.
    """

    def __init__(self, code_dim: int, codebook_size: int = 2048,
                 entropy_loss_weight: float = 0.1, diversity_gamma: float = 1.0,
                 inv_temperature: float = 100.0,
                 experimental_softplus_entropy_loss: bool = False,
                 entropy_loss_offset: float = 5.0,
                 frac_per_sample_entropy: float = 1.0):
        super().__init__()
        assert codebook_size & (codebook_size - 1) == 0, \
            f"codebook_size must be a power of 2, got {codebook_size}"
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        self.inv_temperature = inv_temperature

        # frac_per_sample_entropy<1 subsamples tokens in the per-sample entropy;
        # memory there scales with n_tokens * codebook_size.
        self.lfq = LFQ(
            codebook_size=codebook_size,
            dim=code_dim,
            entropy_loss_weight=entropy_loss_weight,
            diversity_gamma=diversity_gamma,
            experimental_softplus_entropy_loss=experimental_softplus_entropy_loss,
            entropy_loss_offset=entropy_loss_offset,
            frac_per_sample_entropy=frac_per_sample_entropy,
        )

    def forward(self, x, temperature=None, m_lens=None,
                start_drop=None, quantize_dropout_prob=None):
        """x: (B, C, T). Returns (x_q, loss, perplexity).

        `loss` is LFQ's entropy_aux_loss (non-zero; may be negative).
        temperature/m_lens/dropout are accepted for signature compatibility;
        padding is masked outside the quantizer, as for VQ/FSQ.
        """
        x_btc = x.transpose(1, 2)
        quantized, indices, entropy_aux_loss = self.lfq(
            x_btc, inv_temperature=self.inv_temperature)
        x_q = quantized.transpose(1, 2)

        loss = entropy_aux_loss
        perplexity = self._perplexity(indices)
        return x_q, loss, perplexity

    @torch.no_grad()
    def _perplexity(self, indices):
        """exp(entropy) of the batch index distribution (codebook usage)."""
        idx = indices.reshape(-1)
        counts = torch.bincount(idx, minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp(min=1)
        nz = probs > 0
        entropy = -(probs[nz] * probs[nz].log()).sum()
        return entropy.exp()

    # ---- names expected by HRVQVAE.encode/forward_decoder ----

    def quantize_all(self, x, m_lens=None, return_latent=False):
        """x (B, C, T) -> (code_idx (B, T), all_codes (B, C, T))."""
        x_btc = x.transpose(1, 2)
        quantized, indices, _ = self.lfq(x_btc, inv_temperature=self.inv_temperature)
        codes = quantized.transpose(1, 2)
        if return_latent:
            return indices, codes
        return indices

    def get_codes_from_indices(self, indices):
        codes = self.lfq.indices_to_codes(indices)
        return codes.transpose(1, 2)
