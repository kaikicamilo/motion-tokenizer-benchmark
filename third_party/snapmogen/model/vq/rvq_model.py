import random
import torch
import torch.nn as nn
from model.cnn_networks import EncoderAttn, DecoderAttn
from model.vq.quantizer import HRQuantizeEMAReset, HRQuantizeEMAResetV2
from model.vq.fsq_quantizer import FSQMotionQuantizer
from model.vq.residual_fsq_quantizer import ResidualFSQMotionQuantizer
from model.vq.lfq_quantizer import LFQMotionQuantizer
from model.vq.bsq_quantizer import BSQMotionQuantizer
from model.vq.rvq_independent import RVQIndependent

def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


class HRVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn=False,
                 norm=None):

        super().__init__()
        output_emb_width = args.quantizer.code_dim
        # self.quant = args.quantizer
        # self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
        #                        dilation_growth_rate, activation=activation, norm=norm)
        # self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
        #                        dilation_growth_rate, activation=activation, norm=norm)
        self.encoder = EncoderAttn(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, use_attn=use_attn)
        self.decoder = DecoderAttn(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, use_attn=use_attn)
        self.cfg= args
        if getattr(self.cfg.quantizer, 'type', None) == 'fsq':
            # FSQ — no codebook/EMA/commit. `levels` defines the implicit
            # vocabulary (product). Masking and commit_loss=0 handled outside
            # the quantizer, as for VQ.
            self.quantizer = FSQMotionQuantizer(code_dim=args.quantizer.code_dim,
                                                levels=args.quantizer.levels)
        elif getattr(self.cfg.quantizer, 'type', None) == 'residual_fsq':
            # Residual FSQ — codebook-free analogue of RVQ (N residual stages).
            self.quantizer = ResidualFSQMotionQuantizer(code_dim=args.quantizer.code_dim,
                                                        levels=args.quantizer.levels,
                                                        num_quantizers=args.quantizer.num_quantizers)
        elif getattr(self.cfg.quantizer, 'type', None) == 'lfq':
            # LFQ — implicit binary codebook + entropy loss. The returned loss
            # is the entropy_aux_loss (non-zero, may be negative); it enters
            # the total loss through the trainer's lambda_commit.
            self.quantizer = LFQMotionQuantizer(code_dim=args.quantizer.code_dim,
                                                codebook_size=args.quantizer.codebook_size,
                                                entropy_loss_weight=args.quantizer.entropy_loss_weight,
                                                diversity_gamma=args.quantizer.diversity_gamma,
                                                inv_temperature=args.quantizer.inv_temperature,
                                                experimental_softplus_entropy_loss=getattr(args.quantizer, 'softplus_entropy_loss', False),
                                                entropy_loss_offset=getattr(args.quantizer, 'entropy_loss_offset', 5.0),
                                                frac_per_sample_entropy=getattr(args.quantizer, 'frac_per_sample_entropy', 1.0))
        elif getattr(self.cfg.quantizer, 'type', None) == 'bsq':
            # BSQ — codebook-free with spherical norm (q_scale=1/sqrt(embed_dim))
            # that bounds the quantization error -> stable where LFQ collapses.
            # Loss = commit(beta) + entropy_penalty / inv_temperature.
            self.quantizer = BSQMotionQuantizer(code_dim=args.quantizer.code_dim,
                                                embed_dim=args.quantizer.embed_dim,
                                                group_size=args.quantizer.group_size,
                                                beta=args.quantizer.beta,
                                                gamma_0=args.quantizer.gamma_0,
                                                gamma_1=args.quantizer.gamma_1,
                                                inv_temperature=args.quantizer.inv_temperature,
                                                l2norm_input=args.quantizer.l2norm_input)
        elif getattr(self.cfg.quantizer, 'type', None) == 'rvq_independent':
            # Canonical RVQ: N independent EMA codebooks (one per stage), vs the
            # shared variant (HRQuantizeEMAResetV2 scales=[1xN], one codebook reused).
            self.quantizer = RVQIndependent(nb_code=args.quantizer.nb_code,
                                            code_dim=args.quantizer.code_dim,
                                            mu=args.quantizer.mu,
                                            num_stages=args.quantizer.num_stages)
        elif 'version' in self.cfg.quantizer and self.cfg.quantizer.version == 'v2':
            self.quantizer = HRQuantizeEMAResetV2(nb_code=args.quantizer.nb_code,
                                                code_dim=args.quantizer.code_dim,
                                                mu=args.quantizer.mu,
                                                scales=args.quantizer.scales,
                                                share_quant_resi=args.quantizer.share_quant_resi,
                                                quant_resi=args.quantizer.quant_resi)
        else:
            self.quantizer = HRQuantizeEMAReset(nb_code=args.quantizer.nb_code, 
                                                code_dim=args.quantizer.code_dim, 
                                                mu=args.quantizer.mu, 
                                                scales=args.quantizer.scales)
        self.down_t = down_t

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x, m_lens=None):
        # N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in, m_lens)

        # if m_lens is not None:

        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize_all(x_encoder, m_lens, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x, m_lengths=None):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in, m_lengths)

        if m_lengths is not None:
            m_lengths //= 2**self.down_t
        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, commit_loss, perplexity = self.quantizer(x_encoder, temperature=0.5, 
                                                              m_lens=m_lengths,
                                                              start_drop=self.cfg.quantizer.start_drop,
                                                              quantize_dropout_prob=self.cfg.quantizer.quantize_dropout_prob)

        if m_lengths is not None:
            x_quantized = x_quantized.permute(0, 2, 1)
            # m_lengths //= 2**self.down_t
            mask = length_to_mask(m_lengths, x_quantized.shape[1])
            x_quantized[~mask] = 0
            x_quantized = x_quantized.permute(0, 2, 1)
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized, m_lengths)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x, m_lengths=None):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        if len(x_d.shape) == 4:
            x = x_d.sum(dim=0)

        if m_lengths is not None:
            # x = x.permute(0, 2, 1)
            m_lengths //= 2**self.down_t
            mask = length_to_mask(m_lengths, x_d.shape[1])
            x_d[~mask] = 0
        x_d = x_d.permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x_d, m_lengths)
        # x_out = self.postprocess(x_decoder)
        return x_out
    
    def decode(self, x, m_lengths=None):
        # x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        if m_lengths is not None:
            x = x.permute(0, 2, 1)
            m_lengths //= 2**self.down_t
            mask = length_to_mask(m_lengths, x.shape[1], x.device)
            x[~mask] = 0
            x = x.permute(0, 2, 1)
        # x = torch.zeros_like(x)
        # x = x.permute(0, 2, 1)
        # decoder
        x_out = self.decoder(x, m_lengths)
        # x_out = self.postprocess(x_decoder)
        return x_out