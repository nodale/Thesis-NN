from __future__ import annotations
 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
 
from mamba_ssm import Mamba
from mamba_ssm import Mamba2
from mamba_ssm import Mamba3
from mamba_ssm.ops.triton.layer_norm import RMSNorm
 
 
# BLOCKS
 
def make_norm(d_model, norm_type="rms"):
    if norm_type == "rms":
        return RMSNorm(d_model)

    if norm_type == "layer":
        return nn.LayerNorm(d_model)

    if norm_type == "none":
        return nn.Identity()

    raise ValueError(
        f"norm_type must be rms/layer/none, got {norm_type}"
    )

class Projection(nn.Module):
    """
    Linear projection + normalization
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        norm="rms",
        activation=False,
    ):
        super().__init__()

        layers = [
            nn.Linear(in_dim, out_dim),
            make_norm(out_dim, norm),
        ]

        if activation:
            layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # saved with model, not trainable
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

 
class DropPath(nn.Module):
    """Stochastic depth: drops the residual branch with prob p (per sample)."""
 
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.size(0), 1, 1).bernoulli_(keep).div_(keep)
        return x * mask
 
 
class GatedMLP(nn.Module):
    """SwiGLU feed-forward — the modern default channel mixer."""
 
    def __init__(self, d_model: int, mult: float = 4.0):
        super().__init__()
        d_hidden = int(d_model * mult * 2 / 3)  # param-match vs vanilla 4x MLP
        self.w_in   = nn.Linear(d_model, d_hidden)
        self.w_gate = nn.Linear(d_model, d_hidden)
        self.w_out  = nn.Linear(d_hidden, d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_in(x)) * self.w_gate(x))
 
 
# =============================================================================
# Mamba blocks
# =============================================================================
 
class SimpleMambaBlock(nn.Module):
    """Minimal block: pre-norm + Mamba + residual.
 
    Use for shallow stacks, fast prototyping, or when you want fewest moving parts.
    """
 
    def __init__(self, d_model: int, norm: str = "rms", mamba_impl=Mamba, **mamba_kwargs):
        super().__init__()
        self.norm  = make_norm(d_model, norm)
        self.mixer = mamba_impl(d_model=d_model, **mamba_kwargs)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))
 
 
class AdvancedMambaBlock(nn.Module):
    """Two-sublayer block (transformer-style):
        x = x + DropPath( gamma1 * Mamba(  norm1(x) ) )    # time   mixing
        x = x + DropPath( gamma2 * SwiGLU( norm2(x) ) )    # channel mixing
 
    Recommended default for deeper stacks and best quality.
    """
 
    def __init__(
        self,
        d_model: int,
        norm: str = "rms",
        mlp_mult: float = 4.0,
        layer_scale: float = 1e-4,
        drop_path: float = 0.0,
        mamba_impl=Mamba, 
        **mamba_kwargs,
    ):
        super().__init__()
        # Mamba sublayer
        self.norm1  = make_norm(d_model, norm)
        self.mixer  = mamba_impl(d_model=d_model, **mamba_kwargs)
        self.drop1  = DropPath(drop_path)
        self.gamma1 = (
            nn.Parameter(layer_scale * torch.ones(d_model))
            if layer_scale > 0 else None
        )
 
        # MLP sublayer
        self.norm2  = make_norm(d_model, norm)
        self.mlp    = GatedMLP(d_model, mult=mlp_mult)
        self.drop2  = DropPath(drop_path)
        self.gamma2 = (
            nn.Parameter(layer_scale * torch.ones(d_model))
            if layer_scale > 0 else None
        )
 
    @staticmethod
    def _scale(h: torch.Tensor, gamma) -> torch.Tensor:
        return h if gamma is None else h * gamma
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mixer(self.norm1(x))
        x = x + self.drop1(self._scale(h, self.gamma1))
        h = self.mlp(self.norm2(x))
        x = x + self.drop2(self._scale(h, self.gamma2))
        return x
 
 
_BLOCKS = {
    "simple":   SimpleMambaBlock,
    "advanced": AdvancedMambaBlock,
}

_MAMBA_IMPLS = {
    "mamba": Mamba,
    "mamba2": Mamba2,
    "mamba3": Mamba3,
}
 
 
# =============================================================================
# Time adapters: [B, T_in, D] -> [B, T_out, D]
# =============================================================================
 
class LinearTimeAdapter(nn.Module):
    """Dense linear over time. Channel-independent. O(T_in * T_out) params."""
 
    def __init__(self, d_model: int, input_len: int, output_len: int):
        super().__init__()
        self.proj = nn.Linear(input_len, output_len)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.transpose(1, 2)).transpose(1, 2)
 
 
class PoolTimeAdapter(nn.Module):
    """Adaptive avg/max pooling. Parameter-free, lossy, cheap."""
 
    def __init__(self, d_model: int, input_len: int, output_len: int, mode: str = "avg"):
        super().__init__()
        self.output_len = output_len
        self.fn = F.adaptive_avg_pool1d if mode == "avg" else F.adaptive_max_pool1d
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x.transpose(1, 2), self.output_len).transpose(1, 2)
 
 
class ConvTimeAdapter(nn.Module):
    """Strided Conv1d + adaptive pool. Local mixing, O(T) cost."""
 
    def __init__(self, d_model: int, input_len: int, output_len: int, kernel_size: int = 5):
        super().__init__()
        stride = max(1, input_len // max(output_len, 1))
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            stride=stride, padding=kernel_size // 2,
        )
        self.target = output_len
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))
        h = F.adaptive_avg_pool1d(h, self.target)  # snap to exact T_out
        return h.transpose(1, 2)
 
 
class AttnTimeAdapter(nn.Module):
    """Cross-attention with output_len learned query tokens (Perceiver-style).
    Content-dependent length mapping. O(T_in * T_out * d)."""
 
    def __init__(self, d_model: int, input_len: int, output_len: int, n_heads: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_len, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.queries.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return out
 
class MLPTimeAdapter(nn.Module):

    def __init__(
        self,
        d_model,
        input_len,
        output_len,
        hidden=None,
    ):
        super().__init__()

        hidden = hidden or d_model*2

        self.net = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_len)
        )


    def forward(self,x):

        # B,T,D -> B,D,T
        h = x.transpose(1,2)

        h = self.net(h)

        return h.transpose(1,2)

class LastTokenAdapter(nn.Module):

    def __init__(
        self,
        d_model,
        input_len,
        output_len,
    ):
        super().__init__()

        self.output_len = output_len

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(d_model, output_len*d_model),
            nn.GELU(),
            )

        #self.proj = nn.Linear(
        #    d_model,
        #    output_len*d_model
        #)


    def forward(self,x):

        h = x[:, -1]

        h = self.proj(h)

        return h.view(x.size(0), self.output_len, -1)
 
_ADAPTERS = {

    "linear":
        LinearTimeAdapter,

    "conv":
        ConvTimeAdapter,

    "pool":
        PoolTimeAdapter,

    "attn":
        AttnTimeAdapter,

    "mlp":
        MLPTimeAdapter,

    "last_token":
        LastTokenAdapter,
}
 
# =============================================================================
# Full seq2seq model
# =============================================================================
 
class JeuralJetwork(nn.Module):
    """
    Mamba-based seq2seq encoder for drone trajectory analysis.
 
    Input:  [B, input_len,  n_dim]
    Output: [B, output_len, out_dim]
 
    Args:
        n_dim:             dim of input vectors
        out_dim:           dim of output vectors
        input_len:         input sequence length
        output_len:        output sequence length
        d_model:           internal hidden dim (multiple of 64 recommended)
        n_encoder_layers:  Mamba blocks before the time adapter
        n_decoder_layers:  Mamba blocks after the time adapter
        time_adapter:      'linear' | 'pool' | 'conv' | 'attn'
        adapter_kwargs:    extra kwargs to the chosen adapter
        block_type:        'simple' | 'advanced'  (advanced = +SwiGLU MLP, LayerScale, DropPath)
        use_norm:          pre-norm RMSNorm in blocks (recommended True)
        layer_scale:       LayerScale init for advanced blocks; 0 disables
        drop_path:         stochastic depth probability for advanced blocks
        mlp_mult:          SwiGLU expansion factor for advanced blocks
        mamba_kwargs:      extra kwargs forwarded to Mamba3 (d_state, headdim, ...)
    """
 
    def __init__(
        self,
        n_dim: int,
        out_dim: int,
        input_len: int,
        output_len: int,
        d_model: int = 256,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 1,
        time_adapter: str = "linear",
        adapter_kwargs: dict | None = None,
        block_type: str = "advanced",
        in_norm: str = "rms",
        out_norm: str = "rms",
        layer_scale: float = 1e-4,
        drop_path: float = 0.0,
        mlp_mult: float = 4.0,
        mamba_type: str = "mamba",
        mamba_kwargs: dict | None = None,
    ):
        super().__init__()

        self.input_len = input_len
        self.output_len = output_len
        self.out_dim = out_dim
        self.n_dim = n_dim
 
        ak = adapter_kwargs or {}
        mk = mamba_kwargs or {}
 
        if block_type not in _BLOCKS:
            raise ValueError(
                f"block_type must be one of {list(_BLOCKS)}, got {block_type!r}"
            )
        Block = _BLOCKS[block_type]
 
        if mamba_type not in _MAMBA_IMPLS:
            raise ValueError(
                f"mamba_type must be one of "
                f"{list(_MAMBA_IMPLS)}"
            )
        mamba_impl = _MAMBA_IMPLS[mamba_type]

        block_kwargs = dict(mamba_impl=mamba_impl, **mk)
        if block_type == "advanced":
            block_kwargs.update(
                mlp_mult=mlp_mult,
                layer_scale=layer_scale,
                drop_path=drop_path,
            )

        # input
        self.in_proj = Projection(
                n_dim,
                d_model,
                norm=in_norm,
            )


        # encoder
        self.encoder = nn.ModuleList(
                Block(d_model=d_model, **block_kwargs)
                for _ in range(n_encoder_layers)
            )


        # time mapping
        self.time_projector = _ADAPTERS[time_adapter](
                d_model,
                input_len,
                output_len,
                **ak,
            )


        # decoder
        self.decoder = nn.ModuleList(
                Block(d_model=d_model, **block_kwargs)
                for _ in range(n_decoder_layers)
            )


        # output
        self.out_proj = Projection(
                d_model,
                out_dim,
                norm=out_norm,
            )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)

        for blk in self.encoder:
            h = blk(h)


        h = self.time_projector(h)


        for blk in self.decoder:
            h = blk(h)


        return self.out_proj(h)
 
 
# =============================================================================
# Smoke test
# =============================================================================
 
if __name__ == "__main__":
    # NOTE: Mamba-3's fast kernel path expects bf16 — that's why we cast model
    #       and inputs to torch.bfloat16 below. For fp32 prototyping, swap
    #       `from mamba_ssm import Mamba3` for `Mamba2` (and replace Mamba3
    #       inside the block classes), then drop the `.to(torch.bfloat16)` calls.
    #       Also note: Mamba-3's `chunk_size` must divide the sequence length
    #       the block sees (input_len for encoder, output_len for decoder),
    #       so prefer lengths divisible by 16/32 or pass an explicit chunk_size
    #       via mamba_kwargs.
 
    torch.manual_seed(0)
 
    B, T_in, T_out = 4, 400, 100
    n_dim, out_dim = 9, 3   # e.g. 9 input features (pos+vel+acc), 3 output features (xyz)
 
    model = JeuralJetwork(
        n_dim=n_dim,
        out_dim=out_dim,
        input_len=T_in,
        output_len=T_out,
        d_model=256,
        n_encoder_layers=3,
        n_decoder_layers=1,
        time_adapter="conv",         # try also: 'linear', 'pool', 'attn'
        block_type="advanced",       # or 'simple'
        use_norm=True,
        layer_scale=1e-4,
        drop_path=0.05,
    ).to("cuda").to(torch.bfloat16)
 
    x = torch.randn(B, T_in, n_dim, dtype=torch.bfloat16, device="cuda")
    y = model(x)
 
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
 

