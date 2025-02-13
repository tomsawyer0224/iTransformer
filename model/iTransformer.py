import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

import lightning as L

from beartype import beartype
from beartype.typing import Optional, Union, Tuple

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from model.attend import Attend


# helper functions


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t


# attention


class Attention(Module):
    def __init__(self, dim, dim_head=32, heads=4, dropout=0.0, flash=True):
        super().__init__()
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads),
        )

        self.attend = Attend(flash=flash, dropout=dropout)

        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)

        out = self.attend(q, k, v)

        return self.to_out(out)


# feedforward


def FeedForward(dim, mult=4, dropout=0.0):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim),
    )


# main class


class iTransformer(L.LightningModule):
    @beartype
    def __init__(
        self,
        *,
        num_variates: int,
        lookback_len: int,
        depth: int,
        dim: int,
        num_tokens_per_variate=1,
        pred_length: Union[int, Tuple[int, ...]],
        dim_head=32,
        heads=4,
        attn_dropout=0.0,
        ff_mult=4,
        ff_dropout=0.0,
        num_mem_tokens=4,
        flash_attn=True
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.num_variates = num_variates
        self.lookback_len = lookback_len

        self.mem_tokens = (
            nn.Parameter(torch.randn(num_mem_tokens, dim))
            if num_mem_tokens > 0
            else None
        )

        pred_length = cast_tuple(pred_length)
        self.pred_length = pred_length

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            flash=flash_attn,
                        ),
                        nn.LayerNorm(dim),
                        FeedForward(dim, mult=ff_mult, dropout=ff_dropout),
                        nn.LayerNorm(dim),
                    ]
                )
            )

        self.mlp_in = nn.Sequential(
            nn.Linear(lookback_len, dim * num_tokens_per_variate),
            Rearrange("b v (n d) -> b (v n) d", n=num_tokens_per_variate),
            nn.LayerNorm(dim),
        )

        self.pred_heads = ModuleList([])

        for one_pred_length in pred_length:
            head = nn.Sequential(
                Rearrange("b (v n) d -> b v (n d)", n=num_tokens_per_variate),
                nn.Linear(dim * num_tokens_per_variate, one_pred_length),
                Rearrange("b v n -> b n v"),
            )

            self.pred_heads.append(head)

    @beartype
    def forward(
        self, x: Tensor, targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ):
        """
        einstein notation

        b - batch
        n - time
        v - variate
        """
        has_mem = exists(self.mem_tokens)
        assert x.shape[1:] == (self.lookback_len, self.num_variates)

        # the crux of the paper is basically treating variates as the spatial dimension in attention
        # there is a lot of opportunity to improve on this, if the paper is successfully replicated

        x = rearrange(x, "b n v -> b v n")
        x = self.mlp_in(x)

        # memory tokens

        if has_mem:
            m = repeat(self.mem_tokens, "m d -> b m d", b=x.shape[0])
            x, mem_ps = pack([m, x], "b * d")

        # attention and feedforward layers

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        # splice out memory tokens

        if has_mem:
            _, x = unpack(x, mem_ps, "b * d")

        # predicting multiple times

        pred_list = [fn(x) for fn in self.pred_heads]

        # calculate loss if targets is passed in

        if exists(targets):
            targets = cast_tuple(targets)
            assert len(targets) == len(pred_list)

            # assert self.training
            mse_loss = 0.0
            for target, pred in zip(targets, pred_list):
                # assert targets.shape == pred_list.shape
                assert target.shape == pred.shape
                mse_loss = mse_loss + F.mse_loss(target, pred)

            return mse_loss

        if len(pred_list) == 0:
            return pred_list[0]

        pred_dict = dict(zip(self.pred_length, pred_list))
        return pred_dict

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

    def training_step(self, batch):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        inputs, targets = batch
        targets = tuple(targets.values())  # convert targets from dict to tuple

        mse_loss = self(inputs, targets)
        self.manual_backward(mse_loss)
        optimizer.step()
        output_dict = {"mse": mse_loss}
        self.log_dict(output_dict, prog_bar=True, on_step=True, on_epoch=True)
        return output_dict

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = tuple(targets.values())  # convert targets from dict to tuple

        with torch.no_grad():
            mse_loss = self(inputs, targets)

        output_dict = {"mse": mse_loss}
        self.log_dict(output_dict, prog_bar=True, on_step=True, on_epoch=True)
        return output_dict
