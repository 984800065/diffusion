import math
import loguru
import torch

from torch import nn
from typing import Optional

class TimeEmbedding(nn.Module):
    def __init__(
        self,
        time_channels: int,
    ):
        super().__init__()
        self.time_channels = time_channels
        self.linear_1 = nn.Linear(self.time_channels // 4, self.time_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(self.time_channels, self.time_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_channels // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.linear_1(emb))
        emb = self.linear_2(emb)

        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 32, 
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1, 
        d_k: int = None, 
        n_groups: int = 32
    ):
        super().__init__()
        if d_k is None:
            assert n_channels % n_heads == 0, "n_channels must be divisible by n_heads"
            d_k = n_channels // n_heads
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ = t
        batch_size, n_channels, height, width = x.shape
        x_in = x
        
        # Pre normalize input
        x = self.norm(x)
        # (B, C, H, W) -> (B, H * W, C) == (B, seq, C)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        # (B, seq, C) -> (B, seq, n_heads * d_k * 3)
        qkv: torch.Tensor = self.projection(x)
        # (B, seq, n_heads * d_k * 3) -> (B, seq, n_heads, 3 * d_k)
        qkv = qkv.view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # (B, seq, n_heads, 3 * d_k) -> (B, seq, n_heads, d_k), (B, seq, n_heads, d_k), (B, seq, n_heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # (B, seq, n_heads, d_k), (B, seq, n_heads, d_k) -> (B, seq, seq, n_heads)
        attn: torch.Tensor = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # (B, seq, seq, n_heads) -> (B, seq, seq, n_heads)
        attn = attn.softmax(dim=2)

        # (B, seq, seq, n_heads), (B, seq, n_heads, d_k) -> (B, seq, n_heads, d_k)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # (B, seq, n_heads, d_k) -> (B, seq, n_heads * d_k)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)

        # (B, seq, n_heads * d_k) -> (B, seq, C)
        res = self.output(res)

        res: torch.Tensor = res
        # (B, seq, C) -> (B, C, H, W)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        # Residual connection
        res += x_in
        return res


class DownBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_channels: int, 
        has_attn: bool
    ):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual(x, t)
        x = self.attn(x)
        return x


class DownSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, stride=2)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.conv(x)


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int, 
        time_channels: int, 
        has_attn: bool
    ):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual(x, t)
        x = self.attn(x)
        return x


class UpSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size = 4, stride = 2, padding = 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.tconv(x)


class DDPMUnet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 64,
        channel_mults: list[int] = [1, 2, 2, 2], 
        is_attn: list[bool] = [False, False, False, False], 
        num_res_blocks: int = 2,
    ):
        super().__init__()
        n_resolutions = len(channel_mults)
        # (B, 3, H, W) -> (B, C, H, W)
        self.image_projection = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)
        # (B, C * 4)
        self.time_embedding = TimeEmbedding(n_channels * 4)

        # (B, C, H, W) -> (B, C * \prod\limits_{i=0}^{n_resolutions-1} channel_mults[i], H // 2 ** (n_resolutions - 1), W // 2 ** (n_resolutions - 1))
        down = []
        down_out_channels = n_channels
        self.skip_channels = [n_channels]
        for i in range(n_resolutions):
            tmp_down_in_channels = down_out_channels
            down_out_channels = down_out_channels * channel_mults[i]
            for _ in range(num_res_blocks):
                down.append(DownBlock(tmp_down_in_channels, down_out_channels, n_channels * 4, is_attn[i]))
                self.skip_channels.append(down_out_channels)
                tmp_down_in_channels = down_out_channels
            if i < n_resolutions - 1: 
                down.append(DownSample(down_out_channels))
        self.down = nn.ModuleList(down)
        
        middle_out_channels = middle_in_channels = down_out_channels
        self.middle = MiddleBlock(middle_out_channels, n_channels * 4)

        up = []
        for i in range(n_resolutions):
            if i > 0:
                up.append(UpSample(self.skip_channels[-1]))
            for _ in range(num_res_blocks):
                tmp_up_in_channels = self.skip_channels.pop()
                assert len(self.skip_channels) > 0, "skip_channels must be non-empty"
                up.append(UpBlock(tmp_up_in_channels + tmp_up_in_channels, self.skip_channels[-1], n_channels * 4, is_attn[i]))
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.image_projection(x)
        t = self.time_embedding(t)
        h = [x]

        for down_layer in self.down:
            if isinstance(down_layer, DownSample):
                x = down_layer(x, t)
            else:
                x = down_layer(x, t)
                h.append(x)
        
        x = self.middle(x, t)

        for up_layer in self.up:
            if isinstance(up_layer, UpSample):
                x = up_layer(x, t)
            else:
                s = h.pop()
                x = torch.cat([x, s], dim=1)
                x = up_layer(x, t)
        
        return self.final(self.act(self.norm(x)))