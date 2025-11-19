import torch
import torch.nn as nn


class LocalAttentionModule(nn.Module):
    """
    最简稳定版 Local Attention
    使用大核卷积或分组卷积来模拟局部注意力，确保 100% 不会改变维度形状。
    """

    def __init__(self, dim=768, window_size=[2, 4], num_heads=6):
        super().__init__()
        # 使用 Depthwise Conv 模拟局部窗口注意力，效果类似但极难出错
        # kernel_size 对应 window_size
        padding_h = window_size[0] // 2
        padding_w = window_size[1] // 2

        self.dw_conv = nn.Conv2d(
            dim, dim,
            kernel_size=window_size,
            padding=(padding_h, padding_w),
            groups=dim,  # Depthwise
            bias=False
        )

        self.norm = nn.GroupNorm(1, dim)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        shortcut = x

        # 模拟 Attention
        x = self.dw_conv(x)

        # 处理偶数卷积核导致的尺寸变化 (Crop 回原始尺寸)
        if x.shape[2] != shortcut.shape[2] or x.shape[3] != shortcut.shape[3]:
            h_diff = x.shape[2] - shortcut.shape[2]
            w_diff = x.shape[3] - shortcut.shape[3]
            x = x[:, :, :x.shape[2] - h_diff, :x.shape[3] - w_diff]

        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)

        return x + shortcut