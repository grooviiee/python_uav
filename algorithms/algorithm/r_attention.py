import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w_qs = nn.Conv2d(size, size, 1)
        self.w_ks = nn.Conv2d(size, size, 1)
        self.w_vs = nn.Conv2d(size, size, 1)

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v):
        residual = q
        q = self.w_qs(q).permute(0, 2, 3, 1)
        k = self.w_ks(k).permute(0, 2, 3, 1)
        v = self.w_vs(v).permute(0, 2, 3, 1)

        attention = self.attention(q, k, v).permute(0, 3, 1, 2)

        out = attention + residual
        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(2, 3))
        output = torch.matmul(attn, v)

        return output