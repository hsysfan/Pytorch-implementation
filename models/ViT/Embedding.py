from einops.layers.torch import Rearrange

import torch
import torch.nn as nn


class Patch_Embedding(nn.Module):
    def __init__(self, in_channels=3, vector_dim=768, img_size=224, patch_size=16):
        super(Patch_Embedding, self).__init__()
        self.n_patch = patch_size ** 2
        self.embed_vector = nn.Sequential(
            Rearrange('b c (h1 n_patch_h) (w1 n_patch_w) -> b (n_patch_h n_patch_w) (n_patch_h n_patch_w c)'),
            # einops Rearrange를 통해서 아래와 같이 바꿔줌
            # (1, 3, (height // n_patch_h * n_patch_h) (width // n_patch_w * n_patch_w))
            # (1, n_patch_h * n_patch_w, n_patch_h * n_patch_w * c)
            nn.Linear(self.n_patch, vector_dim))
        """
        아래의 코드로 Conv2d를 이용하는게 성능적으로 더 빠르다는데 이유를 아직 몰라서 주석 처리
        kernel_size, stride가 patch_size와 같아지면 한 커널마다 patch_size * patch_size로 만들어줄 수 있다.
        self.embed_vector = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        """

    def forward(self, x):
        """
        Code
        # explain
        """
        batch, channels, height, width = x.shape
        # x = (batch, channels, height, width)
        x = self.embed_vector(x)
        # x = (batch, n_patch * n_patch, embed_vector_dim)


        return x
