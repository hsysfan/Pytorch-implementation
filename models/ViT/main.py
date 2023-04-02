import torch
import torch.nn as nn
from MLP import Multi_Layer_Perceptron
from MSA import Multi_Head_Self_Attention
from Embedding import Patch_Embedding
from einops import repeat


class Block(nn.Module):
    def __init__(self, embed_vector_size, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attention_p=0.0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_vector_size, eps=1e-6)
        self.attention = Multi_Head_Self_Attention(
            n_heads,
            qkv_bias,
            embed_vector_size,
            attention_p,
            p
        )
        self.norm2 = nn.LayerNorm(embed_vector_size, eps=1e-6)
        hidden_features = int(mlp_ratio * embed_vector_size)
        self.mlp = Multi_Layer_Perceptron(
            embed_vector_size,
            hidden_features,
            embed_vector_size
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # residual x + Multi_Head_Self_Attention 진행
        x = x + self.mlp(self.norm2(x))
        # residual
        return x


class Vision_Transformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=1000, depth=12,
                 embed_vector_size=768, n_heads=12, mlp_ratio=4.0, qkv_bias=True, p=0.0, attention_p=0.0):
        super(Vision_Transformer, self).__init__()
        self.patch_embedding = Patch_Embedding(
            in_channels, embed_vector_size, img_size, patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_vector_size))
        # cls_token은 nn.Parameter를 이용해서 학습 도중에 계속해서 값이 변합니다.
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.n_patch, embed_vector_size))
        # pos_embed는 nn.Parameter로 구성되어 있고 패치의 위치 정보를 위해 추가됩니다.
        # 설명은 저렇게 써놨지만 지금까지도 정확히 어떻게 순서를 기억하는지 이해가 좀 안감
        self.pos_dropout = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([
            Block(embed_vector_size=embed_vector_size, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p,
                  attention_p=attention_p)
            for _ in range(depth)]
            # 12개의 block을 쌓는 과정
        )

        self.norm = nn.LayerNorm(embed_vector_size, eps=1e-6)
        self.head = nn.Linear(embed_vector_size, n_classes)

    def forward(self, x):
        n_batch = x.shape[0]
        x = self.patch_embedding(x)  # (n_batch, n_patch, embed_vector_size)
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=n_batch)  # (n_batch, 1, embed_vector_size)
        # cls_token = (batch, 1, embed_vector_dim)
        x = torch.cat((cls_token, x), dim=1)
        # cls_token을 붙여줘서 n_patch + 1이 됩니다.
        # (n_batch, 1+n_patch, embed_vector_size)
        x = torch.add(x, self.pos_embedding)
        # position_embedding을 위해서 더해줍니다.
        # (n_batch, 1+n_patch, embed_vector_size)
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        classification = x[:, 0]
        x = self.head(classification)

        return x


