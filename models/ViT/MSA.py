import torch.nn as nn


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, n_heads=12, qkv_bias=True, embed_vector=16 * 16 * 3, attention_drop_out=0.0, projection_drop_out=0.0):
        super(Multi_Head_Self_Attention, self).__init__()
        self.n_heads = n_heads  # n_heads = 16
        self.embed_vector = embed_vector  # n_patch * n_patch * 3 = 16 * 16 * 3
        self.one_head_size = embed_vector // n_heads  # 768 // 16 = 48
        self.scale = self.one_head_size ** (-0.5)  # root
        self.qkv = nn.Linear(embed_vector, embed_vector * 3,bias=qkv_bias)  # 768 -> 2304
        self.attention_drop = nn.Dropout(attention_drop_out)  # dropout
        self.projection = nn.Sequential(nn.Linear(embed_vector, embed_vector), nn.Dropout(projection_drop_out))

    def forward(self, x):
        """
        Code
        # explain
        """
        n_batch, n_patches, embed_vector = x.shape
        # Embedding_vector (n_batch, n_patches+1, embed_vector)
        qkv = self.qkv(x)
        # embed_vector => 3*embed_vector 로 늘려주면서 Feature를 더 세세하게 나눔
        qkv = qkv.reshape(n_batch, n_patches, 3, self.n_heads, self.one_head_size)
        # Query, Key, Value로 나누기 위해서 3개 차원으로 나눔
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Q,K,V 로 나누기 편하도록 차원 재구성
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Attention 계산을 위해 각각의 값을 변수에 저장 (n_batch, n_heads, n_patches+1, one_head_size)
        k_t = k.transpose(2, 3)
        # Transpose for dot product k_t = (n_batch, n_heads, one_head_size, n_patches+1)
        # q = (n_batch, n_heads, n_patches+1, one_head_size)
        # q @ k_t = (n_batch, n_heads, n_patches+1, n_patches+1)
        attn = (q @ k_t) * self.scale
        # Query에 대한 Attention을 구하기 위해 k_t와 dot product를 하고 scaling
        attn = attn.softmax(dim=-1)
        attn = self.attention_drop(attn)
        # Dropout
        attn_result = attn @ v
        # attn = (n_batch, n_heads, n_patches+1, n_patches+1)
        # v = (n_batch, n_heads, n_patches+1, one_head_size)
        # attn @ v = (n_batch, n_heads, n_patches+1, one_head_size)
        attn_result_t = attn_result.traspose(1, 2)
        # attn_result_t = (n_batch, n_patches+1, n_heads, one_head_size)
        final_result = attn_result_t.flatten(2)
        # final_result = (n_batch, n_patches+1, n_heads * one_head_size)
        x = self.projection(final_result)

        return x