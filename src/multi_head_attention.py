import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        # 分割するヘッド数が割り切れるかどうかチェック
        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # クエリー、キー、バリューの生成レイヤを定義
        self.project_query = nn.Linear(embed_size, embed_size)
        self.project_key = nn.Linear(embed_size, embed_size)
        self.project_value = nn.Linear(embed_size, embed_size)

        # 最終層の定義
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        # バッチサイズを保持
        num_batch = query.shape[0]

        # トークンサイズを保持
        seq_length = query.shape[1]

        # キー、クエリ、バリューの生成
        query = self.project_query(query)
        key = self.project_key(key)
        value = self.project_value(value)

        queries = query.view(num_batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = key.view(num_batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = value.view(num_batch, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nhql,nhkl->nhqk", queries, keys)

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=-1)
        out = torch.einsum("nhqk,nhvl->nhql", attention, values)
        out = out.transpose(1, 2).contiguous().view(num_batch, seq_length, self.embed_size)
        return self.fc_out(out)