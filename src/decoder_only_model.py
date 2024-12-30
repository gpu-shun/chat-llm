import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerBlock


class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, max_seq_len, num_heads, dropout_rate=0.1):
        super(DecoderOnlyModel, self).__init__()
        # vocab_sizeは扱うトークンIDの種類を表す
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._generate_positional_embedding(embed_dim, max_seq_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout_rate, 4) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        embeddings = self.embedding(input_ids)
        # 位置エンコーディングの値をトークンの長さ分取り出す。その後バッチの数だけ複製する。
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = embeddings + pos_encoding

        for transformer_block in self.transformer_blocks:
            seq_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)
            x = transformer_block(x, x, x, seq_mask)
        
        x = self.layer_norm(x)
        return self.output_layer(x)

    def _generate_positional_embedding(self, embed_dim, max_seq_len):
        # pytorchのtorch.agange()はデフォルトでint64になるため。torch.floatを指定してfloat32にする
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe