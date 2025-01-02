from multi_head_attention import MultiHeadAttention

import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate, forward_expansion):
        super(TransformerBlock, self).__init__()

        # Masked Multi-Head Attention
        self.attention = MultiHeadAttention(embed_size=embed_size, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        # Masked Self-Attention
        masked_attention_out = self.attention(query, key, value, mask)

        # Add & Norm
        x = self.norm1(masked_attention_out + query)
        x = self.dropout(x)

        # Feed Forward Network
        forward_out = self.feed_forward(x)
        
        # Add & Norm
        x = self.norm2(forward_out + x)
        return self.dropout(x)