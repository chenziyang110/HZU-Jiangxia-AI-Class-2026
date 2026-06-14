import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------
# 1. 位置编码
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ---------------------------
# 2. 多头注意力
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        assert d_model % nhead == 0

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(1)

        q = self.w_q(q).view(-1, batch_size, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(-1, batch_size, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(-1, batch_size, self.nhead, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(-1, batch_size, self.d_model)
        output = self.fc(output)
        return output

# ---------------------------
# 3. 前馈网络
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ---------------------------
# 4. 编码器层
# ---------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x

# ---------------------------
# 5. 解码器层
# ---------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x

# ---------------------------
# 6. 完整 Transformer
# ---------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.src_emb(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        y = self.pos_enc(self.tgt_emb(tgt))
        for layer in self.decoder_layers:
            y = layer(y, enc_out, src_mask, tgt_mask)
        return y

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.fc(dec_out)

# ---------------------------
# 测试：成功复现！
# ---------------------------
if __name__ == "__main__":
    src_vocab = 1000
    tgt_vocab = 1000

    model = Transformer(src_vocab, tgt_vocab)

    # 模拟输入
    src = torch.randint(0, src_vocab, (10, 2))
    tgt = torch.randint(0, tgt_vocab, (8, 2))

    out = model(src, tgt)
    print("✅ Transformer 模型复现成功！")
    print("输出形状:", out.shape)