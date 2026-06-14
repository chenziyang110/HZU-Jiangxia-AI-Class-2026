import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================
# 1. 位置编码
# ==========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ==========================
# 2. 掩码多头注意力
# ==========================
class MaskedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.w_o(out)


# ==========================
# 3. Decoder 层
# ==========================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MaskedAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 100), nn.ReLU(), nn.Linear(100, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


# ==========================
# 4. 语言模型 = 纯 Decoder
# ==========================
class GPTLite(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layer = DecoderLayer(d_model, n_heads)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.emb(x)
        x = self.pos(x)
        x = self.layer(x, mask)
        return self.fc(x)


# ==========================
# 生成掩码
# ==========================
def get_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    return mask


# ==========================
# 训练 + 生成
# ==========================
if __name__ == "__main__":
    # 语料（统一长度，避免报错）
    sentences = ["我爱吃苹果", "今天天气好", "我想学习啦", "我爱深度学习"]

    # 构建词表
    vocab = ["<pad>"] + sorted(set("".join(sentences)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    # 统一句子长度
    max_len = 6
    data = []
    for s in sentences:
        ids = [w2i[c] for c in s]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))  # pad填充
        data.append(torch.tensor(ids))
    data = torch.stack(data)

    # 模型
    model = GPTLite(vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    print("开始训练语言模型...")
    for epoch in range(800):
        opt.zero_grad()
        x = data[:, :-1]
        y = data[:, 1:]
        mask = get_mask(x.size(1))
        pred = model(x, mask)
        loss = F.cross_entropy(pred.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        opt.step()


    # 生成
    def generate(prompt, max_len=5):
        model.eval()
        ids = [w2i[c] for c in prompt]
        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([ids])
                mask = get_mask(x.size(1))
                pred = model(x, mask)
                next_id = pred.argmax(-1)[:, -1].item()
                ids.append(next_id)
        return "".join([i2w[i] for i in ids])


    print("\n测试生成：")
    print("输入：我 →", generate("我"))
    print("输入：今天 →", generate("今天"))
    print("输入：我爱 →", generate("我爱"))
