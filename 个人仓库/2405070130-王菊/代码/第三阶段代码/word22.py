import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 掩码多头注意力
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
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)

        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B,L,-1)
        return self.w_o(out)

# Decoder层
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

# 迷你GPT语言模型
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

# 生成下三角掩码
def get_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    return mask

if __name__ == "__main__":
    sentences = ["我爱吃苹果","今天天气好","我想学习啦","我爱深度学习"]
    vocab = ["<pad>"] + sorted(set("".join(sentences)))
    w2i = {w:i for i,w in enumerate(vocab)}
    i2w = {i:w for i,w in enumerate(vocab)}
    vocab_size = len(vocab)

    max_len = 6
    data = []
    for s in sentences:
        ids = [w2i[c] for c in s]
        if len(ids) < max_len:
            ids += [0]*(max_len-len(ids))
        data.append(torch.tensor(ids))
    data = torch.stack(data)

    model = GPTLite(vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("开始训练...")
    for epoch in range(800):
        opt.zero_grad()
        x = data[:,:-1]
        y = data[:,1:]
        mask = get_mask(x.size(1))
        pred = model(x, mask)
        loss = F.cross_entropy(pred.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        opt.step()
        if epoch%200==0:
            print(f"epoch{epoch},loss:{loss.item():.3f}")

    # 生成函数
    def generate(prompt, max_gen=5):
        model.eval()
        # 过滤词表不存在的字符
        ids = []
        for char in prompt:
            if char in w2i:
                ids.append(w2i[char])
            else:
                return f"【{char}】不在词库，可用字：{list(w2i.keys())}"
        with torch.no_grad():
            for _ in range(max_gen):
                inp = torch.tensor([ids])
                m = get_mask(inp.size(1))
                out = model(inp, m)
                next_idx = out.argmax(-1)[0,-1].item()
                ids.append(next_idx)
        return "".join([i2w[i] for i in ids])

    # 自定义输入循环
    print("\n=====训练完成，开始自由输入生成=====")
    print("提示：只能输入：我、爱、吃、苹、果、今、天、气、好、想、学、习、啦、深、度、学、A\n输入q退出")
    while True:
        user_input = input("请输入开头文字：")
        if user_input.lower() == "q":
            print("退出程序")
            break
        ans = generate(user_input)
        print("续写结果：", ans,"\n")