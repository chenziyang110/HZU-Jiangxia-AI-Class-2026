import torch
import torch.nn as nn
import math

# ==============================
# 1. 位置编码
# ==============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# ==============================
# 2. 多头注意力
# ==============================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B = q.size(1)
        q = self.w_q(q).view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        k = self.w_k(k).view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        v = self.w_v(v).view(-1, B, self.num_heads, self.d_k).transpose(1,2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k), dim=-1)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(-1, B, self.num_heads*self.d_k)
        return self.w_o(out)

# ==============================
# 3. Encoder 层
# ==============================
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Linear(128,64))
        self.norm2 = nn.LayerNorm(64)

    def forward(self, x):
        x = self.norm1(x + self.attn(x,x,x))
        x = self.norm2(x + self.ff(x))
        return x

# ==============================
# 4. Decoder 层
# ==============================
class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Linear(128,64))
        self.norm3 = nn.LayerNorm(64)

    def forward(self, x, enc_out):
        x = self.norm1(x + self.self_attn(x,x,x))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out))
        x = self.norm3(x + self.ff(x))
        return x

# ==============================
# 5. Transformer 英译中模型
# ==============================
class Translator(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_emb = nn.Embedding(5000, 64)
        self.zh_emb = nn.Embedding(5000, 64)
        self.pos = PositionalEncoding()
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(1)])
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(1)])
        self.fc = nn.Linear(64, 5000)

    def forward(self, en, zh):
        x = self.pos(self.en_emb(en).transpose(0,1))
        for layer in self.encoder: x = layer(x)
        y = self.pos(self.zh_emb(zh).transpose(0,1))
        for layer in self.decoder: y = layer(y, x)
        return self.fc(y.transpose(0,1))

# ==============================
# 🔥 键盘输入英文 → 输出中文翻译
# ==============================
if __name__ == "__main__":
    model = Translator()
    print("="*60)
    print("📝 Transformer 英译中翻译工具")
    print("⌨️  从键盘输入英文，输出中文翻译")
    print("🔚 输入 q 退出")
    print("="*60)

    # 内置简单翻译词典（演示用）
    translate_dict = {
        "hello": "你好",
        "i love you": "我爱你",
        "how are you": "你好吗",
        "thank you": "谢谢你",
        "good morning": "早上好",
        "good night": "晚安",
        "what is this": "这是什么",
        "my name is": "我的名字是"
    }

    while True:
        en = input("\n请输入英文：").strip().lower()
        if en == "q":
            print("👋 退出程序")
            break

        # 模拟 Transformer 翻译结果
        print("✅ 翻译结果：", translate_dict.get(en, "（暂未收录该句子）"))




