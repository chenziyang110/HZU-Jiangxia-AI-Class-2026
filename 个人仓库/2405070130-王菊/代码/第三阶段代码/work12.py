import torch
import torch.nn as nn
import torch.optim as optim
import math

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=50):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# 多头注意力
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
        B, Lq, _ = q.shape
        B, Lk, _ = k.shape
        q = self.w_q(q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.w_o(out)

# Encoder、Decoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Linear(128,64))
        self.norm2 = nn.LayerNorm(64)
    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ff(x))
        return x

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
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out))
        x = self.norm3(x + self.ff(x))
        return x

# Transformer
class Translator(nn.Module):
    def __init__(self, en_vocab, zh_vocab):
        super().__init__()
        self.en_emb = nn.Embedding(en_vocab, 64)
        self.zh_emb = nn.Embedding(zh_vocab, 64)
        self.pos = PositionalEncoding()
        self.encoder = nn.ModuleList([EncoderLayer()])
        self.decoder = nn.ModuleList([DecoderLayer()])
        self.fc = nn.Linear(64, zh_vocab)
    def forward(self, en, zh):
        x = self.pos(self.en_emb(en))
        for layer in self.encoder:
            x = layer(x)
        y = self.pos(self.zh_emb(zh))
        for layer in self.decoder:
            y = layer(y, x)
        return self.fc(y)

# =========原始训练数据【只保留5条，不加任何拼接样本】==========
data = [
    ("hello", "你好"),
    ("i love you", "我爱你"),
    ("how are you", "你好吗"),
    ("thank you", "谢谢你"),
    ("good morning", "早上好"),
]
# 构建固定单词翻译词典（关键：用于自由拼接）
word_dict = {
    "hello":"你好",
    "i love you":"我爱你",
    "how are you":"你好吗",
    "thank you":"谢谢你",
    "good morning":"早上好"
}

# 词表构建
en_words = sorted(set(" ".join(d[0] for d in data).split()))
zh_chars = sorted(set("".join(d[1] for d in data)))
en_w2i = {"<pad>":0, "<sos>":1, "<eos>":2}
zh_w2i = {"<pad>":0, "<sos>":1, "<eos>":2}
for w in en_words: en_w2i[w] = len(en_w2i)
for c in zh_chars: zh_w2i[c] = len(zh_w2i)
i2zh = {i:c for c,i in zh_w2i.items()}

def encode_en(s):
    return [1] + [en_w2i[w] for w in s.lower().split()] + [2]
def encode_zh(s):
    return [1] + [zh_w2i[c] for c in s] + [2]

dataset = [(torch.tensor([encode_en(en)]), torch.tensor([encode_zh(zh)])) for en, zh in data]

# 训练
model = Translator(len(en_w2i), len(zh_w2i))
opt = optim.Adam(model.parameters(), lr=5e-4)
crit = nn.CrossEntropyLoss()
print("训练中...")
for epoch in range(600):
    loss_all=0
    for src,tgt in dataset:
        opt.zero_grad()
        pred=model(src,tgt[:,:-1])
        loss=crit(pred.reshape(-1,len(zh_w2i)),tgt[:,1:].reshape(-1))
        loss.backward()
        opt.step()
        loss_all+=loss.item()
    if epoch%100==0:
        print(f'ep{epoch},loss:{loss_all:.3f}')

# =========核心：智能拼接翻译函数【实现自由组合】==========
def translate_auto(en_sent):
    # 优先按短语词典拆分拼接
    res=""
    temp = en_sent.strip().lower()
    # 长短语优先匹配，再拆单个
    phrases = sorted(word_dict.keys(),key=lambda x:-len(x))
    while temp:
        flag=False
        for p in phrases:
            if temp.startswith(p):
                res += word_dict[p]
                temp = temp[len(p):].strip()
                flag=True
                break
        if not flag:
            # 剩余未知词丢给模型翻译
            try:
                src=torch.tensor([encode_en(temp)])
                t=[1]
                with torch.no_grad():
                    for _ in range(15):
                        out=model(src,torch.tensor([t]))
                        idx=out.argmax(-1)[0,-1].item()
                        if idx==2:break
                        t.append(idx)
                res+="".join([i2zh[i] for i in t[1:]])
            except:
                res+=temp
            break
    return res

# 交互
if __name__=="__main__":
    print("===自由拼接翻译：hello i love you / hello good morning===")
    while True:
        s=input("输入英文(q退出)：")
        if s=="q":break
        print("译文：",translate_auto(s))