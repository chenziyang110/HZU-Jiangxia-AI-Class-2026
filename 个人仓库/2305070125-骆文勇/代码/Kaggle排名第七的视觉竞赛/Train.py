import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

# 配置类 - 强制使用CUDA
class Config:
    seed = 42
    model_name = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"

    img_size = 448
    embedding_dim = 1024
    num_classes = 31

    num_epochs = 15
    batch_size = 4
    grad_accum = 4

    lr = 2e-5
    weight_decay = 1e-3

    arcface_s = 30.0
    arcface_m = 0.50

    use_tta = True
    use_qe = True
    use_rerank = True

    # 强制使用CUDA，不提供CPU回退
    device = torch.device("cuda")
    device_type = "cuda"
    
    # 模型保存路径
    save_dir = Path("trained_model")
    model_save_name = "eva02_jaguar_final.pth"

# 固定随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 提升CUDA稳定性

seed_everything(Config.seed)

# 数据集类 - 移除错误检查逻辑
class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            unique_ids = sorted(df["ground_truth"].unique())
            self.label_map = {name: i for i, name in enumerate(unique_ids)}
            self.df["label"] = self.df["ground_truth"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        img_path = self.img_dir / img_name
        # 移除错误检查，直接读取图片
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, img_name
        return img, torch.tensor(row["label"], dtype=torch.long)

# 数据增强变换
train_transform = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
        transforms.RandomErasing(p=0.25),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
    ]
)

# GeM池化层
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

# ArcFace分类头
class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine).to(Config.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

# 主模型
class EVABoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            Config.model_name, pretrained=False, num_classes=0
        )
        self.feat_dim = self.backbone.num_features
        self.gem = GeM()
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.head = ArcFaceLayer(
            self.feat_dim, Config.num_classes, s=Config.arcface_s, m=Config.arcface_m
        )
        self.load_local_weights()

    def load_local_weights(self):
        weight_path = Path("f:/桌面文件/Kaggle_tiger/pytorch_model.bin")
        if weight_path.exists():
            state_dict = torch.load(weight_path, map_location=Config.device)
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded local weights from {weight_path}")
        else:
            print(f"⚠️  Warning: {weight_path} not found, using random initialization")

    def forward(self, x, label=None):
        features = self.backbone.forward_features(x)
        if features.dim() == 3:
            B, N, C = features.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                features = features[:, -H * W :, :]
            features = features.permute(0, 2, 1).reshape(B, C, H, W)

        emb = self.gem(features).flatten(1)
        emb = self.bn(emb)
        if label is not None:
            return self.head(emb, label)
        return emb

# 训练单轮epoch
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    loss_meter = 0
    for i, (imgs, labels) in enumerate(tqdm(loader, leave=False)):
        imgs, labels = imgs.to(Config.device), labels.to(Config.device)

        # 修正autocast写法
        with torch.amp.autocast(device_type="cuda"):
            loss = criterion(model(imgs, labels), labels)
            loss = loss / Config.grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % Config.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter += loss.item() * Config.grad_accum
    return loss_meter / len(loader)

# 提取特征
@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, names = [], []
    for imgs, fnames in tqdm(loader, desc="Inference"):
        imgs = imgs.to(Config.device)
        f1 = model(imgs)
        if Config.use_tta:
            f2 = model(torch.flip(imgs, [3]))
            f1 = (f1 + f2) / 2
        feats.append(F.normalize(f1, dim=1).cpu())
        names.extend(fnames)
    return torch.cat(feats, dim=0).numpy(), names

# 查询扩展
def query_expansion(emb, top_k=3):
    print("Applying QE...")
    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]
    new_emb = np.zeros_like(emb)
    for i in range(len(emb)):
        new_emb[i] = np.mean(emb[indices[i]], axis=0)
    return new_emb / np.linalg.norm(new_emb, axis=1, keepdims=True)

# 重排序
def k_reciprocal_rerank(prob, k1=20, k2=6, lambda_value=0.3):
    print("Applying Re-ranking...")
    q_g_dist = 1 - prob
    original_dist = q_g_dist.copy()
    initial_rank = np.argsort(original_dist, axis=1)
    nn_k1 = []
    for i in range(prob.shape[0]):
        forward_k1 = initial_rank[i, : k1 + 1]
        backward_k1 = initial_rank[forward_k1, : k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])
    jaccard_dist = np.zeros_like(original_dist)
    for i in range(prob.shape[0]):
        ind_non_zero = np.where(original_dist[i, :] < 0.6)[0]
        ind_images = [
            inv for inv in ind_non_zero if len(np.intersect1d(nn_k1[i], nn_k1[inv])) > 0
        ]
        for j in ind_images:
            intersection = len(np.intersect1d(nn_k1[i], nn_k1[j]))
            union = len(np.union1d(nn_k1[i], nn_k1[j]))
            jaccard_dist[i, j] = 1 - intersection / union
    return 1 - (jaccard_dist * lambda_value + original_dist * (1 - lambda_value))

# 保存检查点函数（每个epoch后调用，覆盖保存）
def save_checkpoint(model, optimizer, scheduler, epoch, config):
    """保存当前epoch的检查点（覆盖模式）"""
    config.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.save_dir / f"{config.model_save_name.replace('.pth', '')}_checkpoint.pth"
    
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': {
            'img_size': config.img_size,
            'num_classes': config.num_classes,
            'model_name': config.model_name,
            'arcface_s': config.arcface_s,
            'arcface_m': config.arcface_m
        }
    }
    
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

# 加载检查点函数（断点续训）
def load_checkpoint(model, optimizer, scheduler, config):
    """加载检查点，返回起始epoch"""
    checkpoint_path = config.save_dir / f"{config.model_save_name.replace('.pth', '')}_checkpoint.pth"
    
    if checkpoint_path.exists():
        print(f"📂 发现检查点文件: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ 已从 epoch {checkpoint['epoch']} 恢复训练，将从 epoch {start_epoch} 继续")
        return start_epoch
    else:
        print(f"📂 未发现检查点文件，从头开始训练")
        return 0

# 保存模型函数
def save_final_model(model, optimizer, scheduler, config):
    """保存最终训练好的模型权重"""
    # 创建保存目录（不存在则创建）
    config.save_dir.mkdir(parents=True, exist_ok=True)
    save_path = config.save_dir / config.model_save_name
    
    # 保存模型核心信息：模型权重、优化器状态、训练配置（便于后续恢复训练或推理）
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': {
            'img_size': config.img_size,
            'num_classes': config.num_classes,
            'model_name': config.model_name,
            'arcface_s': config.arcface_s,
            'arcface_m': config.arcface_m
        },
        'epoch': config.num_epochs
    }
    
    # 保存到本地（使用cpu格式，便于跨设备加载）
    torch.save(save_dict, save_path)
    print(f"\n✅ 最终模型已保存至: {save_path}")

# 主训练和推理流程
if __name__ == '__main__':
    # 数据路径（请根据实际路径修改）
    TRAIN_CSV = "jaguar-re-id/train.csv"
    TEST_CSV = "jaguar-re-id/test.csv"
    TRAIN_DIR = "jaguar-re-id/train/train"
    TEST_DIR = "jaguar-re-id/test/test"

    # 加载数据
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 构建数据加载器（Windows下num_workers=0避免多进程问题）
    train_loader = DataLoader(
        JaguarDataset(train_df, TRAIN_DIR, train_transform),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,  # CUDA下提升数据传输效率
        drop_last=True    # 避免梯度累积时批次不完整
    )

    # 初始化模型和优化器
    model = EVABoss().to(Config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    # 修正GradScaler初始化
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs
    )

    print(f"🔥 Training EVA-02 Large (448px) on CUDA...")

    # 尝试加载检查点，获取起始epoch
    start_epoch = load_checkpoint(model, optimizer, scheduler, Config)

    # 训练循环
    for epoch in range(start_epoch, Config.num_epochs):
        loss = train_epoch(model, train_loader, optimizer, nn.CrossEntropyLoss(), scaler)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{Config.num_epochs} | Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        
        # 每个epoch后保存模型（覆盖模式）
        save_checkpoint(model, optimizer, scheduler, epoch, Config)
    
    # 训练结束后保存最终模型
    save_final_model(model, optimizer, scheduler, Config)

    # 构建测试集加载器
    unique_test = sorted(set(test_df["query_image"]) | set(test_df["gallery_image"]))
    test_loader = DataLoader(
        JaguarDataset(
            pd.DataFrame({"filename": unique_test}), TEST_DIR, test_transform, True
        ),
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 提取特征并推理
    emb, names = extract_features(model, test_loader)
    img_map = {n: i for i, n in enumerate(names)}

    if Config.use_qe:
        emb = query_expansion(emb)
    sim_matrix = emb @ emb.T
    if Config.use_rerank:
        sim_matrix = k_reciprocal_rerank(sim_matrix)

    # 生成预测结果
    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Mapping"):
        s = sim_matrix[img_map[row["query_image"]], img_map[row["gallery_image"]]]
        preds.append(max(0.0, min(1.0, s)))

    # 保存提交文件
    sub = pd.DataFrame({"row_id": test_df["row_id"], "similarity": preds})
    sub.to_csv("submission.csv", index=False)
    print(f"✅ Done! Mean Sim: {np.mean(preds):.4f}")