import os
import random
import math
import pandas as pd
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# ── SEED & DEVICE SETUP ──────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# ── CONFIGURATION ────────────────────────────────────────────────────────────
class CFG:
    path_data     = "data/ml-1m"
    max_seq_len   = 100
    dim_hidden    = 512
    n_layers      = 4
    dropout_rate  = 0.2
    bs            = 1024
    n_epochs      = 40
    lr            = 1e-3
    wd            = 1e-5
    n_neg_samples = 100
    k_top         = 10

cfg = CFG()

# ── DATASET PROCESSING ───────────────────────────────────────────────────────
class MovieLensDatasetBuilder:
    def __init__(self, path_data):
        file_path = os.path.join(path_data, "ratings.dat")
        df = pd.read_csv(
            file_path, sep="::", engine="python",
            names=["userId","movieId","rating","timestamp"],
            encoding="latin-1"
        )
        self.uid2idx = {uid: i for i, uid in enumerate(df.userId.unique())}
        self.iid2idx = {iid: i for i, iid in enumerate(df.movieId.unique())}
        self.idx2iid = {v+1: k for k, v in self.iid2idx.items()}

        df["uidx"] = df.userId.map(self.uid2idx)
        df["iidx"] = df.movieId.map(self.iid2idx).add(1)  # pad=0

        df = df.sort_values(["uidx", "timestamp"])
        user_histories = defaultdict(list)
        for uid, grp in df.groupby("uidx"):
            user_histories[uid] = grp["iidx"].tolist()

        self.train_sequences = {u: seq[:-1] for u, seq in user_histories.items()}
        self.valid_targets   = {u: seq[-1]  for u, seq in user_histories.items()}
        self.n_users = len(self.train_sequences)
        self.n_items = len(self.iid2idx)

# ── PYTORCH DATASET ──────────────────────────────────────────────────────────
class MovieSeqDataset(Dataset):
    def __init__(self, train_sequences, valid_targets, max_seq_len):
        self.user_ids = list(train_sequences.keys())
        self.train_sequences = train_sequences
        self.valid_targets = valid_targets
        self.max_len = max_seq_len

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        uid = self.user_ids[index]
        sequence = self.train_sequences[uid]
        x = sequence[-self.max_len:]
        pad_len = self.max_len - len(x)
        x = [0] * pad_len + x
        y = self.valid_targets[uid]
        return torch.LongTensor(x), torch.LongTensor([y])

# ── LOAD DATA ────────────────────────────────────────────────────────────────
dataset_builder = MovieLensDatasetBuilder(cfg.path_data)
train_data = MovieSeqDataset(dataset_builder.train_sequences, dataset_builder.valid_targets, cfg.max_seq_len)
train_loader = DataLoader(
    train_data,
    batch_size=cfg.bs,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
class GRURecModel(nn.Module):
    def __init__(self, n_items, cfg):
        super().__init__()
        self.item_embed = nn.Embedding(n_items+1, cfg.dim_hidden, padding_idx=0)
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.gru = nn.GRU(
            input_size=cfg.dim_hidden,
            hidden_size=cfg.dim_hidden,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout_rate
        )
        self.ln = nn.LayerNorm(cfg.dim_hidden)
        self.fc = nn.Linear(cfg.dim_hidden, n_items+1)
        self.fc.weight = self.item_embed.weight

    def forward(self, x):
        x = self.dropout(self.item_embed(x))
        h, _ = self.gru(x)
        h = self.ln(h)
        return self.fc(h)

model = GRURecModel(dataset_builder.n_items, cfg).to(DEVICE)

# ── TRAINING SETUP ────────────────────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
steps_total = cfg.n_epochs * len(train_loader)
warmup_steps = int(steps_total * 0.1)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps_total)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_one_epoch():
    model.train()
    running_loss = 0.0
    for x, y in tqdm(train_loader, desc="Training"):
        x = x.to(DEVICE)
        y = y.squeeze(1).to(DEVICE)

        out = model(x)[:, -1, :]
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

# ── EVALUATION + SUBMISSION ──────────────────────────────────────────────────
def evaluate_and_submit():
    model.eval()
    total_hit, total_ndcg = 0.0, 0.0
    results = []

    with torch.no_grad():
        for uid in range(dataset_builder.n_users):
            seq = dataset_builder.train_sequences[uid]
            if len(seq) > cfg.max_seq_len:
                seq = seq[-cfg.max_seq_len:]
            else:
                seq = [0]*(cfg.max_seq_len - len(seq)) + seq
            x = torch.LongTensor([seq]).to(DEVICE)

            true = dataset_builder.valid_targets[uid]
            seen = set(dataset_builder.train_sequences[uid])
            pool = [i for i in range(1, dataset_builder.n_items+1) if i not in seen]
            candidates = [true] + random.sample(pool, cfg.n_neg_samples)

            scores = model(x)[0, -1, candidates]
            rank = torch.argsort(torch.argsort(-scores))[0].item()

            if rank < cfg.k_top:
                total_hit  += 1
                total_ndcg += 1.0 / math.log2(rank + 2)

            full_scores = model(x)[0, -1, 1:dataset_builder.n_items+1]
            top_indices = torch.topk(full_scores, cfg.k_top).indices.cpu() + 1
            for item_id in top_indices:
                results.append({
                    "userId": uid + 1,
                    "itemId": dataset_builder.idx2iid[item_id.item()]
                })

    pd.DataFrame(results, columns=["userId", "itemId"]).to_csv("gru4rec_sub.csv", index=False)
    return total_hit / dataset_builder.n_users, total_ndcg / dataset_builder.n_users

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
best_score = 0.0
for ep in range(1, cfg.n_epochs + 1):
    train_loss = train_one_epoch()
    hit, ndcg = evaluate_and_submit()
    print(f"Epoch {ep:02d} ▶ Loss: {train_loss:.4f} | Hit@{cfg.k_top}: {hit:.4f} | NDCG@{cfg.k_top}: {ndcg:.4f}")
    if ndcg > best_score:
        best_score = ndcg
        torch.save(model.state_dict(), "best_gru4rec_ml1m.pth")
        print("Best model is saved")

print("Training Complete!")
print("Best NDCG@10:", best_score)
