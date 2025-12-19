# !pip install -q sentence-transformers scikit-learn pandas torch

import os, re, gc, json, time, random
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ==============================
# CONFIG (TWO MODELS)
# ==============================
RNG = 42
random.seed(RNG); np.random.seed(RNG); torch.manual_seed(RNG)

JSON_PATH = "UCFCrime_Train.json"

# Small model for binary
MODEL_BIN = "sentence-transformers/all-mpnet-base-v2"

# Big model for multi-class (Task 2)
MODEL_MC  = "BAAI/bge-large-en-v1.5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

RUN_DIR = "./run_two_models"
os.makedirs(RUN_DIR, exist_ok=True)

# Training knobs
LR = 2e-5

# Binary fine-tune settings (CosineSimilarityLoss on pairs)
EPOCHS_BIN = 1
BATCH_BIN_FT = 32
N_POS_BIN = 4000
N_NEG_BIN = 4000

# Binary probe embedding batch
BATCH_BIN_ENC = 64

# Multi-class fine-tune settings (big model)
EPOCHS_MC = 1
BATCH_MC_ENC = 32        # encoding batch
BATCH_MC_PAIR = 16       # pair loss batch (if OOM -> 8)
BATCH_MC_TRIP = 24       # effective (from sampler); if OOM lower n_classes/n_samples

# Pair counts for multiclass contrastive
N_POS_PER_CLASS_MC = 250    # positives per class
N_NEG_MC = 6000             # negatives total

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================
# LABELS
# ==============================
violent_set = {
    "Abuse","Fighting","Burglary","Robbery","Assault","Shooting","Arrest",
    "Explosion","Arson","Vandalism","Stealing","Shoplifting","RoadAccidents"
}

# ==============================
# LOAD DATA
# ==============================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

rows = []
for video, info in raw.items():
    lab_raw = re.sub(r"\d+.*", "", video)
    lab = "Normal_Videos" if lab_raw.startswith("Normal_Videos") else lab_raw
    txt = " ".join(info.get("sentences", []))
    rows.append({
        "video": video,
        "text": txt,
        "label": lab,
        "binary_label": "Violent" if lab in violent_set else "Non-Violent"
    })

df = pd.DataFrame(rows)
df["text"] = df["text"].fillna("").astype(str)

print("\nSamples:", len(df))
print("Classes:", df["label"].nunique())
print(df["label"].value_counts())

le_all = LabelEncoder().fit(df["label"])
df["label_id"] = le_all.transform(df["label"])
df["bin_id"] = (df["binary_label"] == "Violent").astype(int)

# ==============================
# SPLIT (NO LEAKAGE)
# ==============================
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=RNG)
df_train, df_val  = train_test_split(df_train, test_size=0.2, stratify=df_train["label_id"], random_state=RNG)
print("\nTrain/Val/Test:", len(df_train), len(df_val), len(df_test))

# ==============================
# TASK 0.1: SPEED BENCHMARK (BIN + MC)
# ==============================
def benchmark_ms_per_token(model_name, texts, batch_size=16, runs=3, warmup=1, device="cuda"):
    model = SentenceTransformer(model_name, device=device)
    tok = model.tokenizer
    pad_id = tok.pad_token_id

    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    total_tokens = int((enc["input_ids"] != pad_id).sum().item())
    total_paras = len(texts)

    for _ in range(warmup):
        _ = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    dt = max(t1 - t0, 1e-9)

    out = {
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "paragraphs": total_paras,
        "tokens": total_tokens,
        "ms_per_token": 1000.0 * dt / (runs * max(total_tokens,1)),
        "ms_per_paragraph": 1000.0 * dt / (runs * max(total_paras,1)),
        "paragraphs_per_second": (runs * total_paras) / dt
    }

    del model
    clear_gpu()
    return out

print("\n================ TASK 0.1: SPEED (BIN + MC) ================")
sample_texts = df_train["text"].sample(min(64, len(df_train)), random_state=RNG).tolist()

bench_rows = []
bench_rows.append(benchmark_ms_per_token(MODEL_BIN, sample_texts, batch_size=16, runs=3, warmup=1, device=DEVICE))
bench_rows.append(benchmark_ms_per_token(MODEL_MC,  sample_texts, batch_size=8,  runs=3, warmup=1, device=DEVICE))
bench_df = pd.DataFrame(bench_rows)
print(bench_df)
bench_df.to_csv(os.path.join(RUN_DIR, "task0.1_speed.csv"), index=False)
print("Saved:", os.path.join(RUN_DIR, "task0.1_speed.csv"))

clear_gpu()

# ==============================
# HELPERS
# ==============================
def encode_with(model, texts, batch_size):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

def tune_binary_threshold(clf, X_val, y_val):
    p = clf.predict_proba(X_val)[:, 1]
    ths = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in ths:
        f1 = f1_score(y_val, (p >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1

# ==============================
# BINARY FINE-TUNE (AGAIN) — CosineSimilarityLoss on PAIRS
# (Task 0.2: different loss, Task 1: balanced sampling via constructed pairs)
# ==============================
def build_binary_pairs(df_train, n_pos=4000, n_neg=4000, seed=42):
    rng = random.Random(seed)
    v = df_train[df_train["bin_id"] == 1]["text"].tolist()
    n = df_train[df_train["bin_id"] == 0]["text"].tolist()
    pairs = []

    # POSITIVES (like before): we include (t,t) AND also (same-class different texts)
    for _ in range(n_pos // 2):
        if v:
            a = rng.choice(v); pairs.append(InputExample(texts=[a, a], label=1.0))
        if n:
            a = rng.choice(n); pairs.append(InputExample(texts=[a, a], label=1.0))
    for _ in range(n_pos // 2):
        if len(v) >= 2:
            a, b = rng.sample(v, 2); pairs.append(InputExample(texts=[a, b], label=1.0))
        if len(n) >= 2:
            a, b = rng.sample(n, 2); pairs.append(InputExample(texts=[a, b], label=1.0))

    # NEGATIVES (violent vs nonviolent)
    for _ in range(n_neg):
        a = rng.choice(v)
        b = rng.choice(n)
        pairs.append(InputExample(texts=[a, b], label=0.0))

    rng.shuffle(pairs)
    return pairs

print("\n================ BINARY MODEL (SMALL) ================")
clear_gpu()
bin_encoder = SentenceTransformer(MODEL_BIN, device=DEVICE)

if EPOCHS_BIN > 0:
    print("[Binary fine-tune] CosineSimilarityLoss on balanced pairs...")
    bin_pairs = build_binary_pairs(df_train, n_pos=N_POS_BIN, n_neg=N_NEG_BIN, seed=RNG)
    loader_bin = DataLoader(bin_pairs, batch_size=BATCH_BIN_FT, shuffle=True)
    loss_bin = losses.CosineSimilarityLoss(model=bin_encoder)
    bin_encoder.fit([(loader_bin, loss_bin)],
                    epochs=EPOCHS_BIN,
                    optimizer_params={"lr": LR},
                    use_amp=True,
                    show_progress_bar=True)

# Encode splits with binary encoder
Xtr_bin = encode_with(bin_encoder, df_train["text"].tolist(), batch_size=BATCH_BIN_ENC)
Xva_bin = encode_with(bin_encoder, df_val["text"].tolist(),   batch_size=BATCH_BIN_ENC)
Xte_bin = encode_with(bin_encoder, df_test["text"].tolist(),  batch_size=BATCH_BIN_ENC)

# Binary probe + threshold
clf_bin = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1).fit(Xtr_bin, df_train["bin_id"].values)
best_th, best_val_f1 = tune_binary_threshold(clf_bin, Xva_bin, df_val["bin_id"].values)

pred_bin = (clf_bin.predict_proba(Xte_bin)[:,1] >= best_th).astype(int)
bin_report = classification_report(df_test["bin_id"].values, pred_bin, target_names=["Non-Violent","Violent"], digits=3)
print("\n[BINARY TEST REPORT]\n", bin_report)
print("Best threshold (val):", best_th, "Best val macro-F1:", best_val_f1)

with open(os.path.join(RUN_DIR, "binary_report.txt"), "w") as f:
    f.write(bin_report + f"\n\nbest_threshold={best_th}\nval_macro_f1={best_val_f1}\n")

# ==============================
# MULTI-CLASS (BIG MODEL) — keep what we're doing:
# - minority handling (Task 1)
# - different loss choices (Task 0.2)
# - larger architecture (Task 2)
# - two-stage evaluation: binary gate + violent-class model
# ==============================
print("\n================ MULTI-CLASS MODEL (BIG) ================")
clear_gpu()

train_v = df_train[df_train["binary_label"]=="Violent"].copy()
test_v  = df_test[df_test["binary_label"]=="Violent"].copy()

le_v = LabelEncoder().fit(train_v["label"])
train_v["v_id"] = le_v.transform(train_v["label"])
num_v = len(le_v.classes_)
print("Violent-only classes:", num_v, "| train violent samples:", len(train_v), "| test violent samples:", len(test_v))

# ---- Triplet requires positives in batch => balanced batch sampler
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes=8, n_samples=3, seed=42, batches_per_epoch=200):
        self.labels = np.array(labels)
        self.n_classes = min(n_classes, len(np.unique(labels)))
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)
        self.class_to_idx = {}
        for i, y in enumerate(self.labels):
            self.class_to_idx.setdefault(y, []).append(i)
        self.classes = list(self.class_to_idx.keys())
        self.batch_size = self.n_classes * self.n_samples
        self.batches_per_epoch = batches_per_epoch
    def __len__(self): return self.batches_per_epoch
    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = self.rng.choice(self.classes, size=self.n_classes, replace=False)
            batch = []
            for c in chosen:
                idxs = self.class_to_idx[c]
                take = self.rng.choice(idxs, size=self.n_samples, replace=(len(idxs) < self.n_samples))
                batch.extend(take.tolist())
            yield batch

def build_multiclass_pairs(train_v_df, n_pos_per_class=250, n_neg=6000, seed=42):
    rng = random.Random(seed)
    by_c = {c: train_v_df[train_v_df["label"]==c]["text"].tolist() for c in train_v_df["label"].unique()}

    # sample classes with inverse frequency to boost minorities
    counts = {c: len(by_c[c]) for c in by_c}
    classes = list(by_c.keys())
    weights = np.array([1.0/max(counts[c],1) for c in classes], dtype=np.float64)
    weights = weights / weights.sum()

    pairs = []
    # positives per class (biased to rare classes via sampling)
    for _ in range(n_pos_per_class * len(classes)):
        c = np.random.choice(classes, p=weights)
        if len(by_c[c]) >= 2:
            a, b = rng.sample(by_c[c], 2)
            pairs.append(InputExample(texts=[a, b], label=1.0))

    # negatives across different classes (also biased to include rare)
    for _ in range(n_neg):
        c1 = np.random.choice(classes, p=weights)
        c2 = np.random.choice(classes, p=weights)
        while c2 == c1:
            c2 = np.random.choice(classes, p=weights)
        a = rng.choice(by_c[c1])
        b = rng.choice(by_c[c2])
        pairs.append(InputExample(texts=[a, b], label=0.0))

    rng.shuffle(pairs)
    return pairs

def train_mc_encoder(mode="contrastive"):
    mc_encoder = SentenceTransformer(MODEL_MC, device=DEVICE)

    if mode == "contrastive":
        print("[MC fine-tune] ContrastiveLoss on class-balanced pairs (minority boosted)")
        pairs = build_multiclass_pairs(train_v, n_pos_per_class=N_POS_PER_CLASS_MC, n_neg=N_NEG_MC, seed=RNG)
        loader = DataLoader(pairs, batch_size=BATCH_MC_PAIR, shuffle=True)
        loss_obj = losses.ContrastiveLoss(model=mc_encoder)
        mc_encoder.fit([(loader, loss_obj)], epochs=EPOCHS_MC, optimizer_params={"lr": LR}, use_amp=True, show_progress_bar=True)
        return mc_encoder

    if mode == "triplet":
        print("[MC fine-tune] BatchHardTripletLoss with balanced batches (minority supported)")
        ex = [InputExample(texts=[t], label=int(y)) for t, y in zip(train_v["text"].tolist(), train_v["v_id"].tolist())]
        bb = BalancedBatchSampler(train_v["v_id"].tolist(), n_classes=min(8, num_v), n_samples=3, seed=RNG, batches_per_epoch=200)
        loader = DataLoader(ex, batch_sampler=bb)
        loss_obj = losses.BatchHardTripletLoss(model=mc_encoder, margin=0.3)
        mc_encoder.fit([(loader, loss_obj)], epochs=EPOCHS_MC, optimizer_params={"lr": LR}, use_amp=True, show_progress_bar=True)
        return mc_encoder

    raise ValueError("mode must be contrastive or triplet")

def eval_two_stage(mc_encoder, tag):
    # Encode ALL test texts with mc encoder (stage 2 uses these embeddings)
    Xtr_mc_all = encode_with(mc_encoder, df_train["text"].tolist(), batch_size=BATCH_MC_ENC)
    Xte_mc_all = encode_with(mc_encoder, df_test["text"].tolist(),  batch_size=BATCH_MC_ENC)

    # Train violent-only multi-class probe on violent subset embeddings
    tr_mask = (df_train["binary_label"].values == "Violent")
    Xtr_v = Xtr_mc_all[tr_mask]
    ytr_v = le_v.transform(df_train.loc[tr_mask, "label"].values)

    clf_v = LogisticRegression(max_iter=4000, class_weight="balanced", multi_class="multinomial", n_jobs=-1)
    clf_v.fit(Xtr_v, ytr_v)

    # Stage 1: binary decision from small model/probe
    p_bin = clf_bin.predict_proba(Xte_bin)[:, 1]
    pred_is_violent = (p_bin >= best_th).astype(int)

    # Stage 2: violent class prediction
    pred_full = np.array(["Normal_Videos"] * len(df_test), dtype=object)
    idx_v = np.where(pred_is_violent == 1)[0]
    if len(idx_v) > 0:
        pred_v_ids = clf_v.predict(Xte_mc_all[idx_v])
        pred_full[idx_v] = le_v.inverse_transform(pred_v_ids)

    # Evaluate over ALL labels
    y_true_all = df_test["label"].values
    le_eval = LabelEncoder().fit(y_true_all)
    y_true = le_eval.transform(y_true_all)
    y_pred = le_eval.transform(pred_full)

    rep_txt = classification_report(y_true, y_pred, target_names=le_eval.classes_, digits=3)
    rep_dict = classification_report(y_true, y_pred, target_names=le_eval.classes_, digits=3, output_dict=True)

    # Minority summary (smallest support in test)
    per_class = {k:v for k,v in rep_dict.items() if k not in ["accuracy","macro avg","weighted avg"]}
    minority = sorted([(k, int(v["support"]), float(v["f1-score"])) for k,v in per_class.items()], key=lambda x:x[1])[:8]

    out = {
        "tag": tag,
        "two_stage_macro_f1": rep_dict["macro avg"]["f1-score"],
        "two_stage_weighted_f1": rep_dict["weighted avg"]["f1-score"],
        "two_stage_accuracy": rep_dict["accuracy"],
        "minority_summary": minority
    }

    with open(os.path.join(RUN_DIR, f"{tag}_two_stage_report.txt"), "w") as f:
        f.write(rep_txt)

    print(f"\n[{tag}] TWO-STAGE REPORT (ALL labels)\n{rep_txt}")
    print(f"[{tag}] Minority summary:", minority)
    return out

print("\n[Task 0.2/1/2] Train MC big model with CONTRASTIVE pairs...")
mc_con = train_mc_encoder("contrastive")
res_con = eval_two_stage(mc_con, "mc_contrastive")

clear_gpu()

print("\n[Task 0.2] Train MC big model with TRIPLET...")
mc_tri = train_mc_encoder("triplet")
res_tri = eval_two_stage(mc_tri, "mc_triplet")

summary = pd.DataFrame([res_con, res_tri])
print("\n================ SUMMARY ================")
print(summary)
summary.to_csv(os.path.join(RUN_DIR, "summary.csv"), index=False)
print("\nSaved all outputs to:", RUN_DIR)
