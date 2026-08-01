"""
Profiler-as-Early-Layer vs Traditional Dense Transformer

Tests: what if the profiler IS the first layers of the model, routing
to domain-specific downstream computation?

Architecture A: Traditional dense transformer (all layers shared)
Architecture B: Profiler (first 2 layers) → profile → domain-specific heads

Same data. Same total parameters (matched budgets). Compare PPL and speed.

Run: python profiler_as_layer.py
"""
import numpy as np
import time, json, os, warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
# MULTI-DOMAIN DATA (reuse from transformer_training)
# ═══════════════════════════════════════════════════════════════════

DOMAIN_DATA = {
    'code': """
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)

def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quicksort(left) + [pivot] + quicksort(right)

class Node:
    def __init__(self, value): self.value = value; self.next = None

def reverse_list(head):
    prev = None; current = head
    while current: nxt = current.next; current.next = prev; prev = current; current = nxt
    return prev

for i in range(10):
    if i % 2 == 0: print(i, "even")
    else: print(i, "odd")

def binary_search(arr, target):
    lo, hi = 0, len(arr)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if arr[mid] == target: return mid
        elif arr[mid] < target: lo = mid+1
        else: hi = mid-1
    return -1
""",
    'math': """
The derivative of f(x) = x^n is f'(x) = n*x^(n-1).
Integration by parts: ∫u dv = uv - ∫v du.
The quadratic formula: x = (-b ± √(b²-4ac)) / 2a.
Pythagorean theorem: a² + b² = c² for right triangles.
Euler's identity: e^(iπ) + 1 = 0 connects five fundamental constants.
The area of a circle is A = πr² where r is the radius.
Probability: P(A|B) = P(A∩B)/P(B) by Bayes theorem.
Linear regression minimizes Σ(y_i - (mx_i + b))².
The Fibonacci sequence: F_n = F_{n-1} + F_{n-2} with F_0=0, F_1=1.
Matrix multiplication: (AB)_{ij} = Σ_k A_{ik} B_{kj}.
""",
    'stories': """
Once upon a time in a small village, there lived a curious cat named Whiskers.
Every morning, Whiskers would wander to the edge of the forest and watch the birds.
The old oak tree had stood there for centuries, its branches reaching toward the sky.
Sarah packed her backpack with sandwiches and a thermos of hot chocolate.
The dragon was not fierce at all but rather lonely, seeking a friend to share tea with.
Under the starlit sky, the children gathered around the campfire to hear grandfather's tales.
The magic mirror reflected not one's appearance but one's truest desire.
A gentle rain began to fall as the farmer planted the last seeds of spring.
""",
    'wiki': """
The Industrial Revolution began in Great Britain during the late 18th century.
Photosynthesis is the process by which plants convert sunlight into chemical energy.
The human brain contains approximately 86 billion neurons connected by trillions of synapses.
Quantum mechanics describes the behavior of matter and energy at atomic scales.
The Roman Empire at its peak controlled territories across Europe, North Africa, and Asia.
Water covers about 71 percent of the Earth's surface, mostly in oceans and seas.
The speed of light in vacuum is exactly 299,792,458 meters per second.
DNA replication is the biological process of producing two identical replicas from one DNA molecule.
"""
}


def build_dataset(seq_len=64, repeats=15):
    all_text = ""
    boundaries = []
    pos = 0
    for name, text in DOMAIN_DATA.items():
        rpt = text * repeats
        all_text += rpt
        boundaries.append((pos, pos + len(rpt), name))
        pos += len(rpt)

    chars = sorted(list(set(all_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in all_text], dtype=torch.long)
    n_seq = (len(data) - 1) // seq_len

    X = torch.zeros(n_seq, seq_len, dtype=torch.long)
    Y = torch.zeros(n_seq, seq_len, dtype=torch.long)
    domains = torch.zeros(n_seq, seq_len, dtype=torch.long)
    domain_map = {name: i for i, name in enumerate(DOMAIN_DATA.keys())}

    for i in range(n_seq):
        start = i * seq_len
        X[i] = data[start:start+seq_len]
        Y[i] = data[start+1:start+seq_len+1]
        for j in range(seq_len):
            cp = start + j
            for ds, de, dn in boundaries:
                if ds <= cp < de:
                    domains[i, j] = domain_map[dn]
                    break

    n_train = int(n_seq * 0.8)
    idx = torch.randperm(n_seq)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    return (X[train_idx], Y[train_idx], domains[train_idx],
            X[val_idx], Y[val_idx], domains[val_idx],
            len(chars), domain_map)


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE A: Traditional Dense Transformer
# ═══════════════════════════════════════════════════════════════════

class DenseTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # Causal self-attention
        B, S, D = x.shape
        mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class TraditionalDenseTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=4, d_ff=128, seq_len=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([
            DenseTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, token_ids):
        B, S = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed[:, :S, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# ═══════════════════════════════════════════════════════════════════
# ARCHITECTURE B: Profiler as Early Layers → Domain-Specific Heads
# ═══════════════════════════════════════════════════════════════════

class ProfilerGuidedTransformer(nn.Module):
    """
    First 2 layers = shared profiler (classifies input → domain profile).
    Remaining 2 layers = domain-specific heads (one per domain).
    Router picks top-k heads based on profile from early layers.
    """
    def __init__(self, vocab_size, n_domains=4, d_model=64, n_heads=2,
                 profiler_layers=2, domain_layers=2, d_ff=128, seq_len=64):
        super().__init__()
        self.n_domains = n_domains
        self.d_model = d_model
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Shared profiler (early layers)
        self.profiler_blocks = nn.ModuleList([
            DenseTransformerBlock(d_model, n_heads, d_ff) for _ in range(profiler_layers)
        ])

        # Domain-specific heads (late layers)
        self.domain_blocks = nn.ModuleList([
            nn.ModuleList([
                DenseTransformerBlock(d_model, n_heads, d_ff) for _ in range(domain_layers)
            ]) for _ in range(n_domains)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_domains)
        ])
        # Weight-tying with token embed
        for head in self.lm_heads:
            head.weight = self.token_embed.weight

        # Profiler: maps profiler output → domain scores
        self.profiler_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_domains),
        )

    def forward(self, token_ids, k=2, temperature=0.1):
        B, S = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed[:, :S, :]

        # Shared profiler layers
        for block in self.profiler_blocks:
            x = block(x)

        # Classify: mean-pool over sequence → domain profile
        pooled = x.mean(dim=1)  # (B, D)
        domain_logits = self.profiler_head(pooled)  # (B, n_domains)
        domain_weights = F.softmax(domain_logits / temperature, dim=-1)

        # Top-k domain selection
        top_k_weights, top_k_idx = torch.topk(domain_weights, min(k, self.n_domains), dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Route through selected domain blocks
        logits = torch.zeros(B, S, self.lm_heads[0].out_features, device=x.device)
        for b in range(B):
            for k_i in range(top_k_idx.shape[-1]):
                d_idx = top_k_idx[b, k_i].item()
                w = top_k_weights[b, k_i]
                # Process through domain-specific layers
                x_domain = x[b:b+1]  # Keep batch dim
                for block in self.domain_blocks[d_idx]:
                    x_domain = block(x_domain)
                logits[b] += w * self.lm_heads[d_idx](self.ln_f(x_domain))[0]

        return logits


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=6, batch_size=16, lr=3e-3, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(X_train) // batch_size
    print(f"\n  Training {label} ({epochs} epochs, {n_batches} batches)...")
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = F.cross_entropy(val_logits.reshape(-1, val_logits.shape[-1]), Y_val.reshape(-1)).item()

        elapsed = time.perf_counter() - t0
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs}: train={total_loss/n_batches:.3f} val={val_loss:.3f} ppl={np.exp(val_loss):.1f} t={elapsed:.1f}s")


@torch.no_grad()
def evaluate_model(model, X_val, Y_val, domains_val, domain_map):
    model.eval()
    logits = model(X_val)
    loss_all = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y_val.reshape(-1)).item()
    results = {'overall_ppl': np.exp(loss_all), 'per_domain': {}}
    inv_map = {v: k for k, v in domain_map.items()}
    for d_idx, d_name in inv_map.items():
        mask = (domains_val == d_idx)
        if mask.sum() == 0: continue
        logits_d = logits[mask]; targets_d = Y_val[mask]
        loss_d = F.cross_entropy(logits_d.reshape(-1, logits_d.shape[-1]), targets_d.reshape(-1)).item()
        results['per_domain'][d_name] = {'ppl': np.exp(loss_d), 'n': mask.sum().item()}
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("PROFILER-AS-EARLY-LAYER vs TRADITIONAL DENSE TRANSFORMER")
    print("="*70)

    device = torch.device('cpu')
    X_train, Y_train, D_train, X_val, Y_val, D_val, vocab_size, domain_map = build_dataset(seq_len=64, repeats=15)
    n_domains = len(domain_map)
    print(f"  Vocab: {vocab_size} | Domains: {n_domains} | Train seqs: {X_train.shape[0]}")

    X_train = X_train.to(device); Y_train = Y_train.to(device)
    X_val = X_val.to(device); Y_val = Y_val.to(device)

    # ── Model A: Traditional Dense ──
    model_a = TraditionalDenseTransformer(
        vocab_size, d_model=64, n_heads=2, n_layers=4, d_ff=128, seq_len=64
    ).to(device)
    n_a = sum(p.numel() for p in model_a.parameters())
    print(f"\n  Model A (Traditional Dense): {n_a:,} params (4 shared layers)")

    t0 = time.perf_counter()
    train_model(model_a, X_train, Y_train, X_val, Y_val, epochs=6, batch_size=16, lr=3e-3, label="Traditional Dense")
    time_a = time.perf_counter() - t0
    eval_a = evaluate_model(model_a, X_val, Y_val, D_val, domain_map)

    # Speed
    with torch.no_grad():
        for _ in range(5): model_a(X_val[:4])
        t0 = time.perf_counter()
        for _ in range(20): model_a(X_val[:4])
        ms_a = (time.perf_counter() - t0) / 20 * 1000
        tok_a = X_val[:4].numel() / (ms_a / 1000)

    # ── Model B: Profiler-Routed ──
    model_b = ProfilerGuidedTransformer(
        vocab_size, n_domains=n_domains, d_model=64, n_heads=2,
        profiler_layers=2, domain_layers=2, d_ff=128, seq_len=64
    ).to(device)
    n_b = sum(p.numel() for p in model_b.parameters())
    print(f"\n  Model B (Profiler-Routed):    {n_b:,} params (2 shared + {n_domains}×2 domain layers)")

    t0 = time.perf_counter()
    train_model(model_b, X_train, Y_train, X_val, Y_val, epochs=6, batch_size=16, lr=3e-3, label="Profiler-Routed")
    time_b = time.perf_counter() - t0
    eval_b = evaluate_model(model_b, X_val, Y_val, D_val, domain_map)

    with torch.no_grad():
        for _ in range(5): model_b(X_val[:4])
        t0 = time.perf_counter()
        for _ in range(20): model_b(X_val[:4])
        ms_b = (time.perf_counter() - t0) / 20 * 1000
        tok_b = X_val[:4].numel() / (ms_b / 1000)

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  {'Metric':25s} {'Traditional Dense':>18s} {'Profiler-Routed':>18s}")
    print(f"  {'-'*63}")
    print(f"  {'Total params':25s} {n_a:>18,} {n_b:>18,}")
    print(f"  {'Overall PPL':25s} {eval_a['overall_ppl']:>18.1f} {eval_b['overall_ppl']:>18.1f}")

    for d in sorted(eval_a['per_domain'].keys()):
        pa = eval_a['per_domain'][d]['ppl']
        pb = eval_b['per_domain'][d]['ppl']
        print(f"  {d + ' PPL':25s} {pa:>18.1f} {pb:>18.1f}")

    print(f"  {'Speed (ms)':25s} {ms_a:>18.1f} {ms_b:>18.1f}")
    print(f"  {'Tokens/sec':25s} {tok_a:>18.0f} {tok_b:>18.0f}")
    print(f"  {'Train time (s)':25s} {time_a:>18.1f} {time_b:>18.1f}")

    # Save results
    results = {
        'traditional_dense': {
            'params': n_a, 'overall_ppl': eval_a['overall_ppl'],
            'per_domain': {d: v['ppl'] for d, v in eval_a['per_domain'].items()},
            'speed_ms': ms_a, 'tokens_per_sec': tok_a, 'train_time': time_a,
        },
        'profiler_routed': {
            'params': n_b, 'overall_ppl': eval_b['overall_ppl'],
            'per_domain': {d: v['ppl'] for d, v in eval_b['per_domain'].items()},
            'speed_ms': ms_b, 'tokens_per_sec': tok_b, 'train_time': time_b,
        },
        'config': {'d_model': 64, 'profiler_layers': 2, 'domain_layers': 2, 'n_domains': n_domains},
    }
    with open(os.path.join(HERE, 'profiler_as_layer_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: profiler_as_layer_results.json")


if __name__ == '__main__':
    main()
