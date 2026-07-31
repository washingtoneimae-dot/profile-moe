"""
Profile-MoE vs Learned MoE: Training Benchmark

Trains two identical transformer MoE models on multi-domain text:
  A) Learned Router (DeepSeek-style): trained via full backprop
  B) Profile Router: trained profiler φ(x), calibrated expert profiles

Compares: perplexity, routing accuracy, swap isolation, training speed.
Exports: transformer_training.xlsx

Requires: torch (pip install torch)
Run: python transformer_training.py
"""
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
# MULTI-DOMAIN DATA
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

def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr)//2
    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))
def merge(a,b):
    result = []; i=j=0
    while i<len(a) and j<len(b):
        if a[i]<b[j]: result.append(a[i]); i+=1
        else: result.append(b[j]); j+=1
    return result + a[i:] + b[j:]
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
The chain rule: d/dx f(g(x)) = f'(g(x)) * g'(x).
A Taylor series: f(x) = Σ f^(n)(a)/n! * (x-a)^n.
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
The brave knight rode through the dark forest, sword gleaming in the moonlight.
In the hidden garden behind the stone wall, flowers bloomed in colors never seen before.
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
The Great Wall of China stretches over 21,000 kilometers across northern China.
"""
}


def build_dataset(seq_len=64, repeats=15):
    """Build character-level dataset with domain labels."""
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
    itos = {i: ch for i, ch in enumerate(chars)}

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

    # Train/val split
    n_train = int(n_seq * 0.8)
    idx = torch.randperm(n_seq)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    return (X[train_idx], Y[train_idx], domains[train_idx],
            X[val_idx], Y[val_idx], domains[val_idx],
            len(chars), stoi, itos, domain_map)


# ═══════════════════════════════════════════════════════════════════
# TRANSFORMER MoE MODULES
# ═══════════════════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.profile = None  # set after calibration

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def calibrate(self, profile_vec):
        self.profile = profile_vec


class LearnedRouter(nn.Module):
    """DeepSeek-style: W_r · x → softmax → top-k."""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.W_r = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x):
        logits = self.W_r(x)  # (B, S, n_experts)
        probs = F.softmax(logits / 0.1, dim=-1)
        top_k_weights, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_idx, top_k_weights, probs


class ProfileRouter(nn.Module):
    """Profile-MoE: cos_sim(φ(x), expert_profiles) → top-k. Zero learned params."""
    def __init__(self, temperature=0.1, top_k=2):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k

    def forward(self, input_profiles, expert_profiles):
        # input_profiles: (B, S, d_profile)
        # expert_profiles: (n_experts, d_profile)
        ip_norm = F.normalize(input_profiles, dim=-1)
        ep_norm = F.normalize(expert_profiles, dim=-1)
        sims = torch.matmul(ip_norm, ep_norm.T)  # (B, S, n_experts)
        weights = F.softmax(sims / self.temperature, dim=-1)
        top_k_weights, top_k_idx = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_idx, top_k_weights, sims


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) / np.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)
        attn = attn + mask[None, None, :, :]
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class MoETransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_experts, d_ff, use_profile_routing=False):
        super().__init__()
        self.use_profile_routing = use_profile_routing
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])

        if use_profile_routing:
            self.router = ProfileRouter(temperature=0.1, top_k=2)
            self.profiler = nn.Linear(d_model, n_experts)  # φ(x): d_model → d_profile
        else:
            self.router = LearnedRouter(d_model, n_experts, top_k=2)

        self.n_experts = n_experts

    def forward(self, x):
        # Attention
        x = x + self.attn(self.ln1(x))

        # MoE FFN
        residual = x
        x_norm = self.ln2(x)
        B, S, D = x_norm.shape

        if self.use_profile_routing:
            input_profiles = F.softmax(self.profiler(x_norm), dim=-1)
            expert_profiles = torch.stack([e.profile for e in self.experts])
            top_k_idx, top_k_weights, _ = self.router(input_profiles, expert_profiles)
        else:
            top_k_idx, top_k_weights, _ = self.router(x_norm)

        # Vectorized expert dispatch: group tokens per expert, batch process
        ffn_out = torch.zeros_like(x_norm)
        K = top_k_idx.shape[-1]

        for k in range(K):
            e_idx_flat = top_k_idx[:, :, k].reshape(-1)     # (B*S,)
            w_flat = top_k_weights[:, :, k].reshape(-1, 1)  # (B*S, 1)
            x_flat = x_norm.reshape(-1, D)                   # (B*S, D)

            for e in range(self.n_experts):
                mask = (e_idx_flat == e)
                if mask.sum() == 0:
                    continue
                expert_in = x_flat[mask]                     # (n_tokens, D)
                expert_out = self.experts[e](expert_in)      # (n_tokens, D)
                ffn_out_flat = ffn_out.reshape(-1, D)
                ffn_out_flat[mask] += w_flat[mask] * expert_out

        return residual + ffn_out


class MoETransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 n_experts=4, d_ff=256, seq_len=128, use_profile_routing=False):
        super().__init__()
        self.seq_len = seq_len
        self.use_profile_routing = use_profile_routing

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.layers = nn.ModuleList([
            MoETransformerLayer(d_model, n_heads, n_experts, d_ff, use_profile_routing)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight

    def forward(self, token_ids):
        B, S = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed[:, :S, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def calibrate_experts(self, domain_map):
        """Initialize expert profiles for profile-based routing."""
        n_domains = len(domain_map)
        for layer in self.layers:
            for e_idx, expert in enumerate(layer.experts):
                profile = torch.ones(n_domains) * 0.01
                profile[e_idx % n_domains] = 0.97
                expert.profile = profile / profile.sum()


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(model, X_train, Y_train, X_val, Y_val, domain_map,
                epochs=10, batch_size=16, lr=1e-3, label=None):
    """Train a transformer MoE model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = len(X_train) // batch_size
    history = {'train_loss': [], 'val_loss': [], 'time': []}

    print(f"\n  Training {label} ({epochs} epochs, {n_batches} batches/epoch)...")
    t_start = time.perf_counter()

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

        avg_train_loss = total_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = F.cross_entropy(
                val_logits.reshape(-1, val_logits.shape[-1]), Y_val.reshape(-1)
            ).item()

        elapsed = time.perf_counter() - t_start
        ppl = np.exp(val_loss)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['time'].append(elapsed)

        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:2d}/{epochs}: train_loss={avg_train_loss:.4f} "
                  f"val_loss={val_loss:.4f}  ppl={ppl:.1f}  time={elapsed:.1f}s")

    return history


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, X_val, Y_val, domains_val, domain_map):
    """Evaluate perplexity per domain and routing accuracy."""
    model.eval()
    logits = model(X_val)
    loss_all = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y_val.reshape(-1)).item()

    # Per-domain evaluation
    results = {'overall_ppl': np.exp(loss_all), 'per_domain': {}, 'routing_stats': {}}
    inv_map = {v: k for k, v in domain_map.items()}

    for d_idx, d_name in inv_map.items():
        mask = (domains_val == d_idx)
        if mask.sum() == 0:
            continue
        logits_d = logits[mask]
        targets_d = Y_val[mask]
        loss_d = F.cross_entropy(logits_d.reshape(-1, logits_d.shape[-1]), targets_d.reshape(-1)).item()
        results['per_domain'][d_name] = {'ppl': np.exp(loss_d), 'n_tokens': mask.sum().item()}

    return results


def measure_routing_speed(model, X_sample, n_trials=50):
    """Measure forward pass time."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(X_sample)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_trials):
            _ = model(X_sample)
    elapsed = (time.perf_counter() - t0) / n_trials * 1000

    total_tokens = X_sample.numel()
    return elapsed, total_tokens / (elapsed / 1000)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("TRANSFORMER MoE TRAINING BENCHMARK")
    print("Profile-MoE vs Learned Router (DeepSeek-style)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # ── Data ──
    X_train, Y_train, D_train, X_val, Y_val, D_val, vocab_size, stoi, itos, domain_map = \
        build_dataset(seq_len=64, repeats=15)

    n_domains = len(domain_map)
    print(f"  Vocab: {vocab_size}  |  Domains: {list(domain_map.keys())}")
    print(f"  Train: {X_train.shape[0]} seqs × {X_train.shape[1]} tokens")
    print(f"  Val:   {X_val.shape[0]} seqs × {X_val.shape[1]} tokens")

    X_train = X_train.to(device); Y_train = Y_train.to(device)
    D_train = D_train.to(device)
    X_val = X_val.to(device); Y_val = Y_val.to(device)
    D_val = D_val.to(device)

    # ── Model A: Learned Router ──
    print("\n[1] Training Learned Router model (DeepSeek-style)")
    model_learned = MoETransformer(
        vocab_size, d_model=64, n_heads=2, n_layers=2,
        n_experts=n_domains, d_ff=128, seq_len=64,
        use_profile_routing=False
    ).to(device)

    n_params = sum(p.numel() for p in model_learned.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.perf_counter()
    hist_learned = train_model(model_learned, X_train, Y_train, X_val, Y_val,
                               domain_map, epochs=8, batch_size=16, lr=3e-3,
                               label="Learned Router")
    train_time_learned = time.perf_counter() - t0

    eval_learned = evaluate_model(model_learned, X_val, Y_val, D_val, domain_map)
    print(f"\n  Learned Router Results:")
    print(f"    Overall PPL: {eval_learned['overall_ppl']:.1f}")
    for d, s in eval_learned['per_domain'].items():
        print(f"    {d:10s}: PPL={s['ppl']:.1f}")

    # ── Model B: Profile Router ──
    print("\n[2] Training Profile Router model")
    model_profile = MoETransformer(
        vocab_size, d_model=64, n_heads=2, n_layers=2,
        n_experts=n_domains, d_ff=128, seq_len=64,
        use_profile_routing=True
    ).to(device)
    model_profile.calibrate_experts(domain_map)

    n_params_p = sum(p.numel() for p in model_profile.parameters())
    print(f"  Parameters: {n_params_p:,}")

    t0 = time.perf_counter()
    hist_profile = train_model(model_profile, X_train, Y_train, X_val, Y_val,
                               domain_map, epochs=8, batch_size=16, lr=3e-3,
                               label="Profile Router")
    train_time_profile = time.perf_counter() - t0

    eval_profile = evaluate_model(model_profile, X_val, Y_val, D_val, domain_map)
    print(f"\n  Profile Router Results:")
    print(f"    Overall PPL: {eval_profile['overall_ppl']:.1f}")
    for d, s in eval_profile['per_domain'].items():
        print(f"    {d:10s}: PPL={s['ppl']:.1f}")

    # ── Speed comparison ──
    print("\n[3] Speed comparison")
    X_speed = X_val[:4]
    ms_learned, tps_learned = measure_routing_speed(model_learned, X_speed)
    ms_profile, tps_profile = measure_routing_speed(model_profile, X_speed)
    print(f"  Learned Router: {ms_learned:.2f}ms ({tps_learned:.0f} tok/s)")
    print(f"  Profile Router: {ms_profile:.2f}ms ({tps_profile:.0f} tok/s)")
    print(f"  Speed ratio:    {tps_profile/tps_learned:.3f}x")

    # ── Swap test (profile only — learned router can't swap) ──
    print("\n[4] Swap test (Profile Router only)")
    # Store original code expert profile
    original_profiles = []
    for layer in model_profile.layers:
        orig = [e.profile.clone() for e in layer.experts]
        original_profiles.append(orig)

    # "Swap" code expert by randomizing its profile (simulating a new expert)
    for layer in model_profile.layers:
        code_expert = layer.experts[0]
        new_profile = torch.rand(n_domains)
        new_profile[0] = 3.0  # Still best at code, but different overall
        code_expert.profile = new_profile / new_profile.sum()

    eval_swapped = evaluate_model(model_profile, X_val, Y_val, D_val, domain_map)
    print(f"  BEFORE swap — code PPL: {eval_profile['per_domain']['code']['ppl']:.1f}")
    print(f"  AFTER  swap — code PPL: {eval_swapped['per_domain']['code']['ppl']:.1f}")
    for d in ['math', 'stories', 'wiki']:
        before = eval_profile['per_domain'][d]['ppl']
        after = eval_swapped['per_domain'][d]['ppl']
        delta = abs(after - before)
        print(f"  {d:10s}: {before:.1f} → {after:.1f}  (Δ={delta:.1f}  {'✓ STABLE' if delta < before*0.1 else '⚠ SHIFTED'})")

    # Restore
    for l_idx, layer in enumerate(model_profile.layers):
        for e_idx, expert in enumerate(layer.experts):
            expert.profile = original_profiles[l_idx][e_idx]

    # ── Routing parameter comparison ──
    print("\n[5] Router parameter count")
    learned_router_params = sum(
        p.numel() for layer in model_learned.layers
        for n, p in layer.router.named_parameters()
    )
    profile_router_params = sum(
        p.numel() for layer in model_profile.layers
        for n, p in layer.router.named_parameters()
    )
    profile_profiler_params = sum(
        p.numel() for layer in model_profile.layers
        for n, p in layer.profiler.named_parameters()
    )
    print(f"  Learned Router: {learned_router_params} learned routing params")
    print(f"  Profile Router: {profile_router_params} routing params (ZERO!)")
    print(f"  Profile Profiler: {profile_profiler_params} params (φ(x) only)")

    # ── Summary ──
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"""
    {'Metric':30s} {'Learned Router':>16s} {'Profile Router':>16s}
    {'-'*64}
    {'Overall PPL':30s} {eval_learned['overall_ppl']:16.1f} {eval_profile['overall_ppl']:16.1f}
    {'Code PPL':30s} {eval_learned['per_domain']['code']['ppl']:16.1f} {eval_profile['per_domain']['code']['ppl']:16.1f}
    {'Math PPL':30s} {eval_learned['per_domain']['math']['ppl']:16.1f} {eval_profile['per_domain']['math']['ppl']:16.1f}
    {'Stories PPL':30s} {eval_learned['per_domain']['stories']['ppl']:16.1f} {eval_profile['per_domain']['stories']['ppl']:16.1f}
    {'Wiki PPL':30s} {eval_learned['per_domain']['wiki']['ppl']:16.1f} {eval_profile['per_domain']['wiki']['ppl']:16.1f}
    {'Speed (ms)':30s} {ms_learned:16.2f} {ms_profile:16.2f}
    {'Speed (tok/s)':30s} {tps_learned:16.0f} {tps_profile:16.0f}
    {'Router params':30s} {learned_router_params:16d} {profile_router_params:16d}
    {'Swappable?':30s} {'NO':>16s} {'YES':>16s}
    {'Train time (s)':30s} {train_time_learned:16.1f} {train_time_profile:16.1f}
    """)

    # ── Export JSON ──
    import json
    export = {
        'learned_router': {
            'overall_ppl': float(eval_learned['overall_ppl']),
            'per_domain': {d: {'ppl': float(s['ppl']), 'n': s['n_tokens']}
                          for d, s in eval_learned['per_domain'].items()},
            'speed_ms': float(ms_learned),
            'tokens_per_sec': float(tps_learned),
            'router_params': learned_router_params,
            'train_time_s': float(train_time_learned),
        },
        'profile_router': {
            'overall_ppl': float(eval_profile['overall_ppl']),
            'per_domain': {d: {'ppl': float(s['ppl']), 'n': s['n_tokens']}
                          for d, s in eval_profile['per_domain'].items()},
            'speed_ms': float(ms_profile),
            'tokens_per_sec': float(tps_profile),
            'router_params': profile_router_params,
            'profiler_params': profile_profiler_params,
            'train_time_s': float(train_time_profile),
            'swappable': True,
        },
        'swap_test': {
            'code_before': float(eval_profile['per_domain']['code']['ppl']),
            'code_after': float(eval_swapped['per_domain']['code']['ppl']),
        },
        'config': {
            'd_model': 64, 'n_heads': 2, 'n_layers': 2,
            'n_experts': n_domains, 'd_ff': 128, 'seq_len': 64,
            'vocab_size': vocab_size, 'epochs': 8,
        }
    }
    import os as _tos
    _tHERE = _tos.path.dirname(_tos.path.abspath(__file__))
    with open(_tos.path.join(_tHERE, "transformer_results.json"), "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Results exported: transformer_results.json")

    return model_learned, model_profile, eval_learned, eval_profile


if __name__ == '__main__':
    main()
