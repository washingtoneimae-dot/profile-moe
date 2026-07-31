# DeepSeek MoE Architecture Reference

## Key Papers
1. **DeepSeekMoE** (Jan 2024) — arxiv 2401.06066
   - Fine-grained expert segmentation + shared expert isolation
   - Foundation for V2/V3 architecture

2. **DeepSeek-V2** (May 2024) — arxiv 2405.04434
   - 236B total, 21B active params
   - Device-limited routing, auxiliary loss for load balance
   - Multi-Head Latent Attention (MLA)

3. **DeepSeek-V3** (Dec 2024) — arxiv 2412.19437
   - 671B total, 37B active params
   - Auxiliary-loss-free load balancing (bias-based)
   - Multi-Token Prediction (MTP)

4. **Auxiliary-Loss-Free Load Balancing** (Aug 2024) — arxiv 2408.15664
   - The bias-based routing strategy used in V3

## Architecture (what we benchmark against)

### Expert Structure
```
DeepSeekMoE FFN layer:
  K_s shared experts (always active, capture common knowledge)
  m×N routed experts (fine-grained, selectively activated)
  
  Per token:
    - All K_s shared experts compute (always)
    - Top m×K routed experts selected via learned router
    - Output = Σ shared_outputs + Σ routed_outputs × gating_weights
```

### Routing
```
Traditional (V2, what we compare):
  s_i = softmax(u_t · W_r)           # learned affinity scores
  top_k = TopK(s_i, m×K)              # select experts
  weights = softmax(s_i[top_k])       # renormalize weights

  Load balance: auxiliary loss L_bal = N · Σ(f_i · P_i)
    where f_i = fraction of tokens to expert i
          P_i = average router probability for expert i

V3 (bias-based, for reference only):
  s'_i = s_i + b_i                    # bias adjusted per step
  top_k = TopK(s'_i, m×K)
  b_i += γ if overloaded, b_i -= γ if underloaded
  (bias NOT used in output weight computation, only for top-K)
```

### Key Differences: DeepSeek MoE vs Profile-MoE

| Aspect | DeepSeek MoE | Profile-MoE |
|--------|-------------|-------------|
| Router params | W_r ∈ R^(d_model × n_experts) — learned | None — pure similarity math |
| Routing signal | Hidden state u_t | Profile vector φ(x) |
| Expert identity | Emergent (from co-training) | Declared (from calibration) |
| Swappable? | No — router co-trained with experts | Yes — profile is the API |
| Load balancing | Auxiliary loss or bias term | Profile-based scheduling |
| New expert cost | Retrain router + rebalance | Recalibrate profile (minutes) |
| Routing speed | O(d_model × n) matmul | O(d_profile × n) cosine sim |
| Shared experts | Yes (K_s always active) | Optional (same concept) |
