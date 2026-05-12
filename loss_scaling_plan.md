# Loss Coefficient Scaling Plan for PIV Gap-Fill VAE

## Background

The current training loop uses:
- **BCE** (velocity MSE on masked/observed pixels): weight = 1.0 (fixed)
- **KLD** (KL divergence): weight = `beta`, linearly annealed from 0 → `max_beta` (default 20.0) over `beta_anneal_epochs` (default 100)
- **VortMSE** (vorticity physics loss): weight = `effective_gamma = gamma_base × (1 − coverage)`, linearly annealed from 0 → `max_gamma` (default 1e-7) over `gamma_anneal_epochs` (default 100)

Total loss:
```
TotalLoss = BCE + β·KLD + γ_eff·VortMSE
```

---

## Recommended 3-Phase Epoch Schedule

### Phase 1 — Reconstruction First (epochs 0 → ~20% of total)
**Priority: BCE only, KLD and VortMSE both near zero**

- `beta` → 0.0 (or a very small warm-start value like 1e-4)
- `gamma_base` → 0.0
- **Rationale**: The encoder/decoder must first learn meaningful spatial structure before KL regularization constrains the latent space. Starting with high KLD causes posterior collapse, which destroys reconstruction quality. VortMSE on a poorly reconstructed velocity field produces noisy, uninformative gradients that can destabilize training.

### Phase 2 — KLD Annealing (epochs ~20% → ~60%)
**Priority: BCE dominant, KLD grows gradually; VortMSE still off**

- `beta`: linearly (or with a sigmoid warm-up) ramp from ~0 → `max_beta`
  - Suggested `max_beta` range: **1.0–5.0** (current default of 20.0 is very aggressive and risks over-regularization; lower values typically improve reconstruction fidelity at the cost of latent space disentanglement)
  - Use a sigmoid schedule `beta(t) = max_beta × σ(k·(t − t_mid))` rather than linear to give a slower warm-up near the transition
- `gamma_base` → 0.0
- **Rationale**: KLD annealing (β-VAE style) is well-established. The latent space needs to be structurally sound before physics losses are imposed on the decoded outputs.

### Phase 3 — Vorticity Physics Loss Activation (epochs ~60% → end)
**Priority: BCE + KLD stable, VortMSE gradually grows**

- `beta`: held at `max_beta` (frozen)
- `gamma_base`: ramp from 0 → `max_gamma` (linear or cosine)
  - Suggested `max_gamma` scale: start at **1e-6 to 1e-5** (current 1e-7 may be too weak to meaningfully influence training; compare the raw magnitude of `VortMSE` vs `BCE` in logs to calibrate)
  - The `effective_gamma = gamma_base × (1 − coverage)` already scales automatically with gap size, which is correct
- **Rationale**: VortMSE computes gradients through the entire velocity field. It should only be activated once the reconstruction is already reasonable, so the vorticity penalty refines physics consistency rather than fighting against an unstable decoder.

---

## Concrete Scheduling Summary

| Phase | Epoch Range (N total) | beta | gamma_base |
|---|---|---|---|
| 1 — Recon only | 0 – 0.20·N | 0 or ~1e-4 | 0 |
| 2 — KLD anneal | 0.20·N – 0.60·N | linear 0 → max_beta (1–5) | 0 |
| 3 — Vort anneal | 0.60·N – 1.0·N | max_beta (frozen) | linear 0 → max_gamma (1e-6 to 1e-5) |

- **Replace the current single linear ramp** (both starting from epoch 0) with the two-stage delayed schedule above.
- **Diagnostic check**: log `BCE`, `KLD`, and `VortMSE` individually every epoch (not just every 50). Compare their raw magnitudes. If `β·KLD >> BCE`, reduce `max_beta`. If `γ·VortMSE << 1% of BCE`, increase `max_gamma` by an order of magnitude.

---

## Key Scaling Principles

1. **BCE scale is the anchor** — always keep it at weight 1.0 and calibrate the others relative to it.
2. **KLD scale** (`max_beta`): aim for `β·KLD ≈ 10–30%` of `BCE` in steady state; values of **1–5** are more typical for reconstruction-focused VAEs than the current 20.
3. **VortMSE scale** (`max_gamma`): aim for `γ_eff·VortMSE ≈ 5–15%` of `BCE`; since vorticity involves spatial derivatives it can have a very different numerical scale — measure it empirically in the logs before fixing `max_gamma`.
4. **Never start KLD and VortMSE simultaneously** — stagger their introduction as described above.
5. **Coverage-adaptive gamma** (current `1 − coverage` factor) is a good design — keep it, but ensure `max_gamma` is large enough that the effective weight is meaningful at typical coverage levels.
