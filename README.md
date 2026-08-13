# SurgEffiSkill

Competency-aware mixture-of-experts for surgical skill assessment from laparoscopic video.

Four skill competencies (Depth perception / Bimanual coordination / Efficiency /
Tissue handling) are rated on a 1–5 ordinal scale. The model routes six heterogeneous
evidence experts **per competency**, so that fusion capacity is spent only where the
expert pool actually carries complementary evidence.

## Method

```
s_d(x) = Σ_m α_{d,m}(x) · ŷ_m(x)
α_d(x) = softmax( log π_d + β_d(x) · Δ_d(u_d(x)) )
```

| Component | Definition |
|---|---|
| `π_d` | Per-competency global evidence prior; exhaustive simplex search (step 0.1) on **training-fold out-of-fold** predictions, optimising QWK directly |
| `Δ_d(u)` | Low-rank per-case residual, `tanh(Vu) @ U_d` (V shared 13→rank 2; U_d per-competency 2×6), **zero-initialised** so epoch 0 is bit-identical to the static solution |
| `β_d(x)` | Trigger gate `σ(a_dᵀu + b_d)` |
| `u_d(x)` | **13 scalar routing signals** — expert predictions, inter-expert std, max−min, SFT-vs-ridge gap, distances to the four ordinal thresholds |
| Decoding | Fixed ordinal thresholds `[1.5, 2.5, 3.5, 4.5]`, **no post-hoc calibration** |

Six experts: two ViT streams (28-frame, 4-fps), un-finetuned LLM semantics, SFT-LLM
semantics, a joint concatenation, and the SFT model's own end-to-end predictions.

**Verifiable fallback property.** On competencies where the pool carries no
complementary evidence, the gate provably collapses to a single expert (alpha = 1.0)
and introduces no degradation. On the released split this happens for Depth perception
and Bimanual coordination; fusion capacity is used only on Efficiency and Tissue
handling.

## Layout

| Path | Contents |
|---|---|
| `surgeffiskill/core/` | The method and its evaluation entry point |
| `surgeffiskill/pipeline/` | Split construction, feature caching/pooling, SFT inference |
| `surgeffiskill/ordinal_metrics/` | Ordinal decoding and metrics (QWK, ±1, macro-F1, MAE, rho) |
| `analysis/` | Scripts reproducing the paper's diagnostics and negative results |
| `docs/` | Final report and the full research changelog |

## Reproducing

```bash
python surgeffiskill/pipeline/make_splits82.py          # 8:2 stratified split
python surgeffiskill/pipeline/cache_visual_tokens.py    # frozen ViT tokens
python surgeffiskill/pipeline/cache_llm_tokens.py       # LLM hidden states
python surgeffiskill/pipeline/pool_llm_feats.py
python surgeffiskill/pipeline/probe_vision_features.py
python surgeffiskill/pipeline/concat_feats.py           # -> feats_4src.npz
python surgeffiskill/core/final_eval.py --arm hot       # final model
```

All hyper-parameters are selected from training-set cross-validation only. Paths in the
scripts still point at the authors' cluster layout and need adjusting.

## Data

Not included. The surgical videos and their skill annotations are patient data and are
not redistributable. Derived features, model predictions and per-case result files are
likewise excluded (see `.gitignore`).

## Notes on the changelog

`docs/CHANGELOG.md` is the unedited research log, including retracted claims, corrected
analyses and the full set of negative results for competency-adaptive routing. It is
published as-is for transparency rather than as polished documentation.
